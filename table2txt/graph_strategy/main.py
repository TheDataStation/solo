import json
import os
import random
import argparse
import shutil
from table2txt.graph_strategy.strategy import Strategy
from table2txt.graph_strategy.question_generator import QG
import pytorch_lightning as pl
from webnlg.finetune import main as generate_text_from_graph, SummarizationModule
from fabric_qa.reader.forward_reader.rc_model import get_rc_model as get_reader
from fabric_qa.reader.forward_reader.albert.qa_data import data_to_examples
from tqdm import tqdm
from fabric_qa import utils
import numpy as np
from strategy_constructor import get_strategy_lst 
import time
import pandas as pd

def get_row_index_lst(row_lst):
    row_idx_lst = []
    for row_idx, row_data in enumerate(row_lst):
        cell_lst = row_data['cells']
        non_empty_cells =  [a for a in cell_lst if a['text'].strip() != '']
        if len(non_empty_cells) > 1:
            row_idx_lst.append(row_idx)
    return row_idx_lst

def read_tables(data_file):
    table_lst = []
    M = 3
    with open(data_file) as f:
        for line in f:
            table = json.loads(line)
            row_lst = table['rows']
           
            row_idx_lst = get_row_index_lst(row_lst)
            if len(row_idx_lst) < 1:
                continue
             
            num_rows = len(row_idx_lst)
            num_sample = min(M, num_rows)
            sample_row_idxes = random.sample(row_idx_lst, num_sample)
            sample_rows = [row_lst[a] for a in sample_row_idxes]
            table['rows'] = sample_rows

            table_info = {
                'table':table,
                'row_idxes':sample_row_idxes
            }
            table_lst.append(table_info)
    return table_lst

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_tables', type=str)
    parser.add_argument('--data_tag', type=str)
    parser.add_argument('--dataset_in_dir', type=str)
    parser.add_argument('--dataset_out_dir', type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    return args

def evaluate_strategy(reader, qa_lst, table, stg, args):
    graph_file_info = prepare_graph_file(table, stg, args)
   
    table_code = table['_table_code_'] 
    
    generate_graph(table, stg, args, graph_file_info)

    args.data_dir = graph_file_info['data_dir']
    table_output_dir = os.path.join(args.dataset_out_dir, table['_table_code_'])
    if not os.path.exists(table_output_dir):
        os.makedirs(table_output_dir)
    args.output_dir = os.path.join(table_output_dir, stg.name) 
   
    t1 = time.time()
    table_log_info = {
        'table_code':table_code,
        'used_t_lst':used_t_lst
    }
    args.table_log_info = table_log_info
    generate_text_from_graph(args)
    t2 = time.time()
    
    used_t = t2 - t1
    log_time_info(table_code, 'Generate_text_from_graph', used_t)
    
    t1 = time.time() 
    table_passage_dict = read_passages(args, stg, graph_file_info)
    mean_f1 = evaluate_question(reader, qa_lst, table, table_passage_dict, stg.name)    
    t2 = time.time()
    
    used_t = t2 - t1
    log_time_info(table_code, 'Reader_answer_questions', used_t)
    
    return mean_f1

def get_answer_lst(reader, batch_examples):
    batch_size = 16
    N = len(batch_examples)
    answer_lst = [] 
    for i in range(0, N, batch_size):
        example_lst = batch_examples[i:(i+batch_size)]
        reader_out = reader(example_lst)
        preds = reader_out['preds']
        for example in example_lst:
            item = preds[example.qas_id]
            answer = item['text']
            answer_lst.append(answer)
    
    return answer_lst 

def evaluate_question(reader, qa_lst, table, table_passage_dict, stg_name):
    table_code = table['_table_code_']
    table_id = table['tableId']
    pred_f1_lst = []
    for row, row_questions in tqdm(enumerate(qa_lst), total=len(qa_lst)):
        pasage_key = '%s-%d' % (table_id, row)
        row_passages = table_passage_dict[pasage_key]
        passage_num = len(row_passages)
        p_id_lst = [a for a in range(passage_num)]
        for sub_idx, question_info in tqdm(enumerate(row_questions), total=len(row_questions)):
            qid = '%s-%d-%d' % (table_id, row, sub_idx)
            question = question_info['question']
            passage_num = len(row_passages)
            p_id_lst = [a for a in range(passage_num)]
            q_p_batch = [
                {'question':question, 'passages':row_passages, 'qid':qid, 'p_id_lst':p_id_lst}
            ]
            batch_examples = data_to_examples(q_p_batch)
            answer_lst = get_answer_lst(reader, batch_examples)
            gold_answer = question_info['answer']
            _, f1s = utils.compute_em_f1(answer_lst, [gold_answer])
            best_idx = np.argmax(f1s)
            best_f1 = f1s[best_idx]
            best_passage = row_passages[best_idx]
            best_answer = answer_lst[best_idx]

            log_info = {
                'table_code':table_code,
                'table_id':table_id,
                'row':row,
                'question':question_info['question'],
                'gold answer':gold_answer,
                'strategy':stg_name,
                'passage':best_passage,
                'pred answer':best_answer,
                'f1':best_f1
            }
            log_info_lst.append(log_info)
            pred_f1_lst.append(best_f1)
            
    mean_f1 = np.mean(pred_f1_lst)
    return mean_f1 

def read_meta_info(meta_file):
    meta_info_lst = []
    with open(meta_file) as f:
        for line in f:
            meta_info = json.loads(line)
            meta_info_lst.append(meta_info)
    return meta_info_lst    

def get_passage(table_id, graph_text, graph_tokens):
    table_id_updated = table_id.replace('_', ' ').replace('-', ' ')

    graph_tokens_updated = graph_tokens.replace('<H>', ' ').replace('<R>', ' ').replace('<T>', ' , ')

    passage = table_id_updated + ' . ' + graph_tokens_updated + ' (). ' + graph_text
    return passage

def get_graph_tokens(data_file):
    graph_tokens_lst = []
    with open(data_file) as f:
        for row, line in enumerate(f):
            graph_tokens = line.rstrip()
            graph_tokens_lst.append(graph_tokens)
    return graph_tokens_lst

def read_passages(args, stg, graph_file_info):
    out_passage_file = os.path.join(args.output_dir, 'val_outputs/test_unseen_predictions.txt.debug')
    table_passage_dict = {}
    
    meta_info_lst = read_meta_info(graph_file_info['meta_file'])
    
    graph_tokens_lst = get_graph_tokens(graph_file_info['src_file'])

    with open(out_passage_file) as f:
        for row, text in enumerate(f):
            meta_info = meta_info_lst[row]
            table_id = meta_info['table_id']
            table_title = meta_info['table_title']
            table_row = meta_info['row']

            passage_key = '%s-%d' % (table_id, table_row)
            if passage_key not in table_passage_dict:
                table_passage_dict[passage_key] = []
            passage_lst = table_passage_dict[passage_key]
            graph_text = text.rstrip()
            graph_tokens = graph_tokens_lst[row]
            passage = get_passage(table_id, graph_text, graph_tokens)
            passage_lst.append(passage)
    return table_passage_dict
     
def generate_graph(table, stg, args, graph_file_info):
    f_o_src = open(graph_file_info['src_file'], 'w')
    f_o_tgt = open(graph_file_info['tgt_file'], 'w')
    f_o_meta = open(graph_file_info['meta_file'], 'w')
    _, graph_lst = stg.generate(table) 
    for graph_info in graph_lst:
        f_o_src.write(graph_info['graph'] + '\n')
        f_o_tgt.write('a' + '\n')
        meta_info = {
            'table_id':table['tableId'],
            'table_title':table['documentTitle'],
            'row':graph_info['row']
        }
        f_o_meta.write(json.dumps(meta_info) + '\n')

    f_o_src.close()
    f_o_tgt.close()
    f_o_meta.close()

def prepare_graph_file(table, stg, args):
    table_input_dir = os.path.join(args.dataset_in_dir, table['_table_code_'])
    if not os.path.exists(table_input_dir):
        os.makedirs(table_input_dir)
    
    graph_folder = os.path.join(table_input_dir, stg.name)
    shutil.copytree('/home/cc/code/plms_graph2text/webnlg/data/webnlg/template', graph_folder)
    source_file = os.path.join(graph_folder, 'test_unseen.source')
    target_file = os.path.join(graph_folder, 'test_unseen.target')
    meta_file = os.path.join(graph_folder, 'test_unseen.table_meta')
    graph_file_info = {
        'src_file':source_file,
        'tgt_file':target_file,
        'meta_file':meta_file,
        'data_dir':graph_folder
    }
    return graph_file_info

def output_tables(table_lst):
    out_dir = './output'
    for table in table_lst:
        table_file = os.path.join(out_dir, table['tableId']+'.json')
        with open(table_file, 'w') as f_o:
            f_o.write(json.dumps(table))

def log_time_info(table_code, component, seconds):
    item = {
        'table_code':table_code,
        'component':component,
        'seconds':seconds
    }
    used_t_lst.append(item)
 
def main():
    #random.seed(10)
    args = get_args()
    if os.path.exists(args.dataset_in_dir):
       raise ValueError('[%s] already exists.' % args.dataset_in_dir)
    global used_t_lst
    used_t_lst = []
    global log_info_lst
    log_info_lst = []
    stg_lst = get_strategy_lst()
    table_lst = read_tables(args.input_tables)
    reader = get_reader('albert',
                        '/home/cc/code/fabric_qa/model/reader/forward_reader/model',
                        0)
    q_generator = QG(random)
    report_data = []
    table_best_stg_file = './output/%s_best_stg.jsonl' % args.data_tag 
    f_o_best_stg = open(table_best_stg_file, 'w')
    sample_row_dict = {}
    table_code_lst = []
    for table_no, table_info in tqdm(enumerate(table_lst), total=len(table_lst)):
        
        t1 = time.time()

        table = table_info['table']
        table['_table_code_'] = 'table_%d' % (table_no + 1)
        
        table_code_lst.append(table['_table_code_'])

        print('\n\n%s' % table['_table_code_'])
        sample_row_dict[table['tableId']] = table_info['row_idxes']
        qa_lst = q_generator.generate(table)
        
        stg_f1_lst = []
        for stg_idx, stg in tqdm(enumerate(stg_lst), total=len(stg_lst)):
            mean_f1 = evaluate_strategy(reader, qa_lst, table, stg, args)
            stg_f1_lst.append(mean_f1)
            report_item =  [table['tableId'], stg.name, round(mean_f1 * 100, 2)]
            report_data.append(report_item)
        
        best_stg_idx = np.argmax(stg_f1_lst)
        best_stg_info = {
            'table_id':table['tableId'],
            'strategy':stg_lst[best_stg_idx].name
        }
        f_o_best_stg.write(json.dumps(best_stg_info) + '\n')
        
        t2 = time.time()
        used_t = t2 - t1

        log_time_info(table['_table_code_'], 'Total', used_t)
   
    f_o_best_stg.close() 
    print('\n')
    report_cols = ['Table', 'Strategy', 'F1']
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(report_data, columns=report_cols)
    print(df)
    print('\n')
    for log_info in log_info_lst:
        log_info['row'] = sample_row_dict[log_info['table_id']][log_info['row']] + 1
    df_log = pd.DataFrame(log_info_lst)
    df_log.to_csv('./output/%s_strategy_log.csv' % args.data_tag)
    show_breakdown(args.data_tag, table_code_lst)

def index_time_used():
    used_info_by_table = {}
    for item in used_t_lst:
        table_code = item['table_code']
        if table_code not in used_info_by_table:
           used_info_by_table[table_code] = {}
        
        detail_info = used_info_by_table[table_code]
        detail_info[item['component']] = item
    return used_info_by_table  

def show_breakdown(data_tag, table_code_lst):
    used_info_by_table = index_time_used() 
    for table_code in used_info_by_table:
        detail_info = used_info_by_table[table_code]
        total_seconds = detail_info['Total']['seconds']
        for component in detail_info:
            item = detail_info[component]
            if component != 'Total':
                item['percentage'] = (item['seconds'] / total_seconds) * 100
            else:
                item['percentage'] = 100
    
    sort_used_lst(table_code_lst)  
    df_used = pd.DataFrame(used_t_lst)
    pd.options.display.float_format = '{:,.2f}'.format
    print(df_used)
    to_csv(data_tag)

def to_csv(data_tag):
    with open('./output/%s_perf.csv' % data_tag, 'w') as f_o:
        f_o.write('Table Code, Component, Seconds, Percentage, Order\n')
        for item in used_t_lst:
            item_str = '%s, "%s", %.2f, %.2f, %d\n' % (
                item['table_code'], item['component'], item['seconds'], item['percentage'], item['order']
            )
            f_o.write(item_str)


def sort_used_lst(table_code_lst):
    global used_t_lst 
    for item in used_t_lst:
        component = item['component']
        order = 0
        if component == 'Total':
            order = 6
        elif component == 'Generate_text_from_graph':
            order = 1
            item['component'] = '1. %s' % component
        elif component == 'Graph2text before test':
            order = 2
            item['component'] = '      1.1. %s' % component
        elif component == 'Graph2text test':
            order = 3
            item['component'] = '      1.2. %s' % component
        elif component.startswith('Graph2text test generate by batch'):
            order = 4
            item['component'] = '            1.2.x %s' % component
        elif component == 'Reader_answer_questions':
            order = 5
            item['component'] = '2. %s' % component
        item['order'] = order
    used_t_lst_tmp = sorted(used_t_lst, key = lambda x: (x['table_code'], x['order']))
    
    item_dict = {}
    for item in used_t_lst_tmp:
        table_code = item['table_code']
        if table_code not in item_dict:
            item_dict[table_code] = []
        
        item_lst = item_dict[table_code]
        item_lst.append(item)
    
    used_t_lst = []
    for table_code in table_code_lst:
        item_lst = item_dict[table_code]
        used_t_lst.extend(item_lst)  


if __name__ == '__main__':
    main()
