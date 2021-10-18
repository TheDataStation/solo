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
import pandas as pd

def read_tables(data_file):
    table_lst = []
    M = 3
    with open(data_file) as f:
        for line in f:
            table = json.loads(line)
            row_lst = table['rows']
            num_rows = len(row_lst)
            num_sample = min(M, num_rows)
            sample_rows = random.sample(row_lst, num_sample)
            table['rows'] = sample_rows 
            table_lst.append(table)
    return table_lst

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_tables', type=str)
    parser.add_argument('--dataset_in_dir', type=str)
    parser.add_argument('--dataset_out_dir', type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    return args


def evaluate_strategy(reader, qa_lst, table, stg, args, graph_file_info):
    generate_graph(table, stg, args, graph_file_info)

    args.data_dir = graph_file_info['data_dir']
    args.output_dir = os.path.join(args.dataset_out_dir, stg.name) 
    generate_text_from_graph(args)
    
    table_passage_dict = read_passages(args, stg, graph_file_info)
    table_id = table['tableId']
    mean_f1 = evaluate_question(reader, qa_lst, table_id, table_passage_dict)    
    return mean_f1

def evaluate_question(reader, qa_lst, table_id, table_passage_dict):
    pred_f1_lst = []
    for row, row_questions in enumerate(qa_lst):
        pasage_key = '%s-%d' % (table_id, row)
        row_passages = table_passage_dict[pasage_key]
        passage_num = len(row_passages)
        p_id_lst = [a for a in range(passage_num)]
        for sub_idx, question_info in enumerate(row_questions):
            q_p_batch = [{
                'question':question_info['question'],
                'passages':row_passages,
                'qid':'%s-%d-%d' % (table_id, row, sub_idx),
                'p_id_lst':p_id_lst
            }]
            batch_examples = data_to_examples(q_p_batch)
            reader_out = reader(batch_examples)
            preds = reader_out['preds']
            answer_lst = []
            for example in batch_examples:
                item = preds[example.qas_id]
                answer = item['text']
                answer_lst.append(answer)
            gold_answer = question_info['answer']
            _, f1s = utils.compute_em_f1(answer_lst, [gold_answer])
            best_f1 = max(f1s)
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

def read_passages(args, stg, graph_file_info):
    out_passage_file = os.path.join(args.output_dir, 'val_outputs/test_unseen_predictions.txt.debug')
    table_passage_dict = {}
    
    f_o_meta = graph_file_info['f_o_meta']
    meta_info_lst = read_meta_info(f_o_meta.name)

    with open(out_passage_file) as f:
        for row, text in enumerate(f):
            meta_info = meta_info_lst[row]
            table_id = meta_info['table']
            table_row = meta_info['row']

            passage_key = '%s-%d' % (table_id, table_row)
            if passage_key not in table_passage_dict:
                table_passage_dict[passage_key] = []
            passage_lst = table_passage_dict[passage_key]
            passage_lst.append(text.rstrip())
    return table_passage_dict
     
def generate_graph(table, stg, args, graph_file_info):
    f_o_src = graph_file_info['f_o_src']
    f_o_tgt = graph_file_info['f_o_tgt']
    f_o_meta = graph_file_info['f_o_meta']
    _, graph_lst = stg.generate(table) 
    for graph_info in graph_lst:
        f_o_src.write(graph_info['graph'] + '\n')
        f_o_tgt.write('a' + '\n')

        meta_info = {
            'table':table['tableId'],
            'row':graph_info['row']
        }
        f_o_meta.write(json.dumps(meta_info) + '\n')

    f_o_src.flush()
    f_o_tgt.flush()
    f_o_meta.flush()

def prepare_graph_file(stg, args):
    graph_folder = os.path.join(args.dataset_in_dir, stg.name)
    if os.path.exists(graph_folder):
       raise ValueError('[%s] already exists.' % graph_folder)
    shutil.copytree('/home/cc/code/plms_graph2text/webnlg/data/webnlg/template', graph_folder)
    source_file = os.path.join(graph_folder, 'test_unseen.source')
    target_file = os.path.join(graph_folder, 'test_unseen.target')
    meta_file = os.path.join(graph_folder, 'test_unseen.table_meta')
    f_o_src = open(source_file, 'w')
    f_o_tgt = open(target_file, 'w')
    f_o_meta = open(meta_file, 'w')
    graph_file_info = {
        'f_o_src':f_o_src,
        'f_o_tgt':f_o_tgt,
        'f_o_meta':f_o_meta,
        'data_dir':graph_folder
    }
    return graph_file_info

def output_tables(table_lst):
    out_dir = './output'
    for table in table_lst:
        table_file = os.path.join(out_dir, table['tableId']+'.json')
        with open(table_file, 'w') as f_o:
            f_o.write(json.dumps(table))

def create_graph_file(stg_lst, args):
    graph_file_info_lst = []
    for stg in stg_lst:
        graph_file_info = prepare_graph_file(stg, args)
        graph_file_info_lst.append(graph_file_info)
    return graph_file_info_lst

def close_graph_file(graph_file_info_lst):
    for graph_file_info in graph_file_info_lst:
        graph_file_info['f_o_src'].close()
        graph_file_info['f_o_tgt'].close()
        graph_file_info['f_o_meta'].close()
 
def main():
    random.seed(10)
    args = get_args()
    stg_lst = get_strategy_lst()
    table_lst = read_tables(args.input_tables)

    output_tables(table_lst)

    reader = get_reader('albert', '/home/cc/code/fabric_qa/model/reader/forward_reader/model', 0)
    q_generator = QG()
    graph_file_info_lst = create_graph_file(stg_lst, args)

    report_data = []

    for table in table_lst:
        qa_lst = q_generator.generate(table)
        for stg_idx, stg in enumerate(stg_lst):
            graph_file_info = graph_file_info_lst[stg_idx]
            mean_f1 = evaluate_strategy(reader, qa_lst, table, stg, args, graph_file_info)
            report_item =  [table['tableId'], stg.name, round(mean_f1 * 100, 2)]
            report_data.append(report_item)  
    
    print('\n')
    report_cols = ['Table', 'Strategy', 'F1']
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(report_data, columns=report_cols)
    print(df)
    print('\n')

    close_graph_file(graph_file_info_lst)        

if __name__ == '__main__':
    main()
