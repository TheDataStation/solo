import json
import os
import random
import argparse
import shutil
from table2txt.graph_strategy.strategy import Strategy
from table2txt.graph_strategy.complete_graph import CompleteGraph
import pytorch_lightning as pl
from webnlg.finetune import main as generate_text_from_graph, SummarizationModule

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

def get_strategy_lst():
    stg_lst = []
    stg_1 = CompleteGraph()
    stg_lst.append(stg_1)
    return stg_lst

def evaluate_strategy(table, stg, args, graph_file_info):
    generate_graph(table, stg, args, graph_file_info)

    args.output_dir = os.path.join(args.dataset_out_dir, stg.name) 
    generate_text_from_graph(args)
    
    table_passage_dict = read_passages(args, stg, graph_file_info)
    

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
    args.data_dir = graph_folder
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
        'f_o_meta':f_o_meta
    }
    return graph_file_info

def output_tables(table_lst):
    out_dir = './output'
    for table in table_lst:
        table_file = os.path.join(out_dir, table['tableId']+'.json')
        with open(table_file, 'w') as f_o:
            f_o.write(json.dumps(table))
     
 
def main():
    random.seed(10)
    args = get_args()
    stg_lst = get_strategy_lst()
    table_lst = read_tables(args.input_tables)

    output_tables(table_lst)

    for stg in stg_lst:
        graph_file_info = prepare_graph_file(stg, args)
        for table in table_lst:
            evaluate_strategy(table, stg, args, graph_file_info)
        graph_file_info['f_o_src'].close()
        graph_file_info['f_o_tgt'].close()
        graph_file_info['f_o_meta'].close()

if __name__ == '__main__':
    main()
