import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import random
from multiprocessing import Pool as ProcessPool
from table2txt.graph_strategy.strategy_constructor import get_strategy

def read_tables(data_file):
    table_lst = []
    with open(data_file) as f:
        for line in tqdm(f):
            table = json.loads(line)
            table_lst.append(table)
    return table_lst

def init_worker(strategy_name):
    global g_strategy
    g_strategy = get_strategy(strategy_name) 

def process_table(table):
    return g_strategy.generate(table)

def main(args):
    table2txt_dir = os.path.join(args.work_dir, 'open_table_discovery/table2txt')
    out_dir = os.path.join(table2txt_dir, 'dataset', args.dataset, args.experiment)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_passage_file = os.path.join(out_dir, 'passages.jsonl')
    if os.path.exists(out_passage_file):
        err_msg = ('(%s) already exists.\n' % out_passage_file)
        print(err_msg)
        return {'state':False}
    f_o = open(out_passage_file, 'w')

    table_file_name = args.table_file
    input_table_file = os.path.join(args.work_dir, 'data', args.dataset, 'tables', table_file_name)
    table_lst = read_tables(input_table_file)

    multi_process = False
    if multi_process:
        work_pool = ProcessPool(initializer=init_worker, initargs=(args.strategy,))
        for graph_lst in tqdm(work_pool.imap_unordered(process_table, table_lst), total=len(table_lst)):
            write_graphs(graph_lst, f_o) 
             
    else:
        init_worker(args.strategy)
        for table in tqdm(table_lst):
            graph_lst = process_table(table)
            write_graphs(graph_lst, f_o) 

    f_o.close()
    
    msg_info = {
        'state':True,
        'out_file':out_passage_file
    }
    return msg_info 

g_passage_id = 0

def write_graphs(graph_lst, f_o):
    global g_passage_id
    for graph_info in graph_lst:
        g_passage_id += 1
        passage = graph_info['graph']
        meta_info = {
            'table_id': graph_info['table_id'],
            'row': graph_info['row'],
            'sub_col':graph_info['sub_col'],
            'obj_col':graph_info['obj_col']
        }
        passage_info = {
            'p_id':g_passage_id,
            'passage':passage,
            'tag':meta_info    
        }
        f_o.write(json.dumps(passage_info) + '\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--table_file', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--debug', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)


