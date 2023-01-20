import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import random
from multiprocessing import Pool as ProcessPool
from table2txt.graph_strategy.strategy_constructor import get_strategy
import shutil

def count_tables(data_file):
    count = 0
    with open(data_file) as f:
        for line in f:
            count += 1
    return count

def read_tables(data_file, table_count, batch_size):
    table_lst = []
    num_batch = int(table_count / batch_size) + (1 if (table_count % batch_size) > 0 else 0)
    with open(data_file) as f:
        for line in tqdm(f, total=num_batch):
            table = json.loads(line)
            table_lst.append(table)
            if len(table_lst) >= batch_size:
                yield table_lst
                table_lst = []
    if len(table_lst) > 0:
        yield table_lst

def init_worker(strategy_name):
    global g_strategy
    g_strategy = get_strategy(strategy_name) 

def process_table(arg_info):
    table = arg_info['table']
    out_part_dir = arg_info['out_part_dir']
    file_name = '%s_graph.jsonl' % table['tableId']
    out_file = os.path.join(out_part_dir, file_name)
    with open(out_file, 'w') as f_o:  
        for graph_info in g_strategy.generate(table):
            write_graphs([graph_info], f_o) 
    return out_file

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
    table_count = count_tables(input_table_file)

    out_part_dir = os.path.join(out_dir, 'table_parts')
    if os.path.isdir(out_part_dir):
        shutil.rmtree(out_part_dir)
    os.makedirs(out_part_dir)
    
    out_file_lst = []
    batch_size = 10
    multi_process = True
    work_pool = None
    if multi_process:
        work_pool = ProcessPool(initializer=init_worker, initargs=(args.strategy,))
    else:
        init_worker(args.strategy)
    
    for batch_table_lst in read_tables(input_table_file, table_count, batch_size):
        arg_info_lst = []
        for table in batch_table_lst:
            args_info = {
                'table':table,
                'out_part_dir':out_part_dir
            }
            arg_info_lst.append(args_info)
    
        if multi_process:
            for out_graph_file in tqdm(work_pool.imap_unordered(process_table, arg_info_lst), total=len(arg_info_lst)):
                out_file_lst.append(out_graph_file)
        else:
            for arg_info in arg_info_lst:
                out_graph_file = process_table(arg_info)
                out_file_lst.append(out_graph_file)
   
    print('merging graph files') 
    merge_graph_files(out_file_lst, f_o) 
    shutil.rmtree(out_part_dir)
    f_o.close()
    
    msg_info = {
        'state':True,
        'out_file':out_passage_file
    }
    return msg_info 

g_passage_id = 0

def merge_graph_files(graph_file_list, f_merge):
    for graph_file in graph_file_list:
        with open(graph_file) as f_graph:
            for line in f_graph:
                f_merge.write(line)
        os.remove(graph_file)

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


