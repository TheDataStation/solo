import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import random
from multiprocessing import Pool as ProcessPool
from table2txt.graph_strategy.strategy_constructor import get_strategy
import shutil
import copy
import uuid

def count_batchs(data_file, batch_size, chunk_size):
    num_batch = 0
    for _ in read_tables(data_file, batch_size, chunk_size):
        num_batch += 1
    return num_batch

def read_tables(data_file, batch_size, chunk_size):
    table_lst = []
    with open(data_file) as f:
        for line in tqdm(f):
            table = json.loads(line)
            for sub_table in chunk_tables(table, chunk_size):
                table_lst.append(sub_table)
                if len(table_lst) >= batch_size:
                    yield table_lst
                    table_lst = []

    if len(table_lst) > 0:
        yield table_lst

def chunk_tables(table, chunk_size):
    row_data = table['rows']
    table['row_start_offset'] = 0
    if len(table['rows']) <= chunk_size:
        yield table
    else:
        table['rows'] = []
        for offset in range(0, len(row_data), chunk_size):
            sub_table = copy.deepcopy(table)
            sub_table['row_start_offset'] = offset
            sub_table['rows'] = row_data[offset:(offset+chunk_size)]
            yield sub_table

def init_worker(strategy_name):
    global g_strategy
    g_strategy = get_strategy(strategy_name) 

def process_table(arg_info):
    table = arg_info['table']
    table['show_progress'] = (arg_info['task_idx'] == 0)
    out_part_dir = arg_info['out_part_dir']
    file_name = 'triple_%s.jsonl' % str(uuid.uuid4())
    out_file = os.path.join(out_part_dir, file_name)
    with open(out_file, 'w') as f_o: 
        for graph_info in g_strategy.generate(table):
            row_offset = graph_info['row']
            graph_info['row'] = row_offset + table['row_start_offset']
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
    
    num_batch = count_batchs(input_table_file, args.table_import_batch, args.table_chunk_size)

    out_part_dir = os.path.join(out_dir, 'table_parts')
    if os.path.isdir(out_part_dir):
        shutil.rmtree(out_part_dir)
    os.makedirs(out_part_dir)
    
    out_file_lst = []
    multi_process = True
    work_pool = None
    if multi_process:
        work_pool = ProcessPool(initializer=init_worker, initargs=(args.strategy,))
    else:
        init_worker(args.strategy)
    
    for batch_table_lst in tqdm(read_tables(input_table_file, args.table_import_batch, args.table_chunk_size), 
                                total=num_batch):
        arg_info_lst = []
        for task_idx, table in enumerate(batch_table_lst):
            args_info = {
                'table':table,
                'out_part_dir':out_part_dir,
                'task_idx':task_idx
            }
            arg_info_lst.append(args_info)
    
        if multi_process:
            for out_graph_file in work_pool.imap_unordered(process_table, arg_info_lst):
                out_file_lst.append(out_graph_file)
        else:
            for arg_info in arg_info_lst:
                out_graph_file = process_table(arg_info)
                out_file_lst.append(out_graph_file)
   
    num_triples = merge_graph_files(out_file_lst, f_o) 
    shutil.rmtree(out_part_dir)
    f_o.close()
    
    msg_info = {
        'state':True,
        'out_file':out_passage_file,
        'num_triples':num_triples
    }
    return msg_info 

g_passage_id = 0

def merge_graph_files(graph_file_list, f_merge):
    g_p_id = 0
    for graph_file in tqdm(graph_file_list, desc='merging triples'):
        with open(graph_file) as f_graph:
            for line in f_graph:
                triple_info = json.loads(line)
                g_p_id += 1
                triple_info['p_id'] = g_p_id
                f_merge.write(json.dumps(triple_info) + '\n')
        os.remove(graph_file)
    num_total_triples = g_p_id
    return num_total_triples

def write_graphs(graph_lst, f_o):
    for graph_info in graph_lst:
        passage = graph_info['graph']
        meta_info = {
            'table_id': graph_info['table_id'],
            'row': graph_info['row'],
            'sub_col':graph_info['sub_col'],
            'obj_col':graph_info['obj_col']
        }
        passage_info = {
            'p_id':None, #update when merging
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
    parser.add_argument('--table_import_batch', type=int)
    parser.add_argument('--table_chunk_size', type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)


