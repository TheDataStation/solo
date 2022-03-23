import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import random
from multiprocessing import Pool as ProcessPool
from table2txt.graph_strategy.rel_template import RelationTemplate 

def read_tables(data_file):
    table_lst = []
    with open(data_file) as f:
        for line in tqdm(f):
            table = json.loads(line)
            table_lst.append(table)
    return table_lst

def init_worker():
    global g_strategy
    g_strategy = RelationTemplate() 

def process_table(table):
    return g_strategy.generate(table)

def main():
    args = get_args()
    table2txt_dir = '/home/cc/code/open_table_discovery/table2txt'
    out_dir = os.path.join(table2txt_dir, 'dataset', args.dataset, args.experiment)
    if os.path.isdir(out_dir):
        err_msg = ('[%s] already exists, please use a different value for [--out_dir].\n'
              % (out_dir))
        print(err_msg)
        return
    os.makedirs(out_dir)
    
    out_graph_file = os.path.join(out_dir, 'template_graph.jsonl')
    f_o = open(out_graph_file, 'w')

    out_file_src = os.path.join(out_dir, 'test_unseen.source')
    out_file_tar = os.path.join(out_dir, 'test_unseen.target')
    f_o_src = open(out_file_src, 'w')
    f_o_tar = open(out_file_tar, 'w')

    table_file_name = args.table_file
    input_table_file = os.path.join('/home/cc/data', args.dataset, 'tables', table_file_name)
    table_lst = read_tables(input_table_file)

    multi_process = True
    if multi_process:
        work_pool = ProcessPool(initializer=init_worker)
        for graph_lst in tqdm(work_pool.imap_unordered(process_table, table_lst), total=len(table_lst)):
            write_graphs(graph_lst, f_o, f_o_src, f_o_tar) 
             
    else:
        init_worker()
        for table in tqdm(table_lst):
            graph_lst = process_table(table)
            write_graphs(graph_lst, f_o) 

    f_o.close()
    f_o_src.close()
    f_o_tar.close() 

def write_graphs(graph_lst, f_o, f_o_src, f_o_tar):
    for graph_info in graph_lst:
        f_o_src.write(graph_info['graph'] + '\n')
        f_o_tar.write('a\n')
        f_o.write(json.dumps(graph_info) + '\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--table_file', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--debug', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()


