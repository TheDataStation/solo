import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import random

from table2txt.graph_strategy.strategy_constructor import get_strategy

def read_table_file(table_lst, data_file, table_filter_set):
    with open(data_file) as f:
        for line in tqdm(f):
            table = json.loads(line)
            table_id = table['tableId']
            if table_filter_set is not None:
                if table_id not in table_filter_set:
                    continue
            table_lst.append(table)
    return table_lst

def read_table_filter_set(table_fileter_file):
    table_id_lst = []
    with open(table_fileter_file) as f:
        for line in f:
            table_id = line.strip()
            table_id_lst.append(table_id)
    table_id_set = set(table_id_lst)
    return table_id_set

def read_tables(table_file, table_filter):
    table_lst = []
    table_filter_set = None
    if table_filter is not None:
        table_filter_set = read_table_filter_set(table_filter)
    read_table_file(table_lst, table_file, table_filter_set)
    return table_lst

def process_table(table, strategy):
    return strategy.generate(table)

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
    out_file_src = os.path.join(out_dir, 'test_unseen.source')
    out_file_tar = os.path.join(out_dir, 'test_unseen.target')
    f_o_src = open(out_file_src, 'w')
    f_o_tar = open(out_file_tar, 'w')
    out_row_table_file = os.path.join(out_dir, 'graph_row_table.txt')
    f_o_table = open(out_row_table_file, 'w')

    table_file_name = 'tables.jsonl'
    if args.table_file is not None:
        table_file_name = ('%s.jsonl' % args.table_file)

    input_tables = os.path.join('/home/cc/data', args.dataset, 'tables', table_file_name)
    table_lst = read_tables(input_tables, None)

    strategy = get_strategy(args.strategy)
    for table in tqdm(table_lst):
        _, graph_lst = process_table(table, strategy)
        table_id = table['tableId']
        for graph_info in graph_lst:
            f_o_src.write(graph_info['graph'] + '\n')
            f_o_tar.write('a\n')
            f_o_table.write(table_id + '\n')
        
    f_o_src.close()
    f_o_tar.close()
    f_o_table.close()  

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--table_file', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--debug', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()


