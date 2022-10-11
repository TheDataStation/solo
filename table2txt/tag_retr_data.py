import json
import random
import os
from tqdm import tqdm
import csv
import argparse
from table2txt.retr_utils import process_train, process_dev

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--table_expr', type=str)
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--top_n', type=int, default=100)
    parser.add_argument('--min_tables', type=int, default=5)
    args = parser.parse_args()
    return args

def get_data_dir(args):
    if args.mode == 'test':
        data_dir = '/home/cc/code/data/%s/query/%s/%s' % (args.dataset, args.mode, args.table_expr) 
    else:
        dataset_dir = '/home/cc/code/open_table_discovery/table2question/dataset/'
        data_dir = os.path.join(dataset_dir, '%s/sql_data/%s/%s' % (args.dataset, args.mode, args.table_expr))
    return data_dir

def get_out_file(args):
    data_dir = get_data_dir(args)
    out_file = os.path.join(data_dir, 'fusion_retrieved_tagged.jsonl')
    return out_file 

def print_args(args):
    data = vars(args)
    str_info = 'args ('
    for name in data:
        str_info += f' {name}={data[name]} ,'
    str_info = str_info[:-1] + ')'
    print(str_info) 

def main():
    args = get_args()
    out_file = get_out_file(args)
    if os.path.exists(out_file):
        print('%s already exists' % out_file)
        return
    print_args(args)
    output_data(args, out_file)

def output_data(args, out_dev_file):
    data_dir = get_data_dir(args)
    data_file = os.path.join(data_dir, 'fusion_retrieved.jsonl')
    retr_data = []
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            retr_data.append(item)
  
    table_dict = read_tables(args) 
   
    process_func = None 
    if args.mode.startswith('train'):
        process_func = process_train
    elif args.mode == 'dev':
        process_func = process_dev
    elif args.mode == 'test':
        process_func = process_dev
    else:
        raise ValueError('Unknown mode (%s)' % args.mode)

    updated_retr_data = process_func(retr_data, args.top_n, table_dict, args.strategy, args.min_tables)
    write_data(updated_retr_data, out_dev_file)

def read_tables(args):
    table_file = '/home/cc/code/data/%s/tables/tables.jsonl' % args.dataset
    table_dict = {}
    with open(table_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['tableId']
            table_dict[table_id] = item
    return table_dict

def write_data(data, out_file):
    with open(out_file, 'w') as f:
        for item in tqdm(data):
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    main()

