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
    parser.add_argument('--expr', type=str)
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--sql_expr', type=str)
    parser.add_argument('--top_n_train', type=int, default=100)
    parser.add_argument('--top_n_dev', type=int, default=100)
    parser.add_argument('--min_tables', type=int, default=10)
    args = parser.parse_args()
    return args

def get_out_file(args, mode):
    out_dir = os.path.join('dataset', args.dataset, args.sql_expr, args.expr)
    out_file = os.path.join(out_dir, 'fusion_retrieved_%s_tagged.jsonl' % mode)
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
    out_train_file = get_out_file(args, 'train')
    if os.path.exists(out_train_file):
        print('(%s) already exists' % out_train_file)
        return
    out_dev_file = get_out_file(args, 'dev')
    if os.path.exists(out_dev_file):
        print('(%s) already exists' % out_dev_file)
        return
    print_args(args)
    output_data(args, out_train_file, out_dev_file)

def output_data(args, out_train_file, out_dev_file):
    data_file = os.path.join('dataset', args.dataset, args.sql_expr, args.expr, 'fusion_retrieved.jsonl')
    train_data = []
    dev_data = []
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            qid = item['id']
            mode, _ = qid.split('_')
            if mode == 'train':
                train_data.append(item)
            elif mode == 'dev':
                dev_data.append(item)
            else:
                raise ValueError('qid (%s) unexpected' % qid)
  
    table_dict = read_tables(args) 
    updated_train_data = process_train(train_data, args.top_n_train, table_dict, args.strategy, args.min_tables) 
    write_data(updated_train_data, out_train_file)

    updated_dev_data = process_dev(dev_data, args.top_n_dev, table_dict, args.strategy, args.min_tables)
    write_data(updated_dev_data, out_dev_file)

def read_tables(args):
    table_file = '/home/cc/data/%s/tables/tables.jsonl' % args.dataset
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

