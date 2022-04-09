import json
import random
import os
from tqdm import tqdm
import csv
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--expr', type=str)
    parser.add_argument('--sql_expr', type=str)
    parser.add_argument('--top_n_train', type=int, default=100)
    parser.add_argument('--top_n_dev', type=int, default=100)
    args = parser.parse_args()
    return args

def get_out_file(args, mode):
    out_dir = os.path.join('dataset', args.dataset, args.sql_expr, args.expr)
    out_file = os.path.join(out_dir, 'fusion_retrieved_%s.jsonl' % mode)
    return out_file 

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
   
    updated_train_data = process_train(train_data, args) 
    write_data(updated_train_data, out_train_file)
    updated_dev_data = process_dev(dev_data, args)
    write_data(updated_dev_data, out_dev_file)

def process_train(train_data, args):
    updated_train_data = []
    for item in tqdm(train_data):
        gold_table_lst = item['table_id_lst']
        ctxs = item['ctxs'][:args.top_n_train]
        labels = [int(a['tag']['table_id'] in gold_table_lst) for a in ctxs]
        if (max(labels) < 1) or (min(labels) > 0):
            continue
        item['ctxs'] = ctxs
        updated_train_data.append(item) 
    return updated_train_data

def process_dev(dev_data, args):
    updated_dev_data = []
    for item in tqdm(dev_data):
        ctxs = item['ctxs'][:args.top_n_dev]
        item['ctxs'] = ctxs
        updated_dev_data.append(item)
    return updated_dev_data 
    
def write_data(data, out_file):
    with open(out_file, 'w') as f:
        for item in tqdm(data):
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    main()

