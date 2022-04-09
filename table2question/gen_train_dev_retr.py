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
    f_o_train = open(out_train_file, 'w')
    f_o_dev = open(out_dev_file, 'w')
    data_file = os.path.join('dataset', args.dataset, args.sql_expr, args.expr, 'fusion_retrieved.jsonl')
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            qid = item['id']
            mode, _ = qid.split('_')
            if mode == 'train':
                f_o_train.write(line)
            elif mode == 'dev':
                f_o_dev.write(line)
            else:
                raise ValueError('qid (%s) unexpected' % qid)
    f_o_train.close()
    f_o_dev.close()

if __name__ == '__main__':
    main()

