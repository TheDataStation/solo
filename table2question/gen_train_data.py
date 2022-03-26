import json
import random
import os
from tqdm import tqdm
import csv
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--sql_expr', type=str)
    parser.add_argument('--expr', type=str)
    args = parser.parse_args()
    return args

def get_all_query_data(args):
    all_query_data = {}
    data_file = os.path.join('dataset', args.dataset, args.sql_expr, args.expr, 'fusion_retrieved.jsonl')
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['table_id_lst'][0]
            if table_id not in all_query_data:
                all_query_data[table_id] = []
            table_items = all_query_data[table_id]
            table_items.append(item)
    return all_query_data

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

    all_query_data = get_all_query_data(args)
    all_table_lst = list(all_query_data.keys())
    random.shuffle(all_table_lst)
    
    dev_table_lst = random.sample(all_table_lst, 1000)
    dev_table_set = set(dev_table_lst)
    all_table_set = set(all_table_lst)
    
    train_table_set = all_table_set - dev_table_set  
    
    output_data(out_train_file, train_table_set, all_query_data, 10000)
    output_data(out_dev_file, dev_table_set, all_query_data, 1000)

def output_data(out_file, table_set, all_query_data, num_examples):
    query_items = []
    for table_id in all_query_data:
        if table_id not in table_set:
            continue
        table_items = all_query_data[table_id]
        query_items.extend(table_items) 
    
    if num_examples is not None: 
        random.shuffle(query_items)
        sample_query_items = random.sample(query_items, num_examples)
    else:
        sample_query_items = query_items

    with open(out_file, 'w') as f_o:
        for item in sample_query_items:
            f_o.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    main()

