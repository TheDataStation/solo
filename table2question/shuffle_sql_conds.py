import json
import random
import os
import argparse
from tqdm import tqdm
from table2question.wikisql_preprocess import get_sql_text
import copy
import numpy as np 
from itertools import permutations

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--sql_expr', type=str, required=True)
    args = parser.parse_args()
    return args

def read_tables(dataset):
    table_dict = {}
    table_file = '/home/cc/data/%s/tables/tables.jsonl' % dataset
    with open(table_file) as f:
        for line in tqdm(f):
            table = json.loads(line)
            table_id = table['tableId']
            table_dict[table_id] = table
    return table_dict

def main():
    args = get_args()
    out_dir = os.path.join('dataset', args.dataset, args.sql_expr, 'shuffled_conds')   
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    file_name = 'meta_shuffled.txt'
    out_meta_file = os.path.join(out_dir, file_name)
    if os.path.exists(out_meta_file):
        print('out file (%s) already exists.' % out_meta_file)
        return
    out_src_file = os.path.join(out_dir, 'test_unseen.source')
    out_tgt_file = os.path.join(out_dir, 'test_unseen.target')
    if os.path.exists(out_src_file):
        print('out source file (%s) already exists.' % out_src_file)
        return
    if os.path.exists(out_tgt_file):
        print('out target file (%s) already exists.' % out_tgt_file)
        return
    f_o_meta = open(out_meta_file, 'w')
    f_o_src = open(out_src_file, 'w')
    f_o_tgt = open(out_tgt_file, 'w')
    table_dict = read_tables(args.dataset)    
    input_meta_file = os.path.join('dataset', args.dataset, args.sql_expr, 'meta.txt')
    with open(input_meta_file) as f:
        for src_idx, line in tqdm(enumerate(f)):
            meta_item = json.loads(line)
            table_id = meta_item['table_id']
            query_table = get_query_table(table_dict[table_id])
            shuffled_items = shuffle_conds(meta_item, query_table)
            for query in shuffled_items:
                query['src_line_no'] = src_idx
                f_o_src.write(query['sql_text'] + '\n')
                f_o_tgt.write('a\n')
                f_o_meta.write(json.dumps(query) + '\n')
    f_o_src.close()
    f_o_tgt.close()
    f_o_meta.close()
             
def get_query_table(table):
    columns = table['columns']
    col_name_lst = [a['text'] for a in columns]
    query_table = {
        'id':table['tableId'],
        'header':col_name_lst
    }
    return query_table

def shuffle_conds(meta_item, query_table):
    shuffled_items = []
    cond_lst = meta_item['sql']['conds']
    cond_comb_itr = permutations(cond_lst)
    for cond_comb in cond_comb_itr:
        for title_col_name in ['about', 'given']:
            new_meta_item = copy.deepcopy(meta_item)
            new_meta_item['sql']['conds'] = cond_comb
            sql_info = new_meta_item['sql']
            sql_text = get_sql_text(query_table, sql_info, default_name=title_col_name)
            new_meta_item['sql_text'] = sql_text
            shuffled_items.append(new_meta_item)
    return shuffled_items   
     
if __name__ == '__main__':
    main()

