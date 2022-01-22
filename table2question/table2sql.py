import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import random
from multiprocessing import Pool as ProcessPool
from table2txt.graph_strategy.template_graph import TemplateGraph

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

def init_worker(strategy_name):
    global graph_strategy
    graph_strategy = TemplateGraph()

def infer_table_keys(table):
    col_ent_data = graph_strategy.get_col_entities(table)
    ent_num_lst = []
    for col_info in col_ent_data:
        entities = col_info['entities']
        ent_text_lst = [a['text'].lower() for a in entities]
        ent_set = set(ent_text_lst)  
        ent_num = len(ent_set)
        ent_num_lst.append(ent_num)
    
    sorted_idxes = np.argsort(ent_num_lst)
    key_idx = sorted_idxes[-1]
    max_ent_num = ent_num_lst[key_idx]
    key_idx_lst = [key_idx]
    
    N = len(col_ent_data)
    idx = N - 2
    while idx >= 0:
        col_idx = sorted_idxes[idx]
        ent_num = ent_num_lst[col_idx]
        if ent_num != max_ent_num:
            break
        key_idx_lst.append(col_idx)
        idx -= 1
   
    key_idx_lst = key_idx_lst[:2]
    key_idx_set = set(key_idx_lst)
    non_key_idx_lst = [a for a in range(N) if a not in key_idx_set]
   
    return (col_ent_data, key_idx_lst, non_key_idx_lst)
     
def is_float(text):
    if text == '':
        return False
    dot_lst = []
    digit_lst = []
    for a in text:
        if a == '.':
            dot_lst.append(a)
        elif a.isdigit():
            digit_lst.append(a)
    
    if (len(dot_lst) <= 1) and (len(dot_lst) + len(digit_lst) == len(text)):
        return True
    else:
        return False

def infer_column_type(col_ent_data)
    for col_info in col_ent_data:
        entities = col_info['entities']  
        ent_text_lst = [a['text'] for a in entities] 
        type_lst = [is_float(a) for a in ent_text_lst] 
        col_type = 'float' if all(type_lst) else 'text'
        col_info['type_infered'] = col_type 

def process_table(table):
    col_ent_data, key_idx_lst, non_key_idx_lst = infer_table_keys(table) 
    infer_column_type(col_ent_data)
    sql_lst = sample_sqls(table, col_ent_data, key_idx_lst, non_key_idx_lst)
    return sql_lst 

def sample_sqls(table, col_ent_data, key_idx_lst, non_key_idx_lst):
    max_samles = 6
    num_samples = 0
    
    while (num_samples < max_samles):
        key_col = (random.sample(key_idx_lst, 1))[0]
        non_key_num_lst = [1, 2]
        non_key_num = (random.sample(non_key_num_lst, 1))[0]
        non_key_cols = random.sample(non_key_idx_lst, non_key_num) 
        
        table_id = table['tableId']
        
           


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
    f_o_meta = open(out_row_table_file, 'w')

    table_file_name = args.table_file
    input_tables = os.path.join('/home/cc/data', args.dataset, 'tables', table_file_name)
    table_lst = read_tables(input_tables, None)

    DEBUG = False
    if not DEBUG:
        work_pool = ProcessPool(initializer=init_worker, initargs=(args.strategy,))
        for graph_lst in tqdm(work_pool.imap_unordered(process_table, table_lst), total=len(table_lst)):
            write_graphs(graph_lst, f_o_src, f_o_tar, f_o_meta) 
             
    else:
        init_worker(args.strategy)
        for table in tqdm(table_lst):
            graph_lst = process_table(table)
            write_graphs(graph_lst, f_o_src, f_o_tar, f_o_meta) 
            
    f_o_src.close()
    f_o_tar.close()
    f_o_meta.close()  

def write_graphs(graph_lst, f_o_src, f_o_tar, f_o_meta):
    for graph_info in graph_lst:
        f_o_src.write(graph_info['graph'] + '\n')
        f_o_tar.write('a\n')
        meta_info = {
            'table_id': graph_info['table_id'],
            'row': graph_info['row'],
            'sub_col':graph_info['sub_col'],
            'obj_col':graph_info['obj_col']
        }
        f_o_meta.write(json.dumps(meta_info) + '\n')


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


