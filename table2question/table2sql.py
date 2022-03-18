import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import transformers
import random
from multiprocessing import Pool as ProcessPool
from table2question.sql_data import SqlQuery 
from table2question.wikisql_preprocess import get_sql_text
import re

g_tokenizer = None

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

def init_worker():
    global g_tokenizer
    g_tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

def is_good_col_name(col_name):
    if col_name == '':
        return False
    col_tokens = col_name.split()
    sub_tokens = [a for a in col_tokens if len(a) < 2]
    if len(sub_tokens) > 0:
        return False
    return True

def get_key_cols(col_ent_data, ent_num_lst):
    sorted_col_idxes = np.argsort(ent_num_lst)
    N = len(col_ent_data)
    idx = N - 1
    key_ent_num = -1
    key_col_lst = []
    while idx >= 0:
        col = sorted_col_idxes[idx]
        if is_good_col_name(col_ent_data[col]['col_name']):
            if key_ent_num < 0:
                key_ent_num = ent_num_lst[col]
                key_col_lst.append(col)
            else:
                if key_ent_num == ent_num_lst[col]:
                    key_col_lst.append(col)

        idx -= 1
    
    if len(key_col_lst) > 0:
        key_col_lst.sort()
    return key_col_lst

def get_text_size(text):
    tokens = g_tokenizer.tokenize(text)
    return len(tokens)

def get_col_entities(table):
    col_entity_lst = []
    column_data = table['columns']
    row_data = table['rows']
    for col_idx, col_info in enumerate(column_data):
        col_name = col_info['text'].strip()
        ent_info_lst = []
        for row_idx, row_info in enumerate(row_data):
            ent_text = row_info['cells'][col_idx]['text'].strip()
            ent_size = get_text_size(ent_text)
            ent_info = {'text':ent_text, 'size':ent_size, 'row':row_idx}
            ent_info_lst.append(ent_info)

        col_entity_info = {
            'col_name':col_name,
            'entities':ent_info_lst
        }
        col_entity_lst.append(col_entity_info)
    return col_entity_lst

def infer_table_keys(table):
    col_ent_data = get_col_entities(table)
    ent_num_lst = []
    for col_info in col_ent_data:
        entities = col_info['entities']
        ent_text_lst = [a['text'].lower() for a in entities]
        ent_set = set(ent_text_lst)  
        ent_num = len(ent_set)
        ent_num_lst.append(ent_num)
   
    key_col_lst = get_key_cols(col_ent_data, ent_num_lst) 
    key_col_lst = key_col_lst[:2]
    key_col_set = set(key_col_lst)

    non_key_col_lst = []
    for col, col_info in enumerate(col_ent_data):
        if is_good_col_name(col_info['col_name']) and (col not in key_col_set):
            non_key_col_lst.append(col)
 
    return (col_ent_data, key_col_lst, non_key_col_lst)
     
def is_float(text):
    if text == '':
        return False
    if re.match(r'^-?\d+(?:\.\d+)$', text) is None:
        return False
    return True

def infer_column_type(col_ent_data):
    for col_info in col_ent_data:
        entities = col_info['entities']  
        ent_text_lst = [a['text'] for a in entities] 
        float_flag_lst = [is_float(a) for a in ent_text_lst] 
        col_type = 'float' if all(float_flag_lst) else 'text'
        col_info['type_infered'] = col_type 

def get_query_table(table_id, col_ent_data):
    col_name_lst = [a['col_name'] for a in col_ent_data]
    query_table = {
        'id':table_id,
        'header':col_name_lst
    }
    return query_table

def process_table(table):
    col_ent_data, key_col_lst, non_key_col_lst = infer_table_keys(table) 
    if len(key_col_lst + non_key_col_lst) == 0:
        return []

    infer_column_type(col_ent_data)
    query_lst = sample_queries(table, col_ent_data, key_col_lst, non_key_col_lst)
    query_table = get_query_table(table['tableId'], col_ent_data)

    for query in query_lst:
        sql_info = query['sql']
        sql_text = get_sql_text(query_table, sql_info)
        query['sql_text'] = sql_text

    return query_lst

def sample_queries(table, col_ent_data, key_col_lst, non_key_col_lst):
    sample_query_lst = []

    table_id = table['tableId']
    max_samles = 8
    num_samples = 0
    row_data = table['rows']
    col_lst = key_col_lst + non_key_col_lst
    float_col_lst = []
    text_col_lst = []
    sel_col_type_lst = []
    for col in col_lst:
        col_type = col_ent_data[col]['type_infered']
        sel_col_type_lst.append(col_type)
        if col_type == 'float':
            float_col_lst.append(col)
        else:
            text_col_lst.append(col)
    sel_col_type_lst = list(set(sel_col_type_lst))

    cond_num_lst = [0, 1, 2, 3] # the sql cond will also include the title
    
    cond_op_idx_lst = [a for a in range(len(SqlQuery.cond_ops)-1)]
    try_count = 0
    max_try_count = 100
    while (len(sample_query_lst) < max_samles) and (try_count < max_try_count):
        try_count += 1
        sel_col_type = random.sample(sel_col_type_lst, 1)[0] 
        if sel_col_type == 'float':
            sel_col = random.sample(float_col_lst, 1)[0]
        else:
            sel_col = random.sample(text_col_lst, 1)[0]
        
        if sel_col_type == 'float':
            agg_op = random.sample(SqlQuery.agg_ops[1:], 1)[0] 
        else:
            agg_op = ''
       
        agg_op_idx = SqlQuery.agg_ops.index(agg_op)
        
        if agg_op != '':
            all_cond_cols = [a for a in col_lst if a != sel_col]
        else:
            all_cond_cols = col_lst
       
        cond_num = random.sample(cond_num_lst, 1)[0]
        cond_col_lst = [None]
        if cond_num > 0:
            cond_num = min(len(all_cond_cols), cond_num)
            sample_cond_cols = random.sample(all_cond_cols, cond_num)
            cond_col_lst.extend(sample_cond_cols)
              
        query_col_lst = list(set([sel_col] + all_cond_cols))
        
        row_spaces = get_sample_row_space(row_data, col_ent_data, query_col_lst)
        if len(row_spaces) == 0:
            continue

        sql_cond_lst = []
        row = random.sample(row_spaces, 1)[0]
        for cond_col in cond_col_lst:
            if cond_col is None:
                sql_cond = [None, 0, table['documentTitle'].strip()]
                sql_cond_lst.append(sql_cond)
                continue

            cond_col_type = col_ent_data[cond_col]['type_infered']
            if cond_col_type == 'float':
                cond_op_idx = random.sample(cond_op_idx_lst, 1)[0]
            else:
                cond_op_idx = 0
            
            cond_value = col_ent_data[cond_col]['entities'][row]['text'] 
            cond_op = SqlQuery.cond_ops[cond_op_idx]
            if cond_op == '>':
                float_cond_value = float(cond_value)
                if float_cond_value >= 0:
                    float_cond_value = float_cond_value / 2
                else:
                    float_cond_value = float_cond_value * 2
                if float_cond_value == 0:
                    float_cond_value = -1
                cond_value = str(float_cond_value)
            elif cond_op == '<':
                float_cond_value = float(cond_value)
                if float_cond_value >= 0:
                    float_cond_value = float_cond_value * 2
                else:
                    float_cond_value = float_cond_value / 2
                if float_cond_value == 0:
                    float_cond_value = 1 
                cond_value = str(float_cond_value)
             
            cond_value_size = col_ent_data[cond_col]['entities'][row]['size']
            if cond_value_size > 30:
                continue
                 
            sql_cond = [int(cond_col), int(cond_op_idx), cond_value]
            sql_cond_lst.append(sql_cond)
        
        sql_info = {
            'conds':sql_cond_lst,
            'sel':int(sel_col),
            'agg':int(agg_op_idx),
        }

        query_info = {
            'question':'N/A',
            'sql':sql_info,
            'table_id':table_id,
            'row':row,
        }
        sample_query_lst.append(query_info)
    return sample_query_lst

def get_sample_row_space(row_data, col_ent_data, query_col_lst):
    row_spaces = []
    for row in range(len(row_data)):
        if not is_row_data_missing(row, col_ent_data, query_col_lst):
            row_spaces.append(row)
    return row_spaces 

def is_row_data_missing(row, col_ent_data, query_col_lst):
    for col in query_col_lst:
        if col_ent_data[col]['entities'][row] == '':
            return True
    return False
      

def main():
    args = get_args()
    table2question_dir = '/home/cc/code/open_table_discovery/table2question'
    dataset_dir = os.path.join(table2question_dir, 'dataset', args.dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    out_dir = os.path.join(dataset_dir, args.experiment)
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
    out_meta_file = os.path.join(out_dir, 'meta.txt')
    f_o_meta = open(out_meta_file, 'w')

    table_file_name = args.table_file
    input_tables = os.path.join('/home/cc/data', args.dataset, 'tables', table_file_name)
    table_lst = read_tables(input_tables, None)

    DEBUG = True
    all_query_lst = []
    if not DEBUG:
        work_pool = ProcessPool(initializer=init_worker)
        for query_lst in tqdm(work_pool.imap_unordered(process_table, table_lst), total=len(table_lst)):
            all_query_lst.extend(query_lst)
             
    else:
        init_worker()
        for table in tqdm(table_lst):
            query_lst = process_table(table)
            all_query_lst.extend(query_lst)
    
    sample_query_lst = get_table_queries(all_query_lst) 
    write_query(sample_query_lst, f_o_src, f_o_tar, f_o_meta)
     
    f_o_src.close()
    f_o_tar.close()
    f_o_meta.close()  

def get_table_queries(all_query_lst):
    table_query_dict = {}
    for query in all_query_lst:
        table_id = query['table_id']
        if table_id not in table_query_dict:
            table_query_dict[table_id] = []
        query_lst = table_query_dict[table_id]
        query_lst.append(query)
    table_id_lst = list(table_query_dict.keys())
    N = min(10000, len(table_id_lst))
    sample_table_id_lst = random.sample(table_id_lst, N)
    all_sample_query_lst = []
    for sample_table_id in sample_table_id_lst:
        query_lst = table_query_dict[sample_table_id]
        all_sample_query_lst.extend(query_lst)
    return all_sample_query_lst

def write_query(query_lst, f_o_src, f_o_tar, f_o_meta):
    for query in query_lst:
        f_o_src.write(query['sql_text'] + '\n')
        f_o_tar.write('a\n')
        meta_info = query
        f_o_meta.write(json.dumps(meta_info) + '\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--table_file', type=str)
    parser.add_argument('--experiment', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()


