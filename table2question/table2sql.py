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
import time

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
    MAX_COL_NAME_SIZE = 20
    if col_name == '':
        return False
    col_token_size = get_text_size(col_name)
    if col_token_size > MAX_COL_NAME_SIZE:
        return False
    return True

def get_outlier(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower = 1 # Q1 - 1.5 * IQR
    if lower <= 0:
        lower = 1
    upper = Q3 + 1.5 * IQR
    return (lower, upper)

def stat_tables(table_lst):
    col_name_size_lst = []
    cell_size_lst = []
    for table in tqdm(table_lst):
        columns = table['columns']
        table_col_size_lst = [get_text_size(a['text']) for a in columns]
        col_name_size_lst.extend(table_col_size_lst)
        row_info_lst = table['rows']
        for row_info in row_info_lst:
            cell_lst = row_info['cells']
            row_cell_size_lst = [get_text_size(a['text']) for a in cell_lst]
            cell_size_lst.extend(row_cell_size_lst)
    
    col_name_outlier = get_outlier(col_name_size_lst)
    cell_outlier = get_outlier(cell_size_lst)
    table_stat_info = {
        'col_name_outlier':col_name_outlier,
        'cell_outlier':cell_outlier
    }
    return table_stat_info

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

def get_good_cols(table):
    col_ent_data = get_col_entities(table)
    good_cols = []
    for col, col_info in enumerate(col_ent_data):
        if is_good_col_name(col_info['col_name']):
            good_cols.append(col)
    return (col_ent_data, good_cols) 

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

def generate_queries(sql_dir, mode, table_lst, num_queries, stat_info, sql_dict):
    query_lst = []
    max_try_count = int(1E9)
    if num_queries > max_try_count:
        raise ValueError('The %s query size %d must be less than %d ' % (mode, num_queries, max_try_count))
    try_count = 0
    task_desc = '%s sqls' % mode
    pbar = tqdm(desc=task_desc, total=num_queries)
    progress = 0
    while (len(query_lst) < num_queries) and (try_count < max_try_count):
        try_count += 1
        table = random.sample(table_lst, 1)[0]
        col_ent_data, good_cols = get_good_cols(table)
        if len(good_cols) == 0:
            continue 
        infer_column_type(col_ent_data)
        query = sample_query(table, col_ent_data, good_cols, stat_info)
        if query is not None: 
            query_table = get_query_table(table['tableId'], col_ent_data)
            sql_info = query['sql']
            sql_text = get_sql_text(query_table, sql_info)
            sql_text_key = sql_text.lower()
            if sql_text_key not in sql_dict:
                sql_dict[sql_text_key] = 1 
                query['sql_text'] = sql_text
                query_lst.append(query)
                
                progress_step = (len(query_lst) - progress)
                if progress_step > 0: 
                    pbar.update(progress_step)
                    progress = len(query_lst)
   
    f_o_src, f_o_tar, f_o_meta = create_sql_file(sql_dir)
    write_query(mode, query_lst, f_o_src, f_o_tar, f_o_meta)

def sample_query(table, col_ent_data, col_lst, stat_info):
    table_id = table['tableId']
    row_data = table['rows']
    sel_col = random.sample(col_lst, 1)[0]
    sel_col_type = col_ent_data[sel_col]['type_infered'] 
    if sel_col_type == 'float':
        agg_op = random.sample(SqlQuery.agg_ops[1:], 1)[0] 
    else:
        agg_op = ''
    agg_op_idx = SqlQuery.agg_ops.index(agg_op)
    
    all_cond_cols = [a for a in col_lst if a != sel_col]
    cond_col_num_lst = [0, 1, 2, 3] # the sql cond will also include the title as ('about', =, Title)
    cond_op_idx_lst = [a for a in range(len(SqlQuery.cond_ops)-1)] # ignore the last one 'op'
        
    sql_cond_lst = []
    title = table['documentTitle'].strip()
    if title != '':
        sql_cond = [None, 0, title]
        sql_cond_lst.append(sql_cond)
    
    cond_col_num = random.sample(cond_col_num_lst, 1)[0]
    cond_col_lst = []
    row = None
    if cond_col_num > 0:
        num_sample_cond_col = min(len(all_cond_cols), cond_col_num)
        cond_col_lst = random.sample(all_cond_cols, num_sample_cond_col)
        row_spaces = get_sample_row_space(row_data, col_ent_data, cond_col_lst)
        if len(row_spaces) > 0:
            row = random.sample(row_spaces, 1)[0]
            for cond_col in cond_col_lst:
                sql_cond = get_sql_cond(row, col_ent_data, cond_col, cond_op_idx_lst, stat_info)
                if sql_cond is not None:
                    sql_cond_lst.append(sql_cond)
   
    if len(sql_cond_lst) == 0:
        return None
         
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
    return query_info

def get_sql_cond(row, col_ent_data, cond_col, cond_op_idx_lst, stat_info):
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
    outlier_upper = stat_info['cell_outlier'][1]
    if cond_value_size > outlier_upper:
        return None
         
    sql_cond = [int(cond_col), int(cond_op_idx), cond_value]
    return sql_cond 

def get_sample_row_space(row_data, col_ent_data, col_lst):
    row_spaces = []
    for row in range(len(row_data)):
        if not is_row_data_missing(row, col_ent_data, col_lst):
            row_spaces.append(row)
    return row_spaces 

def is_row_data_missing(row, col_ent_data, col_lst):
    for col in col_lst:
        if col_ent_data[col]['entities'][row] == '':
            return True
    return False

def get_train_dev_tables(args):
    input_table_file = os.path.join(args.work_dir, 'data', args.dataset, 'tables', args.table_file)
    table_lst = read_tables(input_table_file, None)
    num_tables = len(table_lst)
    num_dev = int(num_tables * args.dev_table_pct)
    dev_tables = random.sample(table_lst, num_dev)
    dev_table_id_set = set([a['tableId'] for a in dev_tables])
    train_tables = [a for a in table_lst if a['tableId'] not in dev_table_id_set] 
    return (table_lst, train_tables, dev_tables)

def write_table_split(table_lst, out_dir, file_name):
    out_file = os.path.join(out_dir, file_name)
    with open(out_file, 'w') as f_o:
        for table in table_lst:
            table_id = table['tableId']
            item = {'table_id':table_id}
            f_o.write(json.dumps(item) + '\n')

def write_stat_info(stat_info, out_dir, file_name):
    out_file = os.path.join(out_dir, file_name)
    with open(out_file, 'w') as f_o:
        f_o.write(json.dumps(stat_info))

def create_sql_file(sql_dir):
    if not os.path.isdir(sql_dir):
        os.mkdir(sql_dir)
    out_file_src = os.path.join(sql_dir, 'test_unseen.source')
    out_file_tar = os.path.join(sql_dir, 'test_unseen.target')
    f_o_src = open(out_file_src, 'w')
    f_o_tar = open(out_file_tar, 'w')
    out_meta_file = os.path.join(sql_dir, 'meta.txt')
    f_o_meta = open(out_meta_file, 'w')
    return (f_o_src, f_o_tar, f_o_meta)

def init_data(args):
    table2question_dir = os.path.join(args.work_dir, 'open_table_discovery/table2question')
    dataset_dir = os.path.join(table2question_dir, 'dataset', args.dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    out_dir = os.path.join(dataset_dir, args.experiment)
    if os.path.isdir(out_dir):
        err_msg = ('[%s] already exists, please use a different value for [--out_dir].\n'
              % (out_dir))
        
        msg_info = {
            'state':False,
            'msg':err_msg
        }
        return msg_info
    os.makedirs(out_dir)
    
    all_tables, train_tables, dev_tables = get_train_dev_tables(args)
    write_table_split(train_tables, out_dir, 'train_tables.jsonl')
    write_table_split(dev_tables, out_dir, 'dev_tables.jsonl')

    init_worker()
    stat_info = stat_tables(all_tables)
    write_stat_info(stat_info, out_dir, 'stat_info.json') 
    
    dev_sql_dir = os.path.join(out_dir, 'dev')
    sql_dict = {}
    generate_queries(dev_sql_dir, 'dev', dev_tables, args.num_dev_queries, stat_info, sql_dict)
    msg_info = {
        'state':True,
        'sql_data_dir':out_dir,
        'sql_dict':sql_dict,
        'train_tables':train_tables,
        'stat_info':stat_info
    }
    return msg_info

def write_query(mode, query_lst, f_o_src, f_o_tar, f_o_meta):
    for idx, query in tqdm(enumerate(query_lst), total=len(query_lst)):
        qid = '%s_%d' % (mode, idx)
        query['qid'] = qid
        query['mode'] = mode
        f_o_src.write(query['sql_text'] + '\n')
        f_o_tar.write('a\n')
        f_o_meta.write(json.dumps(query) + '\n')


