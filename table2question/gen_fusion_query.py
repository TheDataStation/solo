import json
import os
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--expr', type=str)
    args = parser.parse_args()
    return args

def read_meta(meta_file):
    meta_data = []
    with open(meta_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            meta_data.append(item)
    return meta_data

def read_questions(q_file):
    questions = []
    with open(q_file) as f:
        for line in tqdm(f):
            questions.append(line.strip())
    return questions

def read_tables(dataset):
    table_dict = {}
    table_title_dict = {} 
    table_file = '/home/cc/data/%s/tables/tables.jsonl' % dataset
    with open(table_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_dict[item['tableId']] = item
            title = item['documentTitle'].strip().lower()
            if title not in table_title_dict:
                table_title_dict[title] = []
            
            same_title_tables = table_title_dict[title]
            same_title_tables.append(item)

    return (table_dict, table_title_dict)

def get_table_col_names(table):
    columns = table['columns']
    col_names = [a['text'].strip().lower() for a in columns]
    return col_names

def get_gold_tables(meta_item, table_dict, table_title_dict):
    table_id = meta_item['table_id']
    meta_table = table_dict[table_id]
    row = meta_item['row']
    sql_info = meta_item['sql']
    sel_col = sql_info['sel']
    sel_col_name = meta_table['columns'][sel_col]['text'].strip().lower()
    conds = sql_info['conds']
    cond_cols = [a[0] for a in conds if a[0] is not None]
    meta_row_info = None
    if row is not None:
        meta_row_info = meta_table['rows'][row]

    title = meta_table['documentTitle'].strip().lower()
    same_title_tables = table_title_dict[title]
    
    gold_table_id_set = set([table_id])

    meta_table_col_names = get_table_col_names(meta_table)

    for other_table in same_title_tables:
        if other_table['tableId'] == table_id:
            continue
        
        other_col_names = get_table_col_names(other_table) 
        if sel_col_name not in other_col_names:
            continue
        
        if meta_row_info is None:
            gold_table_id_set.add(other_table['tableId'])
        else:
            if is_gold_table(other_table, other_col_names, meta_table, meta_row_info, cond_cols, meta_table_col_names):
                gold_table_id_set.add(other_table['tableId'])
    
    gold_table_id_lst = list(gold_table_id_set)
    return gold_table_id_lst          


def is_gold_table(other_table, other_col_names, meta_table, meta_row_info, cond_cols, meta_table_col_names):
    other_table_row_data = other_table['rows']
    for other_row_info in other_table_row_data:
        if same_row_data(other_table, other_col_names, other_row_info, 
                         meta_table, meta_row_info, cond_cols, meta_table_col_names):
            return True
    return False

def same_row_data(other_table, other_col_names, other_row_info, 
                  meta_table, meta_row_info, cond_cols, meta_table_col_names):
    
    for col in cond_cols:
        meta_text = meta_row_info['cells'][col]['text'].strip().lower()
        meta_col_name = meta_table_col_names[col]
        other_col_lst = [idx for idx, other_name in enumerate(other_col_names) if other_name == meta_col_name]
        matched = False
        for other_col in other_col_lst:
            other_text = other_row_info['cells'][other_col]['text'].strip().lower()
            if other_text == meta_text:
                matched = True
                break
        if not matched:
            return False

    return True

def main():
    args = get_args()
    data_dir = os.path.join('dataset', args.dataset, args.expr) 
    meta_file = os.path.join(data_dir, 'meta.txt')
    q_file = os.path.join(data_dir, 'questions.txt')
    out_query_file = os.path.join(data_dir, 'fusion_query.jsonl')
    if os.path.exists(out_query_file):
        print('[%s] already exists' % out_query_file)
        return 
     
    f_o_query = open(out_query_file, 'w') 
    meta_data = read_meta(meta_file)
    q_data = read_questions(q_file)
   
    table_dict, table_title_dict = read_tables(args.dataset) 
    for idx, question in tqdm(enumerate(q_data)):
        meta_item = meta_data[idx]
        qid = meta_item['qid']
        table_id_lst = get_gold_tables(meta_item, table_dict, table_title_dict)
        answers = ['N/A']       
        #passage_info = {
        #    'title':'',
        #    'text':'This is a example passage',
        #}
        out_item = {
            'id':qid,
            'question':question,
            'table_id_lst':table_id_lst,
            'answers':answers,
            'ctxs':[]
        }
        
        f_o_query.write(json.dumps(out_item) + '\n') 
    
    f_o_query.close()

if __name__ == '__main__':
    import time
    t1 = time.time()
    main()
    t2 = time.time()
    print('%d seconds' % (t2 - t1))


