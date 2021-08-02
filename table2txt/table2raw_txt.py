import json
import os
from tqdm import tqdm
import numpy as np
import random
from multiprocessing import Pool as ProcessPool

def read_table_file(table_lst, data_file):
    with open(data_file) as f:
        for line in tqdm(f):
            table = json.loads(line)
            table_lst.append(table)
    return table_lst

def read_tables():
    table_lst = []
    read_table_file(table_lst, '/home/cc/data/nq/tables/tables.jsonl')
    return table_lst

def process_table(table):
    columns = table['columns']
    col_name_lst = []
    for col_info in columns:
        col_name = col_info['text']
        col_name_lst.append(col_name)
    column_text = ' '.join(col_name_lst)
    row_data = table['rows']
    row_text_lst = []
    for row_item in row_data:
        cell_lst = row_item['cells']
        cell_text_lst = []
        for col_idx, cell in enumerate(cell_lst):
            col_name = col_name_lst[col_idx] 
            cell_text = cell['text']
            cell_text_lst.append(cell_text)
        row_text_lst.extend(cell_text_lst)     
    all_row_text = ' '.join(row_text_lst)
    
    table_text = column_text + ' ' + all_row_text 
    table_id = table['tableId']

    return (table_id, table_text)

def write_table_info(table_info, f_o_tables):
    table_id = table_info[0]
    table_text = table_info[1]
    out_data = {
        'id':table_id,
        'passage':table_text,
        'p_id':0,
        'tag':table_id
    }
    f_o_tables.write(json.dumps(out_data) + '\n')
  
def main():
    out_table_file = os.path.join('output', 'table_raw_text.json')
    f_o_tables = open(out_table_file, 'w')

    table_lst = read_tables()
    N = len(table_lst)
    work_pool = ProcessPool()
  
    for table_info in tqdm(work_pool.imap_unordered(process_table, table_lst), total=N):
        write_table_info(table_info, f_o_tables)
   
    ''' 
    for table in table_lst:
        table_info = process_table(table)
        write_table_info(table_info, f_o_tables)
    '''

    f_o_tables.close()
     
if __name__ == '__main__':
    main()


