import json
import os
from tqdm import tqdm
import numpy as np
import random
from multiprocessing import Pool as ProcessPool
from table2graph import read_tables

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--table_file', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--debug', type=int, default=0)
    args = parser.parse_args()
    return args
  
def main():
    args = get_args()
    out_table_file = os.path.join('dataset', args.dataset, args.experiment, 'token_text.jsonl')
    f_o_tables = open(out_table_file, 'w')

    table_lst = read_tables()
    N = len(table_lst)
    work_pool = ProcessPool()
  
    for table_info in tqdm(work_pool.imap_unordered(process_table, table_lst), total=N):
        write_table_info(table_info, f_o_tables)
    
    if not args.debug:  
    ''' 
    for table in table_lst:
        table_info = process_table(table)
        write_table_info(table_info, f_o_tables)
    '''

    f_o_tables.close()
     
if __name__ == '__main__':
    main()


