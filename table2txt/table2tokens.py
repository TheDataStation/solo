import json
import os
from tqdm import tqdm
import numpy as np
import random
from multiprocessing import Pool as ProcessPool
from table2txt.table2graph import read_tables

MAX_NUM_TOKENS = 100

class CellBuffer:
    def __init__(self, max_buffer_size):
        self.max_buffer_size = max_buffer_size
        self.buffer_size = 0
        self.text_buffer = []

        self.custom_attr_text = '_custom_attr_text'
        self.custom_attr_size = '_custom_attr_size'
    
    def can_add(col_info, cell_info):
        if self.buffer_size == 0:
            return True
        text = col_info['text'] + ' ' + cell_info['text'].strip() + ' , '
        cell_info[self.custom_attr_text] = text
        tokens = text.split()
        token_size = len(tokens)
        cell_info[self.custom_attr_size] = token_size
        if token_size + self.buffer_size > self.max_buffer_size:
            return False
        return True

    def add(cell_info):
        text = cell_info[self.custom_attr_text]
        token_size = cell_info[self.custom_attr_size]
        self.text_buffer.append(text)
        self.buffer_size += token_size
    
    def pop()
        assert(len(self.text_buffer) > 0)
        text = ' '.join(self.text_buffer)
        self.buffer_size = 0
        self.text_buffer = []
        return text

class TextStrategy:
    def __init__(self):
        return
    def get_text(self, row_data):
        return None

class SlidingStrategy(TextStrategy):
    def get_text(self, col_info_lst, row_item):
        



def process_table(table):
    table_title = table['documentTitle'].strip()

    columns = table['columns']
    col_info_lst = []
    for col_data in columns:
        col_name = col_info['text'].strip()
        col_name_size = col_name.split()
        col_info = {
            'text':col_name,
            'size':col_name_size
        }
        col_info_lst.append(col_info)

    row_data = table['rows']
    row_text_lst = []
    for row_item in row_data:
        cell_lst = row_item['cells']
        
        buffer_text_lst = []
        buffer_size = 0
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
    
    if not args.debug:  
        work_pool = ProcessPool()
        for table_info in tqdm(work_pool.imap_unordered(process_table, table_lst), total=N):
            write_table_info(table_info, f_o_tables)
    else: 
        for table in tqdm(table_lst):
            table_info = process_table(table)
            write_table_info(table_info, f_o_tables)

    f_o_tables.close()
     
if __name__ == '__main__':
    main()


