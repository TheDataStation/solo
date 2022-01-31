import json
import os
from tqdm import tqdm
import numpy as np
import random
from multiprocessing import Pool as ProcessPool
from table2txt.table2graph import read_tables
import argparse

MAX_NUM_TOKENS = 100

class CellBuffer:
    def __init__(self, max_buffer_size=MAX_NUM_TOKENS, stride=1):
        self.max_buffer_size = max_buffer_size
        self.buffer_size = 0
        self.text_buffer = []

        self.custom_attr_col_idx = '_user_col_idx'
        self.custom_attr_text = '_user_text'
        self.custom_attr_size = '_user_size'
        

    def can_add(self, col_idx, col_info, cell_info):
        if col_info['text'] != '':
            text = col_info['text'] + ' ( ' + cell_info['text'].strip() + ' ) . '
        else:
            text = cell_info['text'].strip() + ' . '

        cell_info[self.custom_attr_col_idx] = col_idx
        cell_info[self.custom_attr_text] = text
        tokens = text.split()
        token_size = len(tokens)
        cell_info[self.custom_attr_size] = token_size
        
        if self.buffer_size == 0:
            return True
        if token_size + self.buffer_size > self.max_buffer_size:
            return False
        return True

    def add(self, cell_info):
        token_size = cell_info[self.custom_attr_size]
        self.buffer_size += token_size
        self.text_buffer.append(cell_info)
    
    def can_pop(self):
        return len(self.text_buffer) > 0
     
    def pop(self, title):
        assert(len(self.text_buffer) > 0)
        out_text = title + '   .   ' + ' '.join([a[self.custom_attr_text] for a in self.text_buffer])
        out_meta = {'cols':[a[self.custom_attr_col_idx] for a in self.text_buffer]}
        out_data = {
            'text':out_text,
            'meta':out_meta
        }
        
        self.text_buffer = self.text_buffer[-1:]
        self.buffer_size = sum([a[self.custom_attr_size] for a in self.text_buffer]) 
        if self.buffer_size > self.max_buffer_size:
            self.text_buffer = []
            self.buffer_size = 0
         
        return out_data

    def reset(self):
        self.buffer_size = 0
        self.stride_buffer = []
        self.text_buffer = []

class TextStrategy:
    def __init__(self):
        self.cell_buffer = CellBuffer()
        return
    
    def reset(self):
        self.cell_buffer.reset()

    def get_text(self, row_data):
        return None

class SlidingStrategy(TextStrategy):
    def __init__(self):
         super(SlidingStrategy, self).__init__()
    
    def get_text(self, title, col_info_lst, row_item):
        cell_lst = row_item['cells']
        text_info_lst = []
        for col_idx, col_info in enumerate(col_info_lst):
            cell_info = cell_lst[col_idx]
            if col_info['text'] == '' and cell_info['text'].strip() == '':
                continue
            
            if self.cell_buffer.can_add(col_idx, col_info, cell_info):
                self.cell_buffer.add(cell_info)
            else:
                text_info = self.cell_buffer.pop(title)
                text_info_lst.append(text_info)
                self.cell_buffer.add(cell_info)
        
        if self.cell_buffer.can_pop():
            text_info = self.cell_buffer.pop(title)
            text_info_lst.append(text_info)
    
        return text_info_lst

def get_strategy(strategy_name):
    if strategy_name == 'slide':
        return SlidingStrategy()
    
    raise ValueError('strategy [%s] not supported')    

def init_worker(strategy_name):
    global g_strategy
    g_strategy = get_strategy(strategy_name) 

def get_col_data(table):
    columns = table['columns']
    col_info_lst = []
    for col_data in columns:
        col_name = col_data['text'].strip()
        col_name_size = col_name.split()
        col_info = {
            'text':col_name,
            'size':col_name_size
        }
        col_info_lst.append(col_info)
    return col_info_lst

def process_table(table):
    table_title = table['documentTitle'].strip()
    col_info_lst = get_col_data(table)
    row_data = table['rows']
    table_text_lst = []
    for row_idx, row_item in enumerate(row_data):
        g_strategy.reset()
        out_text_data = g_strategy.get_text(table_title, col_info_lst, row_item)    
        for text_info in out_text_data:
            text_info['meta']['row'] = row_idx
        
        table_text_lst.extend(out_text_data) 
    return table_text_lst

def write_table_info(p_id, table_id, text_info, f_o_tables):
    text_info['meta']['table_id'] = table_id
    out_data = {
        'p_id':p_id,
        'passage':text_info['text'],
        'tag':text_info['meta']
    }
    f_o_tables.write(json.dumps(out_data) + '\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--table_file', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--parallel', type=int, default=0)
    args = parser.parse_args()
    return args
  
def main():
    args = get_args()
    exprt_dir = os.path.join('dataset', args.dataset, args.experiment)
    if os.path.exists(exprt_dir):
        print('[%s] already exists' % exprt_dir)
        return
    os.makedirs(exprt_dir)
    out_table_file = os.path.join(exprt_dir, 'token_text.jsonl')
    f_o_tables = open(out_table_file, 'w')
    input_table_file = os.path.join('/home/cc/data', args.dataset, 'tables', args.table_file)
    table_lst = read_tables(input_table_file)
    N = len(table_lst)
    
    g_p_id = 0
    init_worker(args.strategy) 
    for table in tqdm(table_lst):
        table_id = table['tableId']
        table_text_lst = process_table(table)
        for text_info in table_text_lst:
            write_table_info(g_p_id, table_id, text_info, f_o_tables)
            g_p_id += 1

    f_o_tables.close()
     
if __name__ == '__main__':
    main()


