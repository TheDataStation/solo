import json
import os
from tqdm import tqdm
import argparse
import re

def get_text_meta(text_dir, text_part_name):
    index = text_part_name.index('_part_')
    part_name = text_part_name[index:]
    meta_file = os.path.join(text_dir, 'template_graph.jsonl' + part_name)
    meta_info_lst = []
    with open(meta_file) as f:
        for line in f:
            item = json.loads(line)
            meta_info_lst.append(item)
    return meta_info_lst

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

def read_tables(args):
    table_data_file = os.path.join('/home/cc/data', args.dataset, 'tables', 'tables.jsonl') 
    table_lst = []
    read_table_file(table_lst, table_data_file, None)
    table_dict = {}
    for table in table_lst:
        table_id = table['tableId']
        table_dict[table_id] = table
    return table_dict

def get_key(table_id, row, sub_col, obj_col):
    return f'{table_id}_{row}_{sub_col}_{obj_col}'

def read_rel_graph(args):
    graph_dict = {}
    data_dir = os.path.join('dataset', args.dataset, args.graph_expr)
    data_file = os.path.join(data_dir, 'passages.jsonl')
    with open(data_file) as :
        for line in tqdm(f):
            item = json.loads(line)
            tag = item['tag']
            key = get_key(tag['table'], tag['row'], tag['sub_col'], tag['obj_col']) 
            graph_dict[key] = item
    return graph_dict 

def merge_text(meta_info, text, table_dict, graph_dict):
    

def main():
    args = get_args()
    table_dict = read_tables(args)
    graph_dict = read_rel_graph(args) 
    import pdb; pdb.set_trace()
    text_expr_dir = os.path.join('dataset', args.dataset, args.text_expr) 
    text_part_dir = os.path.join(text_expr_dir, 'text_parts')
    text_part_lst = sorted(os.listdir(text_part_dir))
    if len(text_part_lst) == 0:
        print('No generated text parts in directory [%s]' % text_part_dir)
        return
    out_passage_file = os.path.join(text_expr_dir, 'merged_passages.json') 
    f_o = open(out_passage_file, 'w')
    table_expand_info = {}
    passage_id = 0
    for text_part_name in tqdm(text_part_lst):
        text_meta_data = get_text_meta(text_dir, text_part_name)
        text_part_file = os.path.join(text_part_dir, text_part_name)
        with open(text_part_file) as f:
            for idx, text in enumerate(f):
                template_graph_text = text.rstrip()
                meta_info = text_meta_data[idx] 
                merge_text(meta_info, text, table_dict, graph_dict)
    f_o.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--graph_expr', type=str)
    parser.add_argument('--text_expr', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
