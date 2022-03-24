import json
import os
from tqdm import tqdm
import argparse
import re
from webnlg.data.template_data import TemplateTag

def get_text_meta(text_dir, text_part_name):
    index = text_part_name.index('_part_')
    part_name = text_part_name[index:]
    meta_file = os.path.join(text_dir, 'graph_parts', 'template_graph.jsonl' + part_name)
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
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            tag = item['tag']
            key = get_key(tag['table_id'], tag['row'], tag['sub_col'], tag['obj_col']) 
            graph_dict[key] = item
    return graph_dict 

def remove_tags(text):
    out_text = text
    tag_lst = [TemplateTag.Subject_Start, TemplateTag.Subject_End, 
               TemplateTag.Object_Start, TemplateTag.Object_End]
    for tag in tag_lst:
        out_text = out_text.replace(tag, '')
    return out_text

def get_meta_range_text(meta_info, start_tag, end_tag):
    graph = meta_info['graph']
    pos_1 = graph.find(start_tag)
    pos_2 = graph.find(end_tag)
    updated_pos_2 = pos_2 + len(TemplateTag.Subject_End)
    return graph[pos_1:updated_pos_2]

def get_range(meta_info, text, start_tag, end_tag):
    pos_1 = text.find(start_tag) 
    pos_2 = text.rfind(end_tag) # find the right mode
    if pos_1 == -1 and pos_2 == -1:
        return None
    if pos_1 >= pos_2:
        return None 
    
    if pos_1 == -1:
        #import pdb; pdb.set_trace()
        graph_text = get_meta_range_text(meta_info, start_tag, end_tag)
        part_text = graph_text.replace(start_tag, '').strip()
        pos_1 = text.find(part_text)
    
    if pos_2 > -1:
        pos_2 += len(TemplateTag.Subject_End) - 1
    else:
        graph_text = get_meta_range_text(meta_info, start_tag, end_tag)
        part_text = graph_text.replace(end_tag, '').strip()
        pos_2_offset = text.find(part_text)
        pos_2 = sub_pos_2_offset + len(part_text) - 1

    return (pos_1, pos_2)

def replace_sub_obj(meta_info, table_data, text, sub, obj):
    sub_range = get_range(meta_info, text, TemplateTag.Subject_Start, TemplateTag.Subject_End)
    if sub_range is None:
        return  ''
    obj_range = get_range(meta_info, text, TemplateTag.Object_Start, TemplateTag.Object_End)
    if obj_range is None:
        return ''
    sub_pos_1, sub_pos_2 = sub_range
    obj_pos_1, obj_pos_2 = obj_range
    out_text = ''
    if (sub_pos_1 < sub_pos_2) and (obj_pos_1 < obj_pos_2):
        if sub_pos_2 < obj_pos_1:
            out_text = text[:sub_pos_1] + sub + text[(sub_pos_2+1):obj_pos_1] + obj + text[(obj_pos_2+1):]
        elif (obj_pos_2 < sub_pos_1):
            out_text =  text[:obj_pos_1] + obj + text[(obj_pos_2+1):sub_pos_1] + sub + text[(sub_pos_2+1):]
    
    return out_text

def merge_text(meta_info, text, table_dict, graph_dict, f_o):
    table_id = meta_info['table_id']
    template_row = meta_info['row']
    sub_col = meta_info['sub_col']
    obj_col = meta_info['obj_col']
    table_data = table_dict[table_id]
    title = table_data['documentTitle'].strip()
    row_data = table_data['rows']
    for row, row_item in enumerate(row_data):
        if template_row == row:
            out_text = remove_tags(text)
        else:
            if sub_col is None:
                sub = title
            else:
                sub = row_item['cells'][sub_col]['text'].strip()
            obj = row_item['cells'][obj_col]['text'].strip() 
            out_text = replace_sub_obj(meta_info, table_data, text, sub, obj) 
        
        key = get_key(table_id, row, sub_col, obj_col)
        graph_item = graph_dict[key]
        graph_item['generated_text'] = out_text
        merge_text = graph_item['passage'].rstrip()
        if out_text != '':
            if merge_text.endswith('.'):
                merge_text += ' ' + out_text
            else:
                merge_text += '  .  ' + out_text
        graph_item['passage'] = merge_text

        f_o.write(json.dumps(graph_item) + '\n')

def main():
    args = get_args()
    text_expr_dir = os.path.join('dataset', args.dataset, args.text_expr) 
    text_part_dir = os.path.join(text_expr_dir, 'text_parts')
    text_part_lst = sorted(os.listdir(text_part_dir))
    if len(text_part_lst) == 0:
        print('No generated text parts in directory [%s]' % text_part_dir)
        return
    out_passage_file = os.path.join(text_expr_dir, 'merged_passages.jsonl') 
    if os.path.exists(out_passage_file):
        print('merged file (%s) already exists' % out_passage_file)
        return
    f_o = open(out_passage_file, 'w')
    table_dict = read_tables(args)
    graph_dict = read_rel_graph(args) 
    for text_part_name in tqdm(text_part_lst):
        text_meta_data = get_text_meta(text_expr_dir, text_part_name)
        text_part_file = os.path.join(text_part_dir, text_part_name)
        with open(text_part_file) as f:
            for idx, text in tqdm(enumerate(f)):
                template_graph_text = text.rstrip()
                meta_info = text_meta_data[idx] 
                merge_text(meta_info, text, table_dict, graph_dict, f_o)
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
