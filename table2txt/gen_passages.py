import json
import os
from tqdm import tqdm
import argparse
import re
from table2txt.graph_strategy.strategy_constructor import get_strategy

def get_template_meta(part_name, args):
    meta_info_lst = []
    table2txt_dir = '/home/cc/code/open_table_discovery/table2txt'
    data_file = os.path.join('dataset', args.dataset, args.experiment, 'graph_parts/graph_row_table.txt_' + part_name)
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            meta_info_lst.append(item)
    return meta_info_lst

def get_table_text(strategy, table, expand_info, meta_info, template_text):
    #generate text from template
    generated_text_lst = strategy.expand_template(table, expand_info, meta_info, template_text) 
    #get the original cell text 
    cell_text_lst = strategy.expand_template(table, expand_info, meta_info, None)
    assert(len(generated_text_lst) == len(cell_text_lst))
    
    passage_info_lst = []
    for idx, generated_text_info in enumerate(generated_text_lst):
        cell_text_info = cell_text_lst[idx]
        cell_text = cell_text_info['text']
        generated_text = generated_text_info['text']
        
        assert(cell_text_info['row'] == generated_text_info['row'])
        sub_col_1 = (cell_text_info['sub_col'] if cell_text_info['sub_col'] is not None else -1)
        sub_col_2 = (generated_text_info['sub_col'] if generated_text_info['sub_col'] is not None else -1)
       
        assert(sub_col_1 == sub_col_2)
        assert(cell_text_info['obj_col'] == generated_text_info['obj_col'])

        row = generated_text_info['row']

        sub_col = meta_info['sub_col']
        obj_col = meta_info['obj_col']

        passage_info = get_passage_info(table, row, sub_col, obj_col, cell_text, generated_text)
        passage_info_lst.append(passage_info)
    return passage_info_lst

def get_passage_info(table, row, sub_col, obj_col, cell_text, generated_text):
    title = table['documentTitle']
    passage = title + '   .   ' +  cell_text + '   .   ' + generated_text
    passage_info = {
        'text':passage,
        'row':row,
        'sub_col':sub_col,
        'obj_col':obj_col
    }
    return passage_info

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

def main():
    args = get_args()
    table_dict = read_tables(args) 
    strategy = get_strategy(args.strategy)    
    
    graph_parts_dir = os.path.join('dataset', args.dataset, args.experiment, 'graph_parts')
    model_out_dir = '/home/cc/code/plms_graph2text/webnlg/outputs/test_model'
    passage_dir = os.path.join(model_out_dir, args.dataset, args.experiment)
    part_name_lst = sorted(os.listdir(passage_dir))

    out_passage_file = os.path.join('dataset', args.dataset, args.experiment, 'graph_passages.json') 
    f_o = open(out_passage_file, 'w')
    table_expand_info = {}
    for part_name in tqdm(part_name_lst):
        graph_data_file = os.path.join(graph_parts_dir, 'test_unseen.source_%s' % part_name)

        template_meta_data = get_template_meta(part_name, args)
        preds_file = os.path.join(passage_dir, part_name, 'val_outputs/test_unseen_predictions.txt.debug')
        with open(preds_file) as f:
            p_id = 0
            for row, text in enumerate(f):
                template_graph_text = text.rstrip()
                meta_info = template_meta_data[row] 
                table_id = meta_info['table_id']
                table_data = table_dict[table_id]
                if table_id not in table_expand_info:
                    table_expand_info[table_id] = strategy.get_expand_info(table_data)

                table_passage_lst = get_table_text(strategy, table_data, table_expand_info[table_id], 
                                                   meta_info, template_graph_text)
                for passage_info in table_passage_lst:
                    tag_info = {
                        'table_id':table_id,
                        'row':passage_info['row'],
                        'sub_col':passage_info['sub_col'],
                        'obj_col':passage_info['obj_col']
                    }
                    passage = passage_info['text']
                    out_item = {
                        'id': part_name,
                        'p_id': p_id,
                        'passage': passage,
                        'tag': tag_info
                    }
                    p_id += 1
                    f_o.write(json.dumps(out_item) + '\n')
    f_o.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--experiment', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
