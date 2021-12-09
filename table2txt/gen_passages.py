import json
import os
from tqdm import tqdm
import argparse
import re
from table2txt import utils

def get_template_meta(part_name, args):
    meta_info_lst = []
    table2txt_dir = '/home/cc/code/open_table_discovery/table2txt'
    data_file = os.path.join('dataset', args.dataset, args.experiment, 'graph_parts/graph_row_table.txt_' + part_name)
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            meta_info_lst.append(item)
    return meta_info_lst

def remove_tags(graph_tokens):
    src_lst = [utils.Ent_Start_Tag, utils.Rel_Tag, utils.Ent_End_Tag]
    target_replace = ' '
    graph_tokens_updated = graph_tokens
    for src in src_lst:
        src_replace = re.compile(src)
        graph_tokens_updated = src_replace.sub(target_replace, graph_tokens_updated)
    
    src_replace = re.compile(utils.Tuple_Start_Tag)
    graph_tokens_updated = src_replace.sub(',', graph_tokens_updated)
    return graph_tokens_updated  

def get_table_text(table, meta_info, template_text, graph_tokens):
    output_text_lst = apply_template(table, meta_info, template_text)
    input_text_lst = apply_template(table, meta_info, graph_tokens)
    assert(len(output_text_lst) == len(input_text_lst))
    passage_text_lst = []
    for idx, input_text_info in enumerate(input_text_lst):
        output_text_info = output_text_lst[idx]
        input_text = input_text_info['text']
        output_text = output_text_info['text']
        assert(input_text_info['row'] == output_text_info['row'])
        row = output_text_info['row']
        passage_info = get_passage_info(table, row, output_text, input_text)
        passage_text_lst.append(passage_info)
    return passage_text_lst

def apply_template(table, meta_info, template_text):
    span_info_lst = utils.read_template(table, meta_info, template_text)
    out_text_lst = []
    row_lst = table['rows']
    for row_idx, row_item in enumerate(row_lst):
        cell_lst = row_item['cells']
        out_span_text_lst = []
        for span_info in span_info_lst:
            if not span_info['is_template']:
                out_span_text_lst.append(span_info['text'])
            else:
                ent_idx = span_info['ent_idx']
                ent_text = cell_lst[ent_idx-1]['text']
                out_span_text_lst.append(ent_text)

        out_item_text = ''.join(out_span_text_lst)
        out_item_info = {'row':row_idx, 'text':out_item_text}
        out_text_lst.append(out_item_info) 

    return out_text_lst 

def get_passage_info(table, row, graph_text, graph_tokens):
    title = table['documentTitle']
    graph_tokens_updated = remove_tags(graph_tokens)
    passage = title + ' . ' + graph_tokens_updated + ' (). ' + graph_text
    passage_info = {
        'row': row,
        'text': passage
    }
    return passage_info

def get_graph_tokens(data_file):
    graph_tokens_lst = []
    with open(data_file) as f:
        for row, line in enumerate(f):
            graph_tokens = line.rstrip()
            graph_tokens_lst.append(graph_tokens)
    return graph_tokens_lst

def read_tables(args):
    table_data_file = os.path.join('/home/cc/data', args.dataset, 'tables', 'tables.jsonl') 
    table_lst = []
    utils.read_table_file(table_lst, table_data_file, None)
    table_dict = {}
    for table in table_lst:
        table_id = table['tableId']
        table_dict[table_id] = table
    return table_dict

def main():
    args = get_args()
    table_dict = read_tables(args) 
    
    graph_parts_dir = os.path.join('dataset', args.dataset, args.experiment, 'graph_parts')
    model_out_dir = '/home/cc/code/plms_graph2text/webnlg/outputs/test_model'
    passage_dir = os.path.join(model_out_dir, args.dataset, args.experiment)
    part_name_lst = sorted(os.listdir(passage_dir))

    out_passage_file = os.path.join('dataset', args.dataset, args.experiment, 'graph_passages.json') 
    f_o = open(out_passage_file, 'w')
    for part_name in tqdm(part_name_lst):
        graph_data_file = os.path.join(graph_parts_dir, 'test_unseen.source_%s' % part_name)
        graph_tokens_lst = get_graph_tokens(graph_data_file)

        template_meta_data = get_template_meta(part_name, args)
        preds_file = os.path.join(passage_dir, part_name, 'val_outputs/test_unseen_predictions.txt.debug')
        with open(preds_file) as f:
            p_id = 0
            for row, text in enumerate(f):
                template_graph_text = text.rstrip()
                meta_info = template_meta_data[row] 
                table_id = meta_info['table_id']
                graph_tokens = graph_tokens_lst[row]
                table_data = table_dict[table_id]
                table_passage_lst = get_table_text(table_data, meta_info, template_graph_text, graph_tokens)
                for passage_info in table_passage_lst:
                    tag_info = {
                        'table_id': table_id,
                        'row': passage_info['row']
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
    parser.add_argument('--experiment', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
