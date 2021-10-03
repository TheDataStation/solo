import json
import os
from tqdm import tqdm
import argparse

def get_passage_tables(part_name, args):
    table_id_lst = []
    table2txt_dir = '/home/cc/code/open_table_discovery/table2txt'
    data_file = os.path.join('dataset', args.dataset, args.experiment, 'graph_parts/graph_row_table.txt_' + part_name)
    with open(data_file) as f:
        for line in f:
            table_id = line.rstrip()
            table_id_lst.append(table_id)
    return table_id_lst

def get_passage(table_id, graph_text, graph_tokens):
    table_id_updated = table_id.replace('_', ' ').replace('-', ' ')

    graph_tokens_updated = graph_tokens.replace('<H>', ' ').replace('<R>', ' ').replace('<T>', ' , ')

    passage = table_id_updated + ' . ' + graph_tokens_updated + ' (). ' + graph_text
    return passage

def get_graph_tokens(data_file):
    graph_tokens_lst = []
    with open(data_file) as f:
        for row, line in enumerate(f):
            graph_tokens = line.rstrip()
            graph_tokens_lst.append(graph_tokens)
    return graph_tokens_lst

def main():
    args = get_args()

    graph_parts_dir = os.path.join('dataset', args.dataset, args.experiment, 'graph_parts')

    model_out_dir = '/home/cc/code/plms_graph2text/webnlg/outputs/test_model'
    passage_dir = os.path.join(model_out_dir, args.dataset, args.experiment)
    part_name_lst = os.listdir(passage_dir)

    out_passage_file = os.path.join('dataset', args.dataset, args.experiment, 'graph_passages.json') 
    f_o = open(out_passage_file, 'w')
    for part_name in tqdm(part_name_lst):
        graph_data_file = os.path.join(graph_parts_dir, 'test_unseen.source_%s' % part_name)
        graph_tokens_lst = get_graph_tokens(graph_data_file)

        passage_tables = get_passage_tables(part_name, args)
        preds_file = os.path.join(passage_dir, part_name, 'val_outputs/test_unseen_predictions.txt.debug')
        with open(preds_file) as f:
            for row, text in enumerate(f):
                graph_text = text.rstrip()
                table_id = passage_tables[row]
                graph_tokens = graph_tokens_lst[row]
                passage = get_passage(table_id, graph_text, graph_tokens)
                #passage = graph_text
                out_item = {
                    'id': part_name,
                    'p_id': row,
                    'passage': passage,
                    'tag': table_id
                }
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
