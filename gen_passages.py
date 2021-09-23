import json
import os
from tqdm import tqdm

def get_passage_tables(part_name):
    table_id_lst = []
    data_file = '/home/cc/code/open_table_discovery/table2txt/output/fetaqa/graph_topic_entity/graph_parts/graph_row_table.txt_' + part_name
    with open(data_file) as f:
        for line in f:
            table_id = line.rstrip()
            table_id_lst.append(table_id)
    return table_id_lst

def get_passage(table_id, graph_text):
    #table_id_updated = table_id.replace('_', ' ').replace('-', ' ')
    passage = table_id + '. ' + graph_text
    return passage

def main():
    passage_dir = '/home/cc/code/plms_graph2text/webnlg/outputs/test_model/fetaqa/expt_topic_entity'
    part_name_lst = os.listdir(passage_dir)

    f_o = open('./table2txt/output/fetaqa/graph_topic_entity/graph_passages.json', 'w')
    for part_name in tqdm(part_name_lst):
        passage_tables = get_passage_tables(part_name)
        preds_file = os.path.join(passage_dir, part_name, 'val_outputs/test_unseen_predictions.txt.debug')
        with open(preds_file) as f:
            for row, text in enumerate(f):
                graph_text = text.rstrip()
                table_id = passage_tables[row]
                #passage = get_passage(table_id, graph_text)
                passage = graph_text
                out_item = {
                    'id': part_name,
                    'p_id': row,
                    'passage': passage,
                    'tag': table_id
                }
                f_o.write(json.dumps(out_item) + '\n')
    f_o.close()

if __name__ == '__main__':
    main()
