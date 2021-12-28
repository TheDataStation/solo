import json
from tqdm import tqdm
from multiprocessing import Pool as ProcessPool
import glob
from predictor.ar_predictor import ArPredictor
from fabric_qa.reader.forward_reader.albert.qa_data import data_to_examples
import torch
import numpy as np

dataset_name='nq_tables'

retr_result_file = './dataset/nq_tables/bm25_template_graph/dev/preds_dev.json'
M = 20

def get_qas_data():
    ret_dict = {}
    data_file = '/home/cc/data/' + dataset_name + '/interactions/dev_qas.jsonl'
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            qid = item['qid']
            ret_dict[qid] = item
    return ret_dict
 
def main():
    out_data_file = './dataset/nq_tables/bm25_template_graph/dev/fusion_in_decoder_data.jsonl'
    f_o = open(out_data_file, 'w')
    qas_data_dict = get_qas_data()
    with open(retr_result_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            passage_tags = item['passage_tags']
           
            out_item = {}
            out_item['id'] = item['qid']
            out_item['question'] = item['question']
            qas_item = qas_data_dict[item['qid']]
            out_item['table_id_lst'] = qas_item['table_id_lst']
            out_item['answers'] = qas_item['answers']
            out_passage_lst = []
            item_passage_lst = item['passages']
            for p_idx, item_passage in enumerate(item_passage_lst):
                out_passage = item_passage # row_passage_info['passage']
                out_passage_info = {
                    'title': '',
                    'text': out_passage,
                    'tag': passage_tags[p_idx]
                }
                out_passage_lst.append(out_passage_info)

            out_item['ctxs'] = out_passage_lst
            f_o.write(json.dumps(out_item) + '\n')
    
    f_o.close()

if __name__ == '__main__':
    main()
