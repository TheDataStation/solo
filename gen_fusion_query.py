import json
from tqdm import tqdm
import glob
import torch
import numpy as np

dataset_name='nq_tables'

def get_qas_data():
    qas_lst = []
    data_file = '/home/cc/data/' + dataset_name + '/interactions/dev_qas.jsonl'
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            qas_lst.append(item)
    return qas_lst
 
def main():
    out_data_file = './fusion_query_dev.json'
    f_o = open(out_data_file, 'w')
    qas_lst = get_qas_data()
    for item in tqdm(qas_lst):
        out_item = {}
        out_item['id'] = item['qid']
        out_item['question'] = item['question']
        out_item['table_id_lst'] = item['table_id_lst']
        out_item['answers'] = item['answers']
        out_passage_lst = []
        
        out_passage_info = {
            'title': '',
            'text': 'This is a example passage.',
        }
        out_passage_lst.append(out_passage_info)

        out_item['ctxs'] = out_passage_lst
        f_o.write(json.dumps(out_item) + '\n')
    
    f_o.close()

if __name__ == '__main__':
    main()
