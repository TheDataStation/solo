import json
from tqdm import tqdm
import glob
import torch
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    return args

def get_qas_data(args):
    qas_lst = []
    data_file = '/home/cc/data/%s/interactions/%s_qas.jsonl' % (args.dataset, args.mode)
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            qas_lst.append(item)
    return qas_lst
 
def main():
    args = get_args()
    out_data_file = './fusion_query_%s.json' % args.mode
    f_o = open(out_data_file, 'w')
    qas_lst = get_qas_data(args)
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
