import json
import argparse
import os
from tqdm import tqdm
import numpy as np
import logging
import csv
import random

def set_logger(args):
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console = logging.StreamHandler()
    logger.addHandler(console)
    return True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--expr', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--data_file', type=str, default='pred_None.jsonl')
    args = parser.parse_args()
    return args

def read_retr_top_100(args):
    data_dir = '/home/cc/code/open_table_discovery/table2txt/dataset/'
    data_file = os.path.join(data_dir, args.dataset, args.expr, 'fusion_retrieved_dev.jsonl')
    top_100_info = {}
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            gold_table_lst = item['table_id_lst']
            ctx_lst = item['ctxs']
            labels = [int(a['tag']['table_id'] in gold_table_lst) for a in ctx_lst]
            max_p = max(labels)
            qid = item['id']
            top_100_info[qid] = max_p
    return top_100_info

def read_qas(args):
    qas_dict = {}
    data_file = '/home/cc/data/%s/interactions/%s_qas.jsonl' % (args.dataset, args.mode)
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            qas_dict[qid] = item
    return qas_dict

def list2str(data_lst):
    data_str = '\n'.join([str(a) for a in data_lst])
    return data_str

def main():
    args = get_args()
    set_logger(args)
    out_file = 'retr_log.csv'
    f_o = open(out_file, 'w')
    writer = csv.writer(f_o, delimiter=',')
    col_name_lst = ['qid', 'question', 'gold_tables', 'gold_answer',
                    'passage', 'tag', 'correct']
    writer.writerow(col_name_lst)
    
    top_100_info = read_retr_top_100(args)
    qas_dict = read_qas(args)
    ref_table_lst = []
    with open(args.data_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            gold_table_id_lst = item['table_id_lst']
            ref_table_lst.extend(gold_table_id_lst)
            passage_tags = item['tags']
            
            correct = (passage_tags[0]['table_id'] in gold_table_id_lst)
            if correct:
                continue 
            
            csv_q_info = [
                qid,
                item['question'],
                list2str(item['table_id_lst']),
                list2str(qas_dict[qid]['answers']),
                '',
                ''
                ''
            ]
            writer.writerow(csv_q_info)

            max_100_correct = top_100_info[qid]
            passage_lst = item['passages']
            for idx, passage in enumerate(passage_lst):
                pred_table = passage_tags[idx]['table_id']
                ref_table_lst.append(pred_table)
                
                cor_info = '%d, max_100=%d' % (int(pred_table in gold_table_id_lst), max_100_correct)

                csv_passage_info = [
                    '',
                    '',
                    '',
                    '',
                    passage,
                    passage_tags[idx],
                    cor_info
                ]
                writer.writerow(csv_passage_info) 

    f_o.close()

    ref_table_lst = list(set(ref_table_lst))
    with open('ref_tables.txt', 'w') as f_o_tables:
        for table in ref_table_lst:
            f_o_tables.write(table + '\n')

if __name__ == '__main__':
    main()
