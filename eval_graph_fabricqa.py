import json
import argparse
import os
from predictor import OpenQA
from tqdm import tqdm
import numpy as np
import logging

def set_logger(args):
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console = logging.StreamHandler()
    logger.addHandler(console)
    expt_path = os.path.join(args.out_dir)
    if os.path.isdir(expt_path):
        err_msg = ('[%s] already exists, please use a different value for [--expt_dir].\nThe full path [%s]\n'
              % (args.out_dir, expt_path))
        raise ValueError(err_msg)
    os.makedirs(expt_path)
    log_path = os.path.join(expt_path, 'log.txt')
    file_hander = logging.FileHandler(log_path, 'w')
    logger.addHandler(file_hander)
    return True

def get_open_qa():
    open_qa = OpenQA(
        ir_host='127.0.0.1',
        ir_port=9200,
        ir_index='fetaqa_tables',
        model_dir='/home/cc/code/fabric_qa/model',
        cuda=0)
    return open_qa

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    return args

def get_questions(mode):
    q_item_lst = []
    qas_file = '/home/cc/data/FeTaQA/data/tf_records/interactions/%s_qas.jsonl' % mode
    with open(qas_file) as f:
        for line in f:
            q_item = json.loads(line)
            q_item_lst.append(q_item)
    return q_item_lst

def table_found(top_k_table_id_lst, gold_table_id_lst):
    for table_id in top_k_table_id_lst:
        if table_id in gold_table_id_lst:
            return 1
    return 0 

def get_top_k_tables(table_name_lst, K):
    top_tables = []
    table_dict = {}
    N = len(table_name_lst)
    for idx, table_name in enumerate(table_name_lst):
        if table_name not in table_dict:
            table_info = {
                'name':table_name,
                'start_idx':idx,
                'end_idx':idx
            }
            top_tables.append(table_info)
            table_dict[table_name] = table_info
        else:
            table_dict[table_name]['end_idx'] = idx

    #if len(top_tables) < K:
    #    raise ValueError('Not enough tables')
    
    top_k_tables = top_tables[:K]
    return top_k_tables

def main():
    args = get_args()
    set_logger(args)
    open_qa = get_open_qa()
    query_info_lst = get_questions(args.mode)
    query_info_dict = {}
    for query_info in query_info_lst:
        query_info_dict[query_info['qid']] = query_info 
    k_lst = [1, 5]
    correct_retr_dict = {}
    for k in k_lst:
        correct_retr_dict[k] = []
    out_file = os.path.join(args.out_dir, 'preds_%s.json' % args.mode)
    f_o = open(out_file, 'w')
    for query_info in tqdm(query_info_lst):
        batch_queries = [query_info] 
        qry_ret_lst = open_qa.search(batch_queries, ir_retr_num=100, pr_retr_num=30, top_num=30)
        for qry_ret in qry_ret_lst:
            qid = qry_ret['qid']
            query_info = query_info_dict[qid]
            gold_table_id_lst = query_info['table_id_lst']
            for k in k_lst:
                top_k_table_info_lst = get_top_k_tables(qry_ret['passage_tags'], k)
                if k == 5:
                    qry_ret['top_5_tables'] = top_k_table_info_lst
                top_k_table_id_lst = [a['name'] for a in top_k_table_info_lst] 
                correct = table_found(top_k_table_id_lst, gold_table_id_lst)
                correct_retr_dict[k].append(correct)
           
            show_precison(correct_retr_dict) 
            f_o.write(json.dumps(qry_ret) + '\n')


    show_precison(correct_retr_dict)
    f_o.close()

def show_precison(correct_retr_dict):
    for k in correct_retr_dict:
        precision = np.mean(correct_retr_dict[k]) * 100
        logger.info('p@%d = %.2f' % (k, precision))
    

if __name__ == '__main__':
    main()
