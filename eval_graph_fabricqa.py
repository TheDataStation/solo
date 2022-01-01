import json
import argparse
import os
#from predictor import OpenQA
from predictor.ar_predictor import ArPredictor
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

def get_open_qa(args):
    open_qa = OpenQA(
        ir_host='127.0.0.1',
        ir_port=9200,
        ir_index=args.index_name,
        model_dir='/home/cc/code/fabric_qa/model',
        cuda=0)
    return open_qa

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_dir', type=str)
    parser.add_argument('--index_name', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    return args

def get_questions(query_dir, mode):
    q_item_lst = []
    qas_file = os.path.join(query_dir,  '%s_qas.jsonl' % mode)
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

def get_top_k_tables(table_id_lst, K):
    top_tables = []
    table_dict = {}
    N = len(table_id_lst)
    for idx, table_id in enumerate(table_id_lst):
        if table_id not in table_dict:
            table_info = {
                'table_id':table_id,
            }
            top_tables.append(table_info)
            table_dict[table_id] = table_info

    top_k_tables = top_tables[:K]
    return top_k_tables

def read_ir_result():
    result = {}
    with open('./dataset/nq_tables/bm25_template_graph/dev/preds_dev.json') as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            result[qid] = item
    return result

def main():
    args = get_args()
    set_logger(args)
    args.index_name = args.index_name.lower()
    #open_qa = get_open_qa(args)
    query_info_lst = get_questions(args.query_dir, args.mode)
    query_info_dict = {}
    for query_info in query_info_lst:
        query_info_dict[query_info['qid']] = query_info 
    k_lst = [1, 5, 10, 20, 30]
    correct_retr_dict = {}
    for k in k_lst:
        correct_retr_dict[k] = []
    
    ar_predictor = ArPredictor('albert_ensemble')
    
    ir_result_data = read_ir_result()
    step = 0
    for query_info in tqdm(query_info_lst):
        step += 1
        qid = query_info['qid']
        question = query_info['question']
        result_item = ir_result_data[qid]
        
        passage_lst = result_item['passages']
        passage_tags = result_item['passage_tags']
        passage_table_id_lst = [a['table_id'] for a in passage_tags]
        top_idxes = ar_predictor.predict(question, passage_lst, ret_num=30)

        top_passsage_lst = [passage_lst[a] for a in top_idxes]
        top_table_id_lst = [passage_table_id_lst[a] for a in top_idxes]    
        
        query_info = query_info_dict[qid]
        gold_table_id_lst = query_info['table_id_lst']
        for k in k_lst:
            top_k_table_id_lst = top_table_id_lst[:k] 
            correct = table_found(top_k_table_id_lst, gold_table_id_lst)
            correct_retr_dict[k].append(correct)
       
        show_precison(correct_retr_dict) 

    show_precison(correct_retr_dict)

def show_precison(correct_retr_dict):
    str_msg = ''
    for k in correct_retr_dict:
        precision = np.mean(correct_retr_dict[k]) * 100
        str_msg += 'p@%d=%.2f  ' % (k, precision)
    
    logger.info(str_msg)
    

if __name__ == '__main__':
    main()
