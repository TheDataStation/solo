import json
import argparse
import os
from predictor import OpenQA
from tqdm import tqdm
import numpy as np

def get_open_qa():
    open_qa = OpenQA(
        ir_host='127.0.0.1',
        ir_port=9200,
        ir_index='nq_tables_graph_txt',
        model_dir='/home/cc/code/fabric_qa/model',
        cuda=0)
    return open_qa

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    return args

def get_questions(mode):
    q_item_lst = []
    qas_file = '/home/cc/code/open_table_qa/qas/nq_%s_qas.json' % mode
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

def main():
    args = get_args()
    open_qa = get_open_qa()
    query_info_lst = get_questions(args.mode)
    query_info_dict = {}
    for query_info in query_info_lst:
        query_info_dict[query_info['qid']] = query_info 
    k_lst = [1, 5]
    correct_retr_dict = {}
    for k in k_lst:
        correct_retr_dict[k] = []
    f_o = open('./output/open_qa_preds_graph_%s.json' % args.mode, 'w')
    for query_info in tqdm(query_info_lst): 
        batch_queries = [query_info] 
        qry_ret_lst = open_qa.search(batch_queries, ir_retr_num=100, pr_retr_num=30, top_num=5)
        for qry_ret in qry_ret_lst:
            f_o.write(json.dumps(qry_ret) + '\n')
            qid = qry_ret['qid']
            query_info = query_info_dict[qid]
            gold_table_id_lst = query_info['table_id_lst']
            for k in k_lst:
                top_k_table_id_lst = qry_ret['passage_tags'][:k]
                correct = table_found(top_k_table_id_lst, gold_table_id_lst)
                correct_retr_dict[k].append(correct)

    for k in correct_retr_dict:
        precision = np.mean(correct_retr_dict[k]) * 100
        print('p@%d = %.2f' % (k, precision))

    f_o.close()

if __name__ == '__main__':
    main()
