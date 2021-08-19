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
        print('[%s] already exists, please use a different value for [--expt_dir].\nThe full path [%s]\n'
              % (args.out_dir, expt_path))
        return None
    os.makedirs(expt_path)
    log_path = os.path.join(expt_path, 'log.txt')
    file_hander = logging.FileHandler(log_path, 'w')
    logger.addHandler(file_hander)
    return True

def get_open_qa():
    open_qa = OpenQA(
        ir_host='127.0.0.1',
        ir_port=9200,
        ir_index='nq_tables_graph_table_name_txt',
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

def search(open_qa, query):
    question = open_qa.process_question(query['question'])
    qry_question = open_qa.get_qry_question(question)
    sub_entity = None
    if 'subject' in query:
        sub_entity = query['subject']
    retr_source_data = open_qa.ir_ranker.search(index_name=open_qa.ir_index,
                                                    question=qry_question,
                                                    entity=sub_entity,
                                                    k=30,
                                                    ret_src=True)
    top_ir_passages = [a['body'] for a in retr_source_data]
    passage_tags = [a['tag'] for a in retr_source_data]
    return (top_ir_passages, passage_tags)

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
    #out_file = os.path.join(args.out_dir, 'preds_%s.json' % args.mode)
    #f_o = open(out_file, 'w')
    for query_info in tqdm(query_info_lst): 
        top_ir_passages, passage_tags = search(open_qa, query_info)
        for p_id, passage in enumerate(top_ir_passages):
            #f_o.write(json.dumps(qry_ret) + '\n')
            qid = query_info['qid']
            query_info = query_info_dict[qid]
            gold_table_id_lst = query_info['table_id_lst']
            for k in k_lst:
                top_k_table_id_lst = passage_tags[:k]
                correct = table_found(top_k_table_id_lst, gold_table_id_lst)
                correct_retr_dict[k].append(correct)

    for k in correct_retr_dict:
        precision = np.mean(correct_retr_dict[k]) * 100
        logger.info('p@%d = %.2f' % (k, precision))

    #f_o.close()

if __name__ == '__main__':
    main()
