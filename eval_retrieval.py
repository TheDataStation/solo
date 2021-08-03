import logging
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
import argparse
import os
from tqdm import tqdm
from fabric_qa.ir_ranker import IRRanker
import json
import numpy as np
import torch
import time
import spacy

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

def read_qas(args):
    quest_data = []
    with open(args.qas_file) as f:
        for line in f:
            item = json.loads(line)
            quest_data.append(item)
    return quest_data 

def get_ir(args):
    eshost = dict()
    eshost['host'] = args.host
    eshost['port'] = args.port
    host_lst = [eshost]
    ir_ranker = IRRanker(hosts=host_lst)
    return ir_ranker

def get_qry_question(question):
    doc = spacy_nlp(question)
    tokens = [str(a) for a in doc]
    qry_question = ' '.join(tokens)
    return qry_question

def eval_data(args):
    ir_ranker = get_ir(args)
    q_data = read_qas(args)
    k_lst = [1, 5]
    correct_retr_dict = {}
    for k in k_lst:
        correct_retr_dict[k] = []
    for q_item in tqdm(q_data):
        qid = q_item['qid']
        question = q_item['question']
        qry_question = get_qry_question(question)
        sub_entity = q_item['subject'] if 'subject' in q_item else ''
        top_src_data = ir_ranker.search(index_name=args.index_name, question=qry_question,
                                           entity=sub_entity, k=args.top_ir, ret_src=True)

        gold_table_id_lst = q_item['table_id_lst']        
        top_table_id_lst = [a['tag'] for a in top_src_data]
        for k in [1, 5]:
            top_k_table_id_lst = top_table_id_lst[:k]
            correct = table_found(top_k_table_id_lst, gold_table_id_lst)
            correct_retr_dict[k].append(correct)
    
    for k in correct_retr_dict:
        precision = np.mean(correct_retr_dict[k]) * 100
        logger.info('p@%d = %.2f' % (k, precision)) 

def table_found(top_k_table_id_lst, gold_table_id_lst):
    for table_id in top_k_table_id_lst:
        if table_id in gold_table_id_lst:
            return 1
    return 0 


def get_question_info(question_doc):
    tokens = [token.text for token in question_doc]
    doc_tensor = question_doc.tensor.tolist()
    return [tokens, doc_tensor]


def log_args(args):
    arg_info = ''
    for arg in vars(args):
        arg_info += (arg + ' = ' + str(getattr(args, arg)) + '\n')
    logger.info(arg_info)

def main():
    args = get_args()
    ret = set_logger(args)
    if not ret:
        return
   
    global spacy_nlp
    spacy_nlp = spacy.load("en_core_web_sm")
    
    log_args(args)
    eval_data(args)

    logger.info('\nok\n') 

def get_args():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='hostname or ip of ElasticSearch server')
    
    parser.add_argument('--port', type=int, default=9200,
                        help='port of ElasticSearch server')
    
    parser.add_argument('--index_name', type=str, default=None,
                        help='index name of ElasticSearch database')

    parser.add_argument('--qas_file', type=str)

    parser.add_argument('--out_dir', type=str,
                        help='directory for expt output')
    
    parser.add_argument('--top_ir', type=int, default=100,
                        help='number of top passages ir return')
    
    parser.add_argument('--gpu', type=int, default=-1,
                        help='gpu device number')

    args = parser.parse_args()

    return args

if __name__ == '__main__':        
    main()

