import json
import argparse
import os
from tqdm import tqdm
import numpy as np
import logging
import csv
from eval_graph_fabricqa import table_found
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
    parser.add_argument('--open_table_qa_retr', type=str)
    parser.add_argument('--experiment', type=str)

    args = parser.parse_args()
    return args

def get_questions(args):
    q_item_lst = []
    qas_file = os.path.join('/home/cc/data', args.dataset, 'interactions/dev_qas.jsonl')
    with open(qas_file) as f:
        for line in f:
            q_item = json.loads(line)
            q_item_lst.append(q_item)
    return q_item_lst

def read_qry_result(args):
    data_file = os.path.join('./dataset', args.dataset, args.experiment, 'dev/preds_dev.json')
    ret_dict = {}
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            ret_dict[item['qid']] = item
    return ret_dict

def list2str(data_lst):
    data_str = '\n'.join([str(a) for a in data_lst])
    return data_str

def read_bm25_retrieval(args):
    data_file = os.path.join('./dataset', args.dataset, 'bm25_' + args.experiment, 'dev/preds_dev.json') 
    ret_dict = {}
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            ret_dict[qid] = item
    return ret_dict

def read_open_table_qa_retr(args):
    retr_dict = {}
    with open(args.open_table_qa_retr) as f:
        for line in f:
            item = json.loads(line)
            qid = item['query_id']
            top_lst = item['table_scores'][:5]  
            top_table_lst = [a['table_id'] for a in top_lst]
            retr_dict[qid] = top_table_lst
    return retr_dict 
     
def main():
    args = get_args()
    set_logger(args)
    out_dir = os.path.join('dataset', args.dataset, args.experiment, 'dev/analysis_log')
    if os.path.isdir(out_dir):
        print('[%s] already exists\n' % out_dir)
        return None
    os.mkdir(out_dir)
    query_info_lst = get_questions(args)
    query_info_dict = {}
    for query_info in query_info_lst:
        query_info_dict[query_info['qid']] = query_info 
    k_lst = [1, 5]
    correct_retr_dict = {}
    for k in k_lst:
        correct_retr_dict[k] = []
    out_file = os.path.join(out_dir, 'retr_log_%s_%s.csv' % (args.dataset, args.experiment))
    f_o = open(out_file, 'w')
    writer = csv.writer(f_o, delimiter=',')
    col_name_lst = ['qid', 'question', 'order', 'fabricqa passage', 
                    'gold (fabricqa) table', 'fabricqa correct?', 'gold (fabricqa) answer',
                    'open_table_qa table', 'open_table_qa correct?',
                    'bm25 passage', 'bm25 table', 'bm25 correct?',
                    'found in top 100 of bm25']
    writer.writerow(col_name_lst)
    qry_result = read_qry_result(args)
    em25_ret = read_bm25_retrieval(args)
    open_table_qa_ret = read_open_table_qa_retr(args)
    log_qid_lst = []
    for query_info in tqdm(query_info_lst):
        batch_queries = [query_info] 
        qry_ret = qry_result[query_info['qid']]
        qid = qry_ret['qid']
        query_info = query_info_dict[qid]
        gold_table_id_lst = query_info['table_id_lst']
        q_correct_dict = {}
        for k in k_lst:
            top_k_table_id_lst = qry_ret['passage_tags'][:k]
            correct = table_found(top_k_table_id_lst, gold_table_id_lst)
            q_correct_dict[k] = correct
            correct_retr_dict[k].append(correct)
        
        top_open_table_qa_table_lst = open_table_qa_ret[query_info['qid']]
        if q_correct_dict[1]: # ignore questions whose retrievaled top 1 table is correct. 
            continue 

        if top_open_table_qa_table_lst[0] not in gold_table_id_lst:
            continue
   
        found_top_100 = ('Y' if em25_ret[query_info['qid']]['found_top_100'] > 0 else 'N')
        log_qid_lst.append(query_info['qid'])    
        csv_q_info = [
            query_info['qid'],
            query_info['question'],
            '',
            '',
            list2str(query_info['table_id_lst']),
            '',
            list2str(query_info['answers']),
            '',
            '',
            '',
            '',
            '',
            found_top_100
        ]
        writer.writerow(csv_q_info)

        passage_lst = qry_ret['passages']
        passage_table_lst = qry_ret['passage_tags']
        answer_lst = qry_ret['answers']

        top_em25_passages = em25_ret[query_info['qid']]['passages']
        top_em25_table_lst = em25_ret[query_info['qid']]['passage_tags']
        top_5_passage_lst = passage_lst[:5]
        for idx, passage in enumerate(top_5_passage_lst):
            fabricqa_correct = ('Y' if passage_table_lst[idx] in gold_table_id_lst else 'N')
            em25_correct = ('Y' if top_em25_table_lst[idx] in gold_table_id_lst else 'N') 
            open_table_qa_correct = ('Y' if top_open_table_qa_table_lst[idx] in gold_table_id_lst else 'N') 
            csv_passage_info = [
                '',
                '',
                (idx + 1),
                passage,
                passage_table_lst[idx],
                fabricqa_correct,
                answer_lst[idx]['answer'],
                top_open_table_qa_table_lst[idx],
                open_table_qa_correct,
                top_em25_passages[idx],
                top_em25_table_lst[idx],
                em25_correct,
                ''
            ]
            writer.writerow(csv_passage_info) 

    for k in correct_retr_dict:
        precision = np.mean(correct_retr_dict[k]) * 100
        logger.info('p@%d = %.2f' % (k, precision))

    '''
    sample_qid_lst = random.sample(log_qid_lst, 100)
    qid_file_name = os.path.join(args.out_dir, 'examples.txt')
    with open(qid_file_name, 'w') as f_o_qid:
        for sample_qid in sample_qid_lst:
            f_o_qid.write(sample_qid + '\n')
    '''

    f_o.close()

if __name__ == '__main__':
    main()
