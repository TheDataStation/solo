import json
import argparse
import os
from tqdm import tqdm
import numpy as np
import logging
import csv
from eval_retrieval_graph import table_found

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
    parser.add_argument('--mode', type=str)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    return args

def get_questions(mode):
    q_item_lst = []
    qas_file = '/home/cc/code/open_table_discovery/qas/nq_%s_qas.json' % mode
    with open(qas_file) as f:
        for line in f:
            q_item = json.loads(line)
            q_item_lst.append(q_item)
    return q_item_lst

def read_qry_result(mode):
    data_file = '/home/cc/code/open_table_discovery/output/graph_table_name_%s/preds_%s.json' % (mode, mode)
    ret_dict = {}
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            ret_dict[item['qid']] = item
    return ret_dict

def list2str(data_lst):
    data_str = '\n'.join([str(a) for a in data_lst])
    return data_str

def read_tapas_retrieval(mode):
    data_file = '/home/cc/code/tapas/models/tapas_nq_hn_retriever_large/%s_knn_small.jsonl' % mode
    ret_dict = {}
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['query_id']
            table_info_lst = item['table_scores'][:5]
            top_table_id_lst = [a['table_id'] for a in table_info_lst]
            ret_dict[qid] = top_table_id_lst
    return ret_dict

def main():
    args = get_args()
    set_logger(args)
    query_info_lst = get_questions(args.mode)
    query_info_dict = {}
    for query_info in query_info_lst:
        query_info_dict[query_info['qid']] = query_info 
    k_lst = [1, 5]
    correct_retr_dict = {}
    for k in k_lst:
        correct_retr_dict[k] = []
    out_file = os.path.join(args.out_dir, 'graph_preds_log_%s.csv' % args.mode)
    f_o = open(out_file, 'w')
    writer = csv.writer(f_o, delimiter='\t')
    col_name_lst = ['qid', 'question', 'order', 'passage', 
                    'gold (graph2txt)', 'graph2txt correct?', 
                    'OpenTableQA', 'OpenTableQA correct?', 'answers']
    writer.writerow(col_name_lst)
    qry_result = read_qry_result(args.mode)
    tapas_ret = read_tapas_retrieval(args.mode)
    out_script_file = './output/cp_ref_tables.sh'
    f_o_script = open(out_script_file, 'w')
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
        
        if q_correct_dict[1]: # ignore questions whose retrievaled top 1 table is correct. 
            continue 

        tapas_table_id_lst = tapas_ret[query_info['qid']]
        if tapas_table_id_lst[0] not in gold_table_id_lst:
            continue

        csv_q_info = [
            query_info['qid'],
            query_info['question'],
            '',
            '',
            list2str(query_info['table_id_lst']),
            '',
            '',
            '',
            list2str(query_info['answers'])
        ]
        writer.writerow(csv_q_info)

        passage_lst = qry_ret['passages']
        passage_table_lst = qry_ret['passage_tags']
        answer_lst = qry_ret['answers']

        all_ref_table_lst = gold_table_id_lst + passage_table_lst
        for ref_table_id in all_ref_table_lst:
            if '/' in ref_table_id:
                ref_table_id = ref_table_id.replace('/', '[left-slash]')
            f_o_script.write('cp ./tables_csv/"%s.csv" ./tables_ref \n' % ref_table_id)
        
        for idx, passage in enumerate(passage_lst):
            graph2txt_correct = ('Y' if passage_table_lst[idx] in gold_table_id_lst else '')
            open_table_qa_correct = ('Y' if tapas_table_id_lst[idx] in gold_table_id_lst else '') 
            csv_passage_info = [
                '',
                '',
                (idx + 1),
                passage,
                passage_table_lst[idx],
                graph2txt_correct,
                tapas_table_id_lst[idx],
                open_table_qa_correct,
                answer_lst[idx]['answer']
            ]
            writer.writerow(csv_passage_info) 

    for k in correct_retr_dict:
        precision = np.mean(correct_retr_dict[k]) * 100
        logger.info('p@%d = %.2f' % (k, precision))

    f_o.close()
    f_o_script.close()

if __name__ == '__main__':
    main()
