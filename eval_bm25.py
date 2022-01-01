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
    parser.add_argument('--index_name', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    return args

def get_questions(mode):
    q_item_lst = []
    qas_file = '/home/cc/data/nq_tables/interactions/%s_qas.jsonl' % mode
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

def group_by_tables(passage_lst, passage_tag_lst, passage_score_lst):
    passage_group_dict = {}
    table_id_lst = []
    for idx, passage in enumerate(passage_lst):
        table_id = passage_tag_lst[idx]['table_id']
        if table_id not in passage_group_dict:
            passage_group_dict[table_id] = []
            table_id_lst.append(table_id)
        table_passages = passage_group_dict[table_id]
    
        passage_info = {
            'passage':passage,
            'table_id':table_id,
            'tag':passage_tag_lst[idx],
            'score':passage_score_lst[idx]
        }
        table_passages.append(passage_info)

    table_score_lst = []
    
    for table_id in table_id_lst:
        top_table_passages = passage_group_dict[table_id][:3]
        top_scores = [a['score'] for a in top_table_passages]
        mean_score = np.mean(top_scores)
        table_score_lst.append(mean_score)
    
    top_idxes = np.argsort(-np.array(table_score_lst))
    
    merged_passage_lst = []
    merged_tags = []
    top_m_idxes = top_idxes[:30]
    top_table_id_lst =[table_id_lst[a] for a in top_m_idxes]

    for top_idx in top_m_idxes:
        table_id = table_id_lst[top_idx]
        table_passage_info_lst = passage_group_dict[table_id][:3]
        table_passages = [a['passage'] for a in table_passage_info_lst] 
        merged_passage = ' . '.join(table_passages)
        merged_passage_lst.append(merged_passage)
        merged_tag = {'table_id':table_id}
        merged_tags.append(merged_tag)

    return merged_passage_lst, merged_tags, top_table_id_lst
        

def search(open_qa, query, max_k):
    question = open_qa.process_question(query['question'])
    qry_question = open_qa.get_qry_question(question)
    sub_entity = None
    if 'subject' in query:
        sub_entity = query['subject']
    retr_source_data = open_qa.ir_ranker.search(index_name=open_qa.ir_index,
                                                    question=qry_question,
                                                    entity=sub_entity,
                                                    k=max_k,
                                                    ret_src=True)
    top_ir_passages = [a['_source']['body'] for a in retr_source_data]
    passage_tags = [a['_source']['tag'] for a in retr_source_data]
    passage_scores = [a['_score'] for a in retr_source_data]
     
    result = group_by_tables(top_ir_passages, passage_tags, passage_scores)
    return result

def main():
    args = get_args()
    set_logger(args)
    open_qa = get_open_qa(args)
    query_info_lst = get_questions(args.mode)
    query_info_dict = {}
    for query_info in query_info_lst:
        query_info_dict[query_info['qid']] = query_info 
    max_k = 1000
    k_lst = [1, 5, 10, 20, 30]
    correct_retr_dict = {}
    for k in k_lst:
        correct_retr_dict[k] = []
    out_file = os.path.join(args.out_dir, 'preds_%s.json' % args.mode)
    f_o = open(out_file, 'w')
    step = 0

    num_tables = 0
    for query_info in tqdm(query_info_lst): 
        merged_passage_lst, merged_tag_lst, top_table_id_lst = search(open_qa, query_info, max_k)
        qid = query_info['qid']
        query_info = query_info_dict[qid]
        gold_table_id_lst = query_info['table_id_lst']
        
        passage_table_id_lst = [a['table_id'] for a in merged_tag_lst]

        table_set = set(passage_table_id_lst)
        num_tables = len(table_set)

        num_passages = len(merged_passage_lst)

        correct_info = {}
        for k in k_lst:
            top_k_table_id_lst = top_table_id_lst[:k]
            correct = table_found(top_k_table_id_lst, gold_table_id_lst)
            correct_info[k] = correct
            correct_retr_dict[k].append(correct)

        step += 1
        if step % 10 == 0: 
            show_precision(correct_retr_dict, num_tables, num_passages)
       
        question = query_info['question'] 
        write_data(f_o, qid, question, merged_passage_lst, merged_tag_lst) 

    show_precision(correct_retr_dict, num_tables, num_passages)
    f_o.close()

def write_data(f_o, qid, question, passage_lst, passage_tags):
    out_item = {
        'qid':qid,
        'question':question,
        'passages':passage_lst,
        'passage_tags':passage_tags
    }
    f_o.write(json.dumps(out_item) + '\n') 

def show_precision(correct_retr_dict, num_tables, num_passages):
    str_msg = ''
    for k in correct_retr_dict:
        precision = np.mean(correct_retr_dict[k]) * 100
        str_msg += ' p@%d = %.2f' % (k, precision)
    str_msg += ' num_tables=%d num_passages=%d' % (num_tables, num_passages)
    logger.info(str_msg)
    

if __name__ == '__main__':
    main()
