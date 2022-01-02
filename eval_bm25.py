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

def search_in_table(question, tag_dict, ir_ranker, index_name):
    ret_passage_info_lst = []
    for tag_key in tag_dict:
        tag_info = tag_dict[tag_key]
        table_id = tag_info['table_id']
        row = tag_info['row']
        result = ir_ranker.search_in_table(index_name=index_name,
                                           question=question,
                                           table_id=table_id,
                                           row=row,
                                           k=20)
         
        for item in result:
            passage_info = search_result_to_passage_info(item)
            assert(passage_info['tag']['table_id'] == table_id)
            assert(passage_info['tag']['row'] == row)
            ret_passage_info_lst.append(passage_info) 

    return ret_passage_info_lst

def group_by_tables(passage_info_lst, question, ir_ranker, index_name):
    passage_group_dict = {}
    table_id_lst = []
    for idx, passage_info in enumerate(passage_info_lst):
        table_id = passage_info['tag']['table_id']
        if table_id not in passage_group_dict:
            passage_group_dict[table_id] = []
            table_id_lst.append(table_id)
        table_passages = passage_group_dict[table_id]
        table_passages.append(passage_info)

    table_score_lst = []
    for table_id in table_id_lst:
        top_table_passages = passage_group_dict[table_id][:3]
        top_scores = [a['score'] for a in top_table_passages]
        mean_score = np.mean(top_scores)
        table_score_lst.append(mean_score)
    
    top_idxes = np.argsort(-np.array(table_score_lst))
    
    top_m_idxes = top_idxes[:20]
    top_table_id_lst =[table_id_lst[a] for a in top_m_idxes]

    ret_passage_info_lst = []
    for top_idx in top_m_idxes:
        table_id = table_id_lst[top_idx]
        table_passage_info_lst = passage_group_dict[table_id][:3]
        #ret_passage_info_lst.extend(table_passage_info_lst)
        
        tag_dict = get_tag_dict(table_passage_info_lst)
        table_second_results = search_in_table(question, tag_dict, ir_ranker, index_name)
        ret_passage_info_lst.extend(table_second_results) 

    #updated_passage_info_lst = remove_duplicate_passages(ret_passage_info_lst)
    return ret_passage_info_lst, top_table_id_lst

def remove_duplicate_passages(passage_info_lst):
    passage_id_dict = {}
    ret_passage_info_lst = []
    for passage_info in passage_info_lst:
        if passage_info['id'] not in passage_id_dict:
            passage_id_dict['id'] = True
            ret_passage_info_lst.append(passage_info)
    return ret_passage_info_lst 

def get_tag_dict(table_passage_info_lst):
    tag_dict = {}
    for passage_info in table_passage_info_lst:
        tag_info = passage_info['tag']
        table_id = tag_info['table_id']
        row = tag_info['row']
        tag_key = '%s_%d' % (table_id, row)
        if tag_key not in tag_dict:
            tag_dict[tag_key] = tag_info
    return tag_dict


def search_result_to_passage_info(item):
    passage_info = {
        'id':item['_id'],
        'passage':item['_source']['body'],
        'tag':{
            'table_id':item['_source']['table_id'],
            'row':item['_source']['row'],
        },
        'score':item['_score']
    }
    return passage_info

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
  
    passage_info_lst = [] 
    for item in retr_source_data:
        passage_info = search_result_to_passage_info(item)
        passage_info_lst.append(passage_info)
    
     
    result = group_by_tables(passage_info_lst, qry_question, open_qa.ir_ranker, open_qa.ir_index)
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
    k_lst = [1, 5, 10, 20]
    correct_retr_dict = {}
    for k in k_lst:
        correct_retr_dict[k] = []
    out_file = os.path.join(args.out_dir, 'preds_%s.json' % args.mode)
    f_o = open(out_file, 'w')
    step = 0

    num_tables = 0
    for query_info in tqdm(query_info_lst): 
        ret_passage_info_lst, top_table_id_lst = search(open_qa, query_info, max_k)
        qid = query_info['qid']
        query_info = query_info_dict[qid]
        gold_table_id_lst = query_info['table_id_lst']
        
        table_set = set(top_table_id_lst)
        num_tables = len(table_set)

        num_passages = len(ret_passage_info_lst)

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
        write_data(f_o, qid, question, ret_passage_info_lst) 

    show_precision(correct_retr_dict, num_tables, num_passages)
    f_o.close()

def write_data(f_o, qid, question, ret_passage_info_lst):
    passage_lst = [a['passage'] for a in ret_passage_info_lst]
    passage_tags = [a['tag'] for a in ret_passage_info_lst]
    passage_scores = [a['score'] for a in ret_passage_info_lst]
    out_item = {
        'qid':qid,
        'question':question,
        'passages':passage_lst,
        'passage_tags':passage_tags,
        'scores':passage_scores
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
