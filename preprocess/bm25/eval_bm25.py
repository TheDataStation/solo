import json
import argparse
import os
from tqdm import tqdm
import numpy as np
from fabric_qa.ir_ranker import IRRanker
import logging
import spacy

def set_logger(args, out_dir):
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console = logging.StreamHandler()
    logger.addHandler(console)
    log_path = os.path.join(out_dir, 'log.txt')
    file_hander = logging.FileHandler(log_path, 'w')
    logger.addHandler(file_hander)
    return True

def get_ir(args):
    eshost = dict()
    eshost['host'] = '127.0.0.1'
    eshost['port'] = 9200
    eshost_lst = [eshost]
    ir_ranker = IRRanker(hosts=eshost_lst)
    return ir_ranker

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--part_no', type=str)
    parser.add_argument('--expr', type=str, default='rel_graph')
    parser.add_argument('--synthetic', type=int, required=True)
    parser.add_argument('--min_tables', type=int, default=5)
    args = parser.parse_args()
    return args

def get_query_dir(args):
    if args.synthetic == 0:
        data_dir = '/home/cc/data/%s/query/%s' % (args.dataset, args.mode) 
    else:
        mode_dir = '/home/cc/code/open_table_discovery/table2question/dataset/%s/sql_data/%s' % (args.dataset, args.mode)
        if args.part_no is not None:
            data_dir = os.path.join(mode_dir, args.expr, 'data_parts')
        else:
            data_dir = mode_dir
    return data_dir

def get_questions(args):
    q_item_lst = []
    query_dir = get_query_dir(args)
    if args.part_no is not None:
        file_name = '%s.jsonl' % args.part_no
    else:
        file_name = 'fusion_query.jsonl'
    
    query_file = os.path.join(query_dir, file_name)
    with open(query_file) as f:
        for line in f:
            q_item = json.loads(line)
            if args.synthetic:
                q_item['qid'] = q_item['id']
            q_item_lst.append(q_item)
    return q_item_lst

def process_question(question):
    res = question.replace(u'\xa0', u' ')
    return res

def get_qry_question(spacy_nlp, question):
    doc = spacy_nlp(question)
    tokens = [str(a) for a in doc]
    qry_question = ' '.join(tokens)
    return qry_question

def search_min_tables(args, ir_ranker, qry_question, top_n, max_retr=1000):
    num_retr = top_n
    satified=False
    while (not satified):
        retr_source_data = ir_ranker.search(index_name=args.index_name,
                                        question=qry_question,
                                        entity=None,
                                        k=num_retr,
                                        ret_src=True)
        table_lst = [a['_source']['table_id'] for a in retr_source_data] 
        table_set = set(table_lst)
        if (len(table_set) < args.min_tables) and (num_retr < max_retr):
            num_retr = max_retr
        else:
            satified = True
    return retr_source_data 
     

def search(ir_ranker, query, args, spacy_nlp):
    question = process_question(query['question'])
    qry_question = get_qry_question(spacy_nlp, question)
    retr_source_data = search_min_tables(args, ir_ranker, qry_question, 100) 
    top_ir_passages = [a['_source']['body'] for a in retr_source_data]
    passage_tags = [
                    {
                        'table_id':a['_source']['table_id'], 
                        'p_id':int(a['_id']),
                        'row':a['_source']['row'],
                        'sub_col':a['_source'].get('sub_col', None),
                        'obj_col':a['_source'].get('obj_col', None)
                    } for a in retr_source_data ]
    return (top_ir_passages, passage_tags)

def get_out_dir(args):
    if args.synthetic == 0:
        out_dir = '/home/cc/code/data/%s/query/%s/%s_bm25' % (args.dataset, args.mode, args.expr)
    else:
        mode_dir = '/home/cc/code/open_table_discovery/table2question/dataset/%s/sql_data/%s' % (args.dataset, args.mode)
        out_dir = os.path.join(mode_dir, args.expr)
    return out_dir

def main():
    args = get_args()
    out_dir = get_out_dir(args)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    retr_file_name = 'fusion_retrieved_bm25.jsonl'
    out_file = os.path.join(out_dir, retr_file_name)
    if os.path.exists(out_file):
        print('(%s) already exists.' % out_file)
        return

    spacy_nlp = spacy.load("en_core_web_sm")
    set_logger(args, out_dir)
    ir_ranker = get_ir(args)
    query_info_lst = get_questions(args)
    query_info_dict = {}
    for query_info in query_info_lst:
        query_info_dict[query_info['qid']] = query_info 
    max_top_lst = [1, 3, 5]
    metric_dict = {}
    for max_top in max_top_lst:
        metric_dict[max_top] = []
    f_o = open(out_file, 'w')
    for query_info in tqdm(query_info_lst): 
        top_ir_passages, passage_tags = search(ir_ranker, query_info, args, spacy_nlp)
        qid = query_info['qid']
        query_info = query_info_dict[qid]
        gold_table_id_lst = query_info['table_id_lst']
        retr_table_id_lst = [a['table_id'] for a in passage_tags]
        correct_lst = [int(a in gold_table_id_lst) for a in retr_table_id_lst]
        retr_passage_id_lst = [a['p_id'] for a in passage_tags]
        for max_top in max_top_lst:
            max_correct = max(correct_lst[:max_top])
            metric_dict[max_top].append(max_correct)

        out_item = {
            'id':qid,
            'question':query_info['question'],
            'table_id_lst':gold_table_id_lst,
            'answers':['N/A'],
        }
        out_passage_lst = []
        for retr_idx, retr_passage in enumerate(top_ir_passages):
            tag_info = passage_tags[retr_idx]
            out_passage_info = {
                'id':tag_info['p_id'],
                'title':'',
                'text':retr_passage,
                'tag':tag_info
            }
            out_passage_lst.append(out_passage_info)
        out_item['ctxs'] = out_passage_lst
        f_o.write(json.dumps(out_item) + '\n')

    str_info = ''
    for max_top in metric_dict:
        precision = np.mean(metric_dict[max_top]) * 100
        str_info += 'P@%d=%.2f ' % (max_top, precision)
    logger.info(str_info)

    f_o.close()

if __name__ == '__main__':
    try:
        main()
    except ValueError as e:
        print(e)
