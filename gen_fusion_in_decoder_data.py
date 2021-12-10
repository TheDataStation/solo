import json
from tqdm import tqdm
from multiprocessing import Pool as ProcessPool
import glob
from predictor import OpenQA
from fabric_qa.reader.forward_reader.albert.qa_data import data_to_examples
import torch
import numpy as np

open_qa_result_file = './dataset/fetaqa/template_graph/dev/preds_dev.json'
M = 5

def read_passage_file(arg_info):
    passage_file = arg_info[0]
    tag_dict = arg_info[1]
    passage_dict = {}
    with open(passage_file) as f:
        for line in f:
            item = json.loads(line)
            tag = item['tag']
            tag_key = get_tag_key(tag)
            if tag_key not in tag_dict:
                continue

            if tag_key not in passage_dict:
                passage_dict[tag_key] = []
            row_item_lst = passage_dict[tag_key]
            row_item_lst.append(item)
    return passage_dict 
 
def get_tag_key(tag):
    table_id = tag['table_id']
    row = tag['row']
    key = '%s-%d' % (table_id, row)
    return key 

def get_top_tags():
    tag_dict = {}
    with open(open_qa_result_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            top_tags = item['passage_tags'][:M]
            for tag in top_tags:
                key = get_tag_key(tag)
                if key not in tag_dict:
                    tag_dict[key] = tag
    return tag_dict

def get_qas_data():
    ret_dict = {}
    data_file = '/home/cc/data/fetaqa/interactions/dev_qas.jsonl'
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            qid = item['qid']
            ret_dict[qid] = item
    return ret_dict

def read_passages(tag_dict):
    work_pool = ProcessPool(30)
    data_file_pattern = './table2txt/dataset/fetaqa/template_graph/passage_parts/graph_passages.json_part_*'
    file_lst = glob.glob(data_file_pattern)
    arg_info_lst = []
    for data_file in file_lst:
        arg_info = [data_file, tag_dict]
        arg_info_lst.append(arg_info)

    all_passage_dict = {}
    for passage_dict in tqdm(work_pool.imap_unordered(read_passage_file, arg_info_lst), total=len(arg_info_lst)):
    #for arg_info in arg_info_lst:
    #    passage_dict = read_passage_file(arg_info)
        for key in passage_dict:
            if key not in all_passage_dict:
                all_passage_dict[key] = []
            item_lst = all_passage_dict[key]
            all_passage_dict[key] = item_lst + passage_dict[key] 
    return all_passage_dict

def get_open_qa():
    open_qa = OpenQA(
        ir_host='127.0.0.1',
        ir_port=9200,
        ir_index='fetaqa_template_graph',
        model_dir='/home/cc/code/fabric_qa/model',
        cuda=0)
    return open_qa

def get_top_row_passages(open_qa, qid, question, passage_lst):
    ret_num = 20
    if len(passage_lst) <= ret_num:
        return passage_lst 

    p_id_lst = [a for a, _ in enumerate(passage_lst)]
    batch_data = [{'qid':qid, 'question':question, 'p_id_lst':p_id_lst, 'passages':passage_lst }]
    batch_examples = data_to_examples(batch_data) 
    with torch.no_grad(): 
        batch_score_lst = open_qa.answer_ranker(batch_data, batch_examples)
    scores = batch_score_lst[0].cpu().numpy()
    top_n = min(len(scores), ret_num)
    top_idxes = np.argpartition(-scores, range(top_n))[:top_n]
    top_passages = [passage_lst[a] for a in top_idxes]
    return top_passages 
 
def main():
    out_data_file = './dataset/fetaqa/template_graph/dev/fusion_in_decoder_data.jsonl'
    f_o = open(out_data_file, 'w')
    tag_dict = get_top_tags()
    passage_dict = read_passages(tag_dict)
    
    qas_data_dict = get_qas_data()
    open_qa = get_open_qa()
     
    with open(open_qa_result_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            top_tags = item['passage_tags'][:M]
            key_set = set()
            for tag in top_tags:
                key = get_tag_key(tag)
                key_set.add(key)
            tag_keys = list(key_set)
           
            out_item = {}
            out_item['id'] = item['qid']
            out_item['question'] = item['question']
            qas_item = qas_data_dict[item['qid']]
            out_item['table_id_lst'] = qas_item['table_id_lst']
            out_item['answers'] = qas_item['answers']
            out_passage_lst = []
            for key in tag_keys:
                row_passage_lst = passage_dict[key]
                text_lst = [a['passage'] for a in row_passage_lst]
                top_row_passages = get_top_row_passages(open_qa, item['qid'], item['question'], text_lst)
                out_passage = ' . '.join(top_row_passages)
                out_passage_info = {
                    'title': '',
                    'text': out_passage,
                    'tag': tag_dict[key]
                }
                out_passage_lst.append(out_passage_info)

            out_item['ctxs'] = out_passage_lst
            f_o.write(json.dumps(out_item) + '\n')
    
    f_o.close()

if __name__ == '__main__':
    main()
