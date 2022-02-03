import json
import os
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--expr', type=str)
    args = parser.parse_args()
    return args

def read_meta(meta_file):
    meta_data = []
    with open(meta_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            meta_data.append(item)
    return meta_data

def read_questions(q_file):
    questions = []
    with open(q_file) as f:
        for line in tqdm(f):
            questions.append(line.strip())
    return questions

def main():
    args = get_args()
    data_dir = os.path.join('dataset', args.dataset, args.expr) 
    meta_file = os.path.join(data_dir, 'meta.txt')
    q_file = os.path.join(data_dir, 'questions.txt')
    out_query_file = os.path.join(data_dir, 'fusion_query.jsonl')
    if os.path.exists(out_query_file):
        print('[%s] already exists' % out_query_file)
        return 
     
    f_o_query = open(out_query_file, 'w') 
    meta_data = read_meta(meta_file)
    q_data = read_questions(q_file)
    
    for idx, question in tqdm(enumerate(q_data)):
        meta_item = meta_data[idx]
        qid = 'q_%d' % idx
        table_id_lst = [meta_item['table_id']]
        answers = ['N/A']       
        passage_info = {
            'title':'',
            'text':'This is a example passage',
        }
        out_item = {
            'id':qid,
            'question':question,
            'table_id_lst':table_id_lst,
            'answers':answers,
            'ctxs':[passage_info]
        }
        
        f_o_query.write(json.dumps(out_item) + '\n') 
    
    f_o_query.close()

if __name__ == '__main__':
    main()

