import json
import argparse
from tqdm import tqdm
import os
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    repre_lst = ['rel_graph', 'graph_text', 'table_token_slide']
    dataset_dir = '/home/cc/code/open_table_discovery/table2question/dataset/%s/sql_data/train_0' % args.dataset 
    
    data_map_0, q_set_0 = get_repre_data(dataset_dir, repre_lst[0])
    data_map_1, q_set_1 = get_repre_data(dataset_dir, repre_lst[1])
    data_map_2, q_set_2 = get_repre_data(dataset_dir, repre_lst[2])

    q_set_share = q_set_0.intersection(q_set_1, q_set_2)
    q_lst = random.sample(list(q_set_share), 1000)
    
    output_repre_data(dataset_dir, repre_lst[0], data_map_0, q_lst)
    output_repre_data(dataset_dir, repre_lst[1], data_map_1, q_lst)
    output_repre_data(dataset_dir, repre_lst[2], data_map_2, q_lst)
    
def get_repre_data(dataset_dir, repre):
    data_map = {}
    q_set = set()
    data_file = os.path.join(dataset_dir, repre, 'fusion_retrieved_tagged.jsonl')
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            qid = item['id']
            q_set.add(qid)
            data_map[qid] = line
     
    return data_map, q_set 

def output_repre_data(dataset_dir, repre, data_map, q_lst):
    out_dir = os.path.join(dataset_dir, repre, 'data_parts')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, 'repre_%s.jsonl' % repre)
    if os.path.isfile(out_file):
        raise ValueError('%s already exists' % out_file)
    with open(out_file, 'w') as f_o:
        for qid in tqdm(q_lst):
            f_o.write(data_map[qid]) 
    

if __name__ == '__main__':
    main()
