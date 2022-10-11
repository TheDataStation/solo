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
    dataset_dir = '/home/cc/code/open_table_discovery/table2question/dataset/%s/sql_data/train_0' % args.dataset 
    
    data_map_0, q_set_0 = get_index_data(dataset_dir, bm25=False)
    data_map_1, q_set_1 = get_index_data(dataset_dir, bm25=True)

    q_set_share = q_set_0.intersection(q_set_1)
    q_lst = random.sample(list(q_set_share), 1000)
    
    output_shared_data(dataset_dir, 'rel_graph_dense', data_map_0, q_lst)
    output_shared_data(dataset_dir, 'rel_graph_bm25', data_map_1, q_lst)
    
def get_index_data(dataset_dir, bm25=False):
    data_map = {}
    q_set = set()
    if bm25:
        file_name = 'fusion_retrieved_bm25_tagged.jsonl'
    else:
        file_name = 'fusion_retrieved_tagged.jsonl'
    data_file = os.path.join(dataset_dir, 'rel_graph', file_name)
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            qid = item['id']
            q_set.add(qid)
            data_map[qid] = line
     
    return data_map, q_set 

def output_shared_data(dataset_dir, index_tag, data_map, q_lst):
    out_dir = os.path.join(dataset_dir, 'rel_graph', 'data_parts')
    out_file = os.path.join(out_dir, '%s.jsonl' % index_tag)
    if os.path.isfile(out_file):
        raise ValueError('%s already exists' % out_file)
    with open(out_file, 'w') as f_o:
        for qid in tqdm(q_lst):
            f_o.write(data_map[qid]) 
    

if __name__ == '__main__':
    main()
