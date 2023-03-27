import faiss
from faiss.contrib.ondisk import merge_ondisk
from tqdm import tqdm
import pickle
import random
import os
import numpy as np
import uuid
import json
import argparse
import glob
import math
import time

class OndiskIndexer:
    def __init__(self, index_file, passage_file):
        self.index = faiss.read_index(index_file)
        self.passage_dict = self.load_passages(passage_file)
   
    def load_passages(self, passage_file):
        passage_dict = {} 
        with open(passage_file) as f_p:
            for line in tqdm(f_p):
                item = json.loads(line)
                p_id = item['p_id']
                passage_dict[int(p_id)] = item 
        return passage_dict
    
    def search(self, query, top_n=100, n_probe=128, min_tables=5, max_retr=1000):
        result = []
        N = len(query)
        for idx in range(0, N):
            pos = idx + 1
            batch_query = query[idx:pos]
            satified=False
            num_retr = top_n
            item_passage_lst = []
            while (not satified):
                item_passage_lst = self.one_query_search(batch_query, top_n=num_retr, n_probe=n_probe)
                table_lst = [a['tag']['table_id'] for a in item_passage_lst]
                table_set = set(table_lst)
                if (num_retr < max_retr) and  (len(table_set) < min_tables) :
                    num_retr = max_retr
                else:
                    satified = True 

            result.append(item_passage_lst)
        return result
         
    def one_query_search(self, query, top_n=100, n_probe=128):
        assert(len(query) == 1)
        self.index.nprobe = n_probe
        batch_dists, batch_p_ids = self.index.search(query, top_n)
        item_result = []

        p_id_lst = batch_p_ids[0]
        p_dist_lst = batch_dists[0]
        for idx, p_id in enumerate(p_id_lst):
            if p_id == -1: #faiss may return -1 if there are not enough elements in an nlist
                continue
            passage_info = self.passage_dict[int(p_id)]
            out_item = {
                'p_id':p_id,
                'passage':passage_info['passage'],
                'score':p_dist_lst[idx],
                'tag':passage_info['tag']
            }
            item_result.append(out_item)
        return item_result

# end of class OndiskIndexer 

def index_data(index_file, data_file, index_out_dir, block_size=5000000):
    print('start indexing passages')
    bno = 0
    block_fnames = []
    emb_file_lst = glob.glob(data_file)
    emb_file_lst.sort()
    for emb_file in emb_file_lst: 
        print('loading file [%s]' % emb_file)
        p_ids, p_embs = load_emb(emb_file)
        N = len(p_ids)
        print('creating block indexes')
        for idx in range(0, N, block_size):
            index = faiss.read_index(index_file)
            index_ivf = faiss.extract_index_ivf(index)
            index_ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
            pos = idx + block_size
            block_p_ids = np.int64(np.array(p_ids[idx:pos]))
            block_p_embs = p_embs[idx:pos]
            index.add_with_ids(block_p_embs, block_p_ids)
            block_file_name = os.path.join(index_out_dir, 'block_%d.index' % bno)
            faiss.write_index(index, block_file_name)
            block_fnames.append(block_file_name)
            bno += 1
   
    merged_file_name = os.path.join(index_out_dir, 'merged_index.ivfdata')
    print('merging block indexes')
    index = faiss.read_index(index_file)
    merge_ondisk(index, block_fnames, merged_file_name)
     
    out_index_file = os.path.join(index_out_dir, 'populated.index')
    print('writing to [%s]' % out_index_file)
    faiss.write_index(index, out_index_file)
   
    #remove the empty trained index and the block files
    os.remove(index_file)
    for block_file_name in block_fnames:
        os.remove(block_file_name)

def get_index_options(num_vecs):
    unit = 1e6 
    if num_vecs < unit:
        num_clusters = int(num_vecs / 50)
        num_clusters = min(num_clusters, num_vecs)
        num_train = 50 * num_clusters
        num_train = min(num_train, num_vecs)
        factory_string = 'IVF%s,Flat' % num_clusters
    else:
        num_clusters = 8192
        factory_string = 'IVF%s,PQ64' % num_clusters
        num_train = num_clusters * 1024
        num_train = min(num_train, num_vecs)
    return (factory_string, num_train)
    
    '''
    unit = 1e6 
    if num_vecs < unit:
        num_clusters = int(16 * math.sqrt(num_vecs))
        num_clusters = min(num_clusters, num_vecs)
        num_train = 256 * num_clusters
        num_train = min(num_train, num_vecs)
        factory_string = 'IVF%s,Flat' % num_clusters

    elif (num_vecs >= unit) and (num_vecs < 10 * unit):
       num_clusters = 65536
       num_train = min(60 * num_clusters, num_vecs)
       factory_string = 'IVF%s_HNSW32,Flat' % num_clusters
    
    elif (num_vecs >= 10 * unit) and (num_vecs < 100 * unit):
       num_clusters = 262144
       num_train = min(60 * num_clusters, num_vecs)
       factory_string = 'IVF%s_HNSW32,Flat' % num_clusters
        
    elif (num_vecs >= 100 * unit) and (num_vecs < 1000 * unit):
       num_clusters = 1048576
       num_train = min(60 * num_clusters, num_vecs)
       factory_string = 'IVF%s_HNSW32,Flat' % num_clusters
    
    else:
        raise ValueError('Not supported right now') 

    print('factory_string=%s, num_train=%d' % (factory_string, num_train))     
    return factory_string, num_train
    '''

def get_num_vecs(emb_file_lst):
    print('collecting the number of vectors')
    num_vecs = 0
    for emb_file in emb_file_lst:
        _, p_embs = load_emb(emb_file)
        num_part_vecs = len(p_embs)
        num_vecs += num_part_vecs
    return num_vecs

def load_emb(emb_file):
    p_id_lst = []
    p_emb_lst = []
    with open(emb_file, 'rb') as f:
        while True:
            try:
                batch_p_id, batch_p_emb = pickle.load(f)
                p_id_lst.append(batch_p_id)
                p_emb_lst.append(batch_p_emb)
            except EOFError:
                break
    all_p_id = [p_id for batch in p_id_lst for p_id in batch]
    all_p_emb = np.concatenate(p_emb_lst, axis=0)
    return all_p_id, all_p_emb 

# create an empty index and train it
def create_train(data_file, index_file):
    if os.path.exists(index_file):
        print('index file [%s] already exists' % index_file)
        return 
    #print('loading data')
    emb_file_lst = glob.glob(data_file)
    emb_file_lst.sort()

    num_vecs = get_num_vecs(emb_file_lst)
    #print('num_vecs=%d' % num_vecs)
    factory_string, num_train = get_index_options(num_vecs)
    print('factory_string=%s, num_train=%d' % (factory_string, num_train))     

    train_emb_lst = []
    for emb_file in emb_file_lst:
        _, p_embs = load_emb(emb_file)
        N = p_embs.shape[0]
        rows = list(np.arange(0, N))

        num_train_in_file = int(num_train * (N / num_vecs))
        num_sample_train = min(len(rows), num_train_in_file)
        
        #print('num_sample_train=', num_sample_train)
        train_rows = random.sample(rows, num_sample_train)
        train_emb = p_embs[train_rows] 
        train_emb_lst.append(train_emb)

    train_all_embs = np.vstack(train_emb_lst)
    train_all_embs = np.float32(train_all_embs)
   
    #print('number of traing vectors = %d' % len(train_all_embs))
    
    D = train_all_embs.shape[1]
    index = faiss.index_factory(D, factory_string, faiss.METRIC_INNER_PRODUCT)
    index_ivf = faiss.extract_index_ivf(index)
    cls_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
    index_ivf.clustering_index = cls_index  
    print('training index')
    t1 = time.time()
    index.train(train_all_embs)
    t2 = time.time()
    print('train time = %d' % (t2 - t1))
    print('wrting trained index to [%s]' % index_file)
    faiss.write_index(index, index_file) 

def main(args):
    data_dir = os.path.join(args.work_dir, 'index')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    index_out_dir = os.path.join(data_dir, 'on_disk_index_%s_%s' % (args.dataset, args.experiment))
    if os.path.exists(index_out_dir):
        msg_text = 'Index directory (%s) already exists' % index_out_dir
        msg_info = {
            'state':False,
            'msg':msg_text
        }
        return msg_info
         
    os.mkdir(index_out_dir)
    dataset_dir = os.path.join(args.work_dir, 'open_table_discovery/table2txt/dataset/')
    exptr_dir = os.path.join(dataset_dir, args.dataset, args.experiment, 'emb')
    data_file = os.path.join(exptr_dir, args.emb_file)
    trained_index_file = os.path.join(index_out_dir, 'trained.index')
    create_train(data_file, trained_index_file)
    t1 = time.time()
    index_data(trained_index_file, data_file, index_out_dir)
    t2 = time.time()
    print('Indexing time = %d' % (t2 - t1))

    msg_info = {
        'state':True,
        'index_dir':index_out_dir
    }
    return msg_info

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--emb_file', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    msg_info = main(args)
    if not mag_info['state']:
        print(msg_info['msg']) 
    
