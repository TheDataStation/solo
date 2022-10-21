import json
from tqdm import tqdm
import glob
import pickle
import csv
import numpy as np

def read_tables(data_file):
    table_id_lst = []
    with open(data_file) as f:
        for line in f:
            table_id = line.strip()
            table_id_lst.append(table_id)
    table_set = set(table_id_lst)
    return table_set

def read_passage_info():
    data_file = './data/graph_passages.json'
    passage_info_dict = {}
    with open(data_file) as f:
        passage_id = 0
        for line in tqdm(f):
            item = json.loads(line)
            passage_id += 1
            tag = item['tag']
            table_id = tag['table_id']
            #if table_id in table_set:
            passage_info_dict[passage_id] = item 

    return passage_info_dict

def gen_passage_embs(passage_info_dict):
    emb_file_pattern = './data/nq_tables_passage_embeddings/nq_tables_passage_embeddings_part_*_00'
    emd_files = glob.glob(emb_file_pattern)
    emd_files = sorted(emd_files)
    
    emb_info_dict = {}
    for emb_file in emd_files:
        with open(emb_file, 'rb') as f_emb:
            p_ids, p_embeddings = pickle.load(f_emb)
            for idx, p_id_str in tqdm(enumerate(p_ids), total=len(p_ids)):
                p_id = int(p_id_str)
                if p_id not in passage_info_dict:
                    continue
                p_emb = p_embeddings[idx]
                meta_info = passage_info_dict[p_id]
                table_id = meta_info['tag']['table_id']
                if table_id not in emb_info_dict:
                    emb_info_dict[table_id] = {}
                row_dict = emb_info_dict[table_id]
                row = int(meta_info['tag']['row'])
                if row not in row_dict:
                    row_dict[row] = []
                cell_items = row_dict[row]
                emb_item = {
                    'p_id':p_id,
                    'emb':p_emb,
                    'tag':meta_info['tag'],
                    'passage':meta_info['passage']
                }
                cell_items.append(emb_item)

    new_p_id_lst, result_ems, updated_meta_dict = get_all_id_embs(emb_info_dict)
    out_emb_file = './data/nq_tables_passage_embeddings/nq_tables_passage_embeddings_all' 
    out_emb_data = (new_p_id_lst, result_ems)
    with open(out_emb_file, 'wb') as f_o:
        pickle.dump(out_emb_data, f_o, protocol=4)

    save_meta(updated_meta_dict)
    
    gen_passages(updated_meta_dict)

def get_all_id_embs(emb_info_dict):
    updated_meta_dict = {}
    p_id_lst = []
    emb_lst = []
    new_p_id = 0
    for table_id in tqdm(emb_info_dict):
        row_dict = emb_info_dict[table_id]
        for row in row_dict:
            cell_items = row_dict[row]
            for emb_item in cell_items:
                p_id_lst.append(new_p_id)
                meta_item = {
                    'tag':emb_item['tag'],
                    'passage':emb_item['passage']
                }
                updated_meta_dict[new_p_id] = meta_item
                emb_lst.append(emb_item['emb'])
                new_p_id += 1

    result_ems = np.vstack(emb_lst)
    return p_id_lst, result_ems, updated_meta_dict

def gen_passages(meta_dict):
    out_passage_file = './data/nq_tables_passages/passage_all.tsv'
    with open(out_passage_file, 'w') as f_o_p:
        csv_writer = csv.writer(f_o_p, delimiter='\t')
        csv_writer.writerow(['id', 'text', 'title'])
        for passage_id in tqdm(meta_dict):
            csv_writer.writerow([passage_id, meta_dict[passage_id]['passage'], ''])

def save_meta(meta_dict):
    out_file = './data/passage_meta_all.jsonl'
    with open(out_file, 'w') as f_o:
        for p_id in tqdm(meta_dict):
            item = meta_dict[p_id]
            out_item = {
                'p_id':p_id,
                'tag':item['tag']
            }
            f_o.write(json.dumps(out_item) + '\n')
 
def main():
    #table_id_file = '/home/cc/data/nq_tables/tables/table_id_small.jsonl'
    #table_set = read_tables(table_id_file)
    table_set = None
    passage_info_dict = read_passage_info()
    gen_passage_embs(passage_info_dict)

if __name__ == '__main__':
    main()
