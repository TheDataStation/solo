import json
from tqdm import tqdm
import random
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--sql_expr', type=str)
    parser.add_argument('--table_expr', type=str)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data_dir = './dataset/%s/%s/%s' % (args.dataset, args.sql_expr, args.table_expr)
    data_file = '%s/retrieved_data.jsonl' % data_dir
    print('loading retrieved data')
    data_table_dict = {}
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            gold_table_lst = item['table_id_lst']
            table_id = gold_table_lst[0]
            passage_info_lst = item['ctxs']
            label_lst = [int (a['tag']['table_id'] in gold_table_lst) for a in passage_info_lst] 
            if (max(label_lst) < 1) or (min(label_lst) > 0):
                continue
             
            if table_id not in data_table_dict:
                data_table_dict[table_id] = []
            table_item_lst = data_table_dict[table_id]
            table_item_lst.append(item)
    
    all_table_id_lst = list(data_table_dict.keys())
    num_dev = 1000
    num_train = len(all_table_id_lst) - num_dev
    train_table_id_lst = random.sample(all_table_id_lst, num_train)
    train_table_id_set = set(train_table_id_lst)
    
    gen_data(data_dir, data_table_dict, train_table_id_set)

def gen_data(data_dir, data_table_dict, train_table_id_set): 
    out_train_file = '%s/fusion_retrieved_%s.jsonl' % (data_dir, 'train')  
    f_o_train = open(out_train_file, 'w')
    out_dev_file = '%s/fusion_retrieved_%s.jsonl' % (data_dir, 'dev')  
    f_o_dev = open(out_dev_file, 'w')
    
    for table_id in tqdm(data_table_dict):
        if table_id in train_table_id_set:
            f_o = f_o_train
        else:
            f_o = f_o_dev
       
        table_item_lst = data_table_dict[table_id]
        for item in table_item_lst:
            f_o.write(json.dumps(item) + '\n') 
    
    f_o_train.close()
    f_o_dev.close()

if __name__ == '__main__':
    main()

