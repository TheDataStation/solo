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
    
    dev_table_file = f'./dataset/{args.dataset}/{args.sql_expr}/dev_table_lst.json'
    with open(dev_table_file) as f_dev_table:
        dev_table_lst = json.load(f_dev_table)
    dev_table_set = set(dev_table_lst)

    out_train_file = '%s/fusion_retrieved_%s.jsonl' % (data_dir, 'train')
    f_o_train = open(out_train_file, 'w')

    out_dev_file = '%s/fusion_retrieved_%s.jsonl' % (data_dir, 'dev')  
    f_o_dev = open(out_dev_file, 'w')

    out_train_data = []
    out_dev_data = []
    print('loading retrieved data')
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            gold_table_lst = item['table_id_lst']
            table_id = gold_table_lst[0]
            if table_id not in dev_table_set:
                passage_info_lst = item['ctxs']
                label_lst = [int (a['tag']['table_id'] in gold_table_lst) for a in passage_info_lst] 
                if (max(label_lst) < 1) or (min(label_lst) > 0):
                    continue
                out_train_data.append(line)
            else:
                out_dev_data.append(line)

    out_dev_data_sample = random.sample(out_dev_data, 2000)
    
    for train_line in out_train_data:
        f_o_train.write(train_line)
    f_o_train.close()
    for dev_line in out_dev_data_sample:
        f_o_dev.write(dev_line)
    f_o_dev.close()

if __name__ == '__main__':
    main()

