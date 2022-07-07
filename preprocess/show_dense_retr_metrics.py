import json
import argparse
from tqdm import tqdm
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--experiment', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    retr_file = './table2txt/dataset/%s/%s/fusion_retrieved_%s.json' % (args.dataset, args.experiment, args.mode)
    with open(retr_file) as f:
        retr_data = json.load(f)
    metric_lst = []
    for item in tqdm(retr_data):
        gold_table_lst = item['table_id_lst']
        top_table_id = item['ctxs'][0]['tag']['table_id']
        correct = int(top_table_id in gold_table_lst)
        metric_lst.append(correct)
    
    mean = np.mean(metric_lst) * 100

    print('p@1=%.2f' % mean)


if __name__ == '__main__':
    main()
