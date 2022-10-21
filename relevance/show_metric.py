import json
import numpy as np
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--top_n', type=int, required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    correct_lst = []
    M = args.top_n
    with open(args.data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            gold_tables = item['table_id_lst']
            ctx_lst = item['ctxs']
            labels = [int(a['tag']['table_id'] in gold_tables) for a in ctx_lst]
            correct = max(labels[:M])
            correct_lst.append(correct)

    result = np.mean(correct_lst) * 100

    print('p@%s=%.2f' % (M, result))

if __name__ == '__main__':
    main()
