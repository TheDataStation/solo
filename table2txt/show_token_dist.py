import json
from tqdm import tqdm
import transformers
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--file_name', type=str)
    args = parser.parse_args()
    return args

def main():
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    passage_file = './dataset/%s/%s/%s' % (args.dataset, args.experiment, args.file_name)
    passage_lst = []
    with open(passage_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            passage = item['passage']
            passage_lst.append(passage)
    
    bsz = 1000
    seq_len_lst = []
    for idx in tqdm(range(0, len(passage_lst), bsz)):
        batch_passages = passage_lst[idx:(idx+bsz)]
        batch_encoded = tokenizer.batch_encode_plus(batch_passages)
        batch_id_data = batch_encoded['input_ids']
        batch_seq_len_lst = [len(a) for a in batch_id_data]
        seq_len_lst.extend(batch_seq_len_lst)

    min_len = np.min(seq_len_lst)
    max_len = np.max(seq_len_lst)
    mean_len = np.mean(seq_len_lst)
    
    print(min_len, max_len, mean_len) 
    for a in [50, 60, 70, 80, 90, 95, 96, 97, 98, 99]: 
        len_value = np.percentile(seq_len_lst, a)
        print('%d percentile : %d' %(a, len_value))

if __name__ == '__main__':
    main()


