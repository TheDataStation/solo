import json
from tqdm import tqdm
import os

def update_data(input_file, output_file):
    assert (not os.path.isfile(output_file)), 'output file %s already exists, nothing done.' % output_file
    
    print('loading %s' % input_file)
    
    with open(input_file) as f:
        data = json.load(f)

    with open(output_file, 'w') as f_o:
        del_key_lst = ['score', 'title_score', 'psg_id']
        for example in tqdm(data):
            num_pos = len(example['positive_ctxs'])
            num_neg = len(example['negative_ctxs']) + len(example['hard_negative_ctxs'])
            if num_pos == 0 or num_neg == 0:
                continue
            
            updated_example = {}
            updated_example['question'] = example['question']
            updated_example['answers'] = example['answers']
            updated_example['target'] = ''

            pos_ctxs = example['positive_ctxs']
            neg_ctxs = example['negative_ctxs']
            hard_neg_ctxs = example['hard_negative_ctxs']
            
            ctxs = pos_ctxs + hard_neg_ctxs + neg_ctxs
            
            pos_idxes = list(range(len(pos_ctxs)))
            hard_neg_idxes = [len(pos_ctxs) + a for a in range(len(hard_neg_ctxs))]
            neg_idxes = [len(pos_ctxs) + len(hard_neg_ctxs) + a for a in range(len(neg_ctxs))]

            for ctx_info in ctxs:
                for key in del_key_lst:
                    if key in ctx_info:
                        del ctx_info[key]

            updated_example['ctxs'] = ctxs
            updated_example['pos_idxes'] = pos_idxes
            updated_example['hard_neg_idxes'] = hard_neg_idxes
            updated_example['neg_idxes'] = neg_idxes        
           
            f_o.write(json.dumps(updated_example) + '\n')
     
    print('output is written to %s' % output_file)

def main():
    update_data('../data/trivia_qa/trivia-train.json',
                '../data/trivia_qa/train.jsonl'
    )

    update_data('../data/trivia_qa/trivia-dev.json',
                '../data/trivia_qa/dev.jsonl'
    )

if __name__ == '__main__':
    main()

