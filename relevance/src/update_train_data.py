import json
from tqdm import tqdm
import os


import copy

def get_retr_ctxs(retr_item):
    ctx_lst = retr_item['ctxs']
    pos_ctxs = []
    neg_ctxs = []
    for ctx in ctx_lst:
        ctx_info = {
            'title':'',
            'text':ctx['text'],
            'score':.0,
            'title_score':.0,
            'psg_id':0,
        }
        if ctx['hasanswer']:
            pos_ctxs.append(ctx_info)
        else:
            neg_ctxs.append(ctx_info) 

    return pos_ctxs, neg_ctxs

def update_data(is_train, input_file, retr_file, output_file):
    assert (not os.path.isfile(output_file)), 'output file %s already exists, nothing done.' % output_file
    
    print('loading %s' % input_file)
    with open(input_file) as f:
        data = json.load(f)
    retr_data = None
    #retr_file = None
    if retr_file is not None:
        print('loading %s' % retr_file)
        with open(retr_file) as f_r:
            retr_data = json.load(f_r)
    
    MAX_CTX_NUM = 100
    with open(output_file, 'w') as f_o:
        del_key_lst = ['score', 'title_score', 'psg_id']
        for offset, example in tqdm(enumerate(data), total=len(data)):
            
            #example_copy = copy.deepcopy(example)
            
            retr_item = None
            retr_pos_ctxs = None
            retr_neg_ctxs = None
            if retr_data is not None:
                retr_item = retr_data[offset]
                assert example['question'] == retr_item['question']
                retr_pos_ctxs, retr_neg_ctxs = get_retr_ctxs(retr_item) 
                if len(example['positive_ctxs']) == 0:
                    example['positive_ctxs'] += retr_pos_ctxs
                example['hard_negative_ctxs'] = retr_neg_ctxs + example['hard_negative_ctxs']
            
            example['hard_negative_ctxs'] = example['hard_negative_ctxs'] + example['negative_ctxs']
            example['negative_ctxs'] = []
            
            num_1 = len(example['positive_ctxs'])           
            num_2 = len(example['hard_negative_ctxs'])
            if num_1 == 0 or num_2 == 0:
                continue

            if (num_1 + num_2) > MAX_CTX_NUM:
                updated_num_1 = int(MAX_CTX_NUM * (num_1 / (num_1 + num_2)))
                example['positive_ctxs'] = example['positive_ctxs'][:updated_num_1]
                updated_num_2 = MAX_CTX_NUM - len(example['positive_ctxs']) 
                example['hard_negative_ctxs'] = example['hard_negative_ctxs'][:updated_num_2]

            num_pos = len(example['positive_ctxs'])
            num_neg = len(example['hard_negative_ctxs'])
          
            if num_pos + num_neg != MAX_CTX_NUM:
                raise ValueError('err')
            
            updated_example = {}
            updated_example['question'] = example['question']
            updated_example['answers'] = example['answers']
            updated_example['target'] = ''

            pos_ctxs = example['positive_ctxs']
            hard_neg_ctxs = example['hard_negative_ctxs']
            
            ctxs = pos_ctxs + hard_neg_ctxs 
            
            pos_idxes = list(range(len(pos_ctxs)))
            hard_neg_idxes = [len(pos_ctxs) + a for a in range(len(hard_neg_ctxs))]
            neg_idxes = [] 

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
    update_data(True,
                '../data/trivia_qa/trivia-train.json',
                '../data/TQA/train_retr_student_float_16.json',
                '../data/trivia_qa/train_merged.jsonl'
    )

    update_data(False,
                '../data/trivia_qa/trivia-dev.json',
                None,
                '../data/trivia_qa/dev_merged.jsonl'
    )

if __name__ == '__main__':
    main()

