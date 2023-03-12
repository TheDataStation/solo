# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np

from tqdm import tqdm

Question_Prefix = 'question:'

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix=Question_Prefix,
                 title_prefix='title:',
                 passage_prefix='context:',
                 sort_by_score=False,
                 ignore_context=False
                 ):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.ignore_context = ignore_context
        if sort_by_score:
            self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if ('ctxs' in example) and (not self.ignore_context):
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            if self.n_context is not None:
                contexts = example['ctxs'][:self.n_context]
            else:
                contexts = example['ctxs']
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            tags = [c.get('tag', None) for c in contexts]
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores, tags = None, None, None

        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores,
            'tags': tags,
            'pos_idxes':example.get('pos_idxes', None),
            'neg_idxes':example.get('hard_neg_idxes', None),
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

def get_max_token_size(batch_text_passages, tokenizer, max_length):
    max_size_lst = []
    for k, text_passages in enumerate(batch_text_passages):
        try_encoded = tokenizer.batch_encode_plus(text_passages)['input_ids']
        max_token_size = max([len(a) for a in try_encoded])
        max_size_lst.append(max_token_size)
    return min(max_length, max(max_size_lst))

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    max_token_size = get_max_token_size(batch_text_passages, tokenizer, max_length)
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_token_size,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)
        passage_tags = [a['tags'] for a in batch]
        return (index, target_ids, target_mask, passage_ids, passage_masks, passage_tags)

def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    examples = []
    with open(data_path, 'r') as f:
        for k, line in tqdm(enumerate(f)):
            example = json.loads(line)
            if not 'id' in example:
                example['id'] = k
            for c in example['ctxs']:
                if not 'score' in c:
                    c['score'] = 1.0 / (k + 1)
            examples.append(example)

    return examples


class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40,
                 all_passages=False, sample_pos_ctx=False, sample_neg_ctx=False, num_neg_ctxs=None):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

        self.all_passages = all_passages       
        self.sample_pos_ctx = sample_pos_ctx
        self.sample_neg_ctx = sample_neg_ctx
        self.num_neg_ctxs = num_neg_ctxs

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        passages = [] #[ex['passages'] for ex in batch]
        meta_lst = []
        if self.all_passages:
            passages = [ex['passages'] for ex in batch]
        else:
            for ex in batch:
                item_passages = ex['passages']
                item_pos_idxes = ex['pos_idxes']
                item_neg_idxes = ex['neg_idxes']
                if self.sample_pos_ctx:
                    pos_ctx_idx = random.sample(item_pos_idxes, 1)[0]
                else:
                    pos_ctx_idx = item_pos_idxes[0]
                
                if self.sample_neg_ctx:
                    assert self.num_neg_ctxs is not None
                    neg_ctx_idxes = random.sample(item_neg_idxes, self.num_neg_ctxs)
                else:
                    if self.num_neg_ctxs is not None:
                        neg_ctx_idxes = item_neg_idxes[:self.num_neg_ctxs]
                    else:
                        neg_ctx_idxes = item_neg_idxes[:] 
                
                sample_ctx_idxes = [pos_ctx_idx] + neg_ctx_idxes
                #The first is the postive passage and others are negative ones
                sample_passages = [item_passages[a] for a in sample_ctx_idxes] 
                passages.append(sample_passages) 
                
                meta_info = {
                    'passage_idxes':sample_ctx_idxes
                }
                meta_lst.append(meta_info)

        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, meta_lst)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
            self.passage_prefix + " " + example[1]
        return example[0], text

class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
