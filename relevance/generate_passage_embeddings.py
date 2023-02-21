# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import argparse
import csv
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import transformers

import src.model
import src.data
import src.util
import src.slurm
import sys
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)

def embed_passages(opt, passages, model, tokenizer, f_o):
    batch_size = opt.per_gpu_batch_size * opt.world_size
    collator = src.data.TextCollator(tokenizer, model.config.passage_maxlength)
    dataset = src.data.TextDataset(passages, title_prefix='title:', passage_prefix='context:')
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=0, collate_fn=collator) # no multiprocessing
    total = 0
    allids, allembeddings = [], []
    num_batch = len(dataloader)
    with torch.no_grad():
        for k, (ids, text_ids, text_mask) in tqdm(enumerate(dataloader), total=num_batch, disable=(not opt.show_progress)):
            embeddings = model.embed_text(
                text_ids=text_ids.cuda(), 
                text_mask=text_mask.cuda(), 
                apply_mask=model.config.apply_passage_mask,
                extract_cls=model.config.extract_cls,
            )
            embeddings = embeddings.cpu().numpy()
            total += len(ids)

            pickle.dump((ids, embeddings), f_o, protocol=4)
            if k % 100 == 0:
                logger.info('Encoded passages %d', total)

def main(opt, is_main):
    src.slurm.init_distributed_mode(opt)
    args = opt
    logger = src.util.init_logger(is_main=is_main)
    output_path = Path(args.output_path)
    save_file = output_path.parent / (output_path.name + f'_{args.shard_id:02d}')
    if os.path.exists(save_file):
        msg_txt = '(%s) already exists' % save_file
        logger.info(msg_txt)
        msg_info = {
            'state':False,
            'msg':msg_txt
        }
        return

    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    model_class = src.model.Retriever
    #model, _, _, _, _, _ = src.util.load(model_class, opt.model_path, opt)
    model = model_class.from_pretrained(opt.model_path)
    
    model.eval()
    model = model.to(opt.device)
    if not opt.no_fp16:
        model = model.half()

    passages = src.util.load_passages(args.passages)

    output_path.parent.mkdir(parents=True, exist_ok=True) 
    with open(save_file, mode='wb') as f_o:
        embed_passages(opt, passages, model, tokenizer, f_o)
        logger.info(f'Saving {len(passages)} passage embeddings to {save_file}')

    msg_info = {
        'state':True,
        'out_file':str(save_file)
    }
    return msg_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_progress', type=int, default=True)
    parser.add_argument('--passages', type=str, default=None, help='Path to passages (.jsonl file)')
    parser.add_argument('--output_path', type=str, default='wikipedia_embeddings/passages', help='prefix path to save embeddings')
    parser.add_argument('--shard_id', type=int, default=0, help="Id of the current shard")
    parser.add_argument('--num_shards', type=int, default=1, help="Total number of shards")
    parser.add_argument('--per_gpu_batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    parser.add_argument('--passage_maxlength', type=int, default=200, help="Maximum number of tokens in a passage")
    parser.add_argument('--model_path', type=str, help="path to directory containing model weights and config file")
    parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    args = parser.parse_args()

    main(args,  is_main=True)


