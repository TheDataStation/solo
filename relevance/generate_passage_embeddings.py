
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import argparse
import csv
import logging
import pickle
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import transformers
import src.model
import src.data
import src.util
import src.slurm
import sys
from tqdm import tqdm
import src.student_retriever
import queue
import threading
import glob

csv.field_size_limit(sys.maxsize)
#logger = logging.getLogger(__name__)
import time
import json

logger = None

queue_token_tensor = queue.Queue()
queue_output = queue.Queue()
queue_output_stat = queue.Queue()
opt = None
g_title_prefix='title:'
g_passage_prefix='context:'

def tok_worker(start_idx, end_idx):
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    collator = src.data.TextCollator(tokenizer, opt.passage_maxlength)
    batch_passages = []
    for idx, line in enumerate(src.util.read_passage(opt.passages)):
        if idx < start_idx:
            continue
        elif idx >= end_idx:
            break
        if len(batch_passages) == opt.per_gpu_batch_size:
            batch_data = collator(batch_passages)
            updated_data = [batch_data[0], batch_data[1].to(opt.device), batch_data[2].to(opt.device)]
            queue_token_tensor.put(updated_data)
            batch_passages = [] 
            
        passage_info = src.util.get_passage_info(line)
        annoated_passage_info = src.data.TextDataset.annoate_passage(passage_info, g_title_prefix, g_passage_prefix)
        batch_passages.append(annoated_passage_info)    
     
    if len(batch_passages) > 0:
        batch_data = collator(batch_passages)
        updated_data = [batch_data[0], batch_data[1].to(opt.device), batch_data[2].to(opt.device)]
        queue_token_tensor.put(updated_data)
        batch_passages = [] 


def start_tok_threading():
    num_rows = g_passage_count
    num_workers = 1
    part_size = num_rows // num_workers

    start_idx = 0
    for w_idx in range(num_workers):
        if w_idx < (num_workers - 1):
            end_idx = start_idx + part_size
        else:
            end_idx = num_rows
        threading.Thread(target=tok_worker, args=(start_idx, end_idx, )).start()
        start_idx = end_idx


def output_worker(part_idx):
    data = queue_output.get()
    id_lst = data[0]
    emb_lst = data[1]
    out_file = opt.output_path + "_part_" + str(part_idx)
    with open(out_file, mode="wb") as f_o:
        for offset, emb in enumerate(emb_lst):
            out_data = [id_lst[offset], emb]
            pickle.dump(out_data, f_o)

    queue_output_stat.put([part_idx, data[2], out_file])


def start_output_threading(results, part_idx):
    queue_output.put(results)
    threading.Thread(target=output_worker, args=(part_idx, )).start()

def embed_passages(model, retriever):
    start_tok_threading()

    num_passages = g_passage_count
    total = 0
    output_part_idx = 0
    output_data = [[], [], 0]
    
    while True:
        batch_data = queue_token_tensor.get() 
        #t2 = time.time()
        ids, text_ids, text_mask = batch_data
        with torch.no_grad():
            embeddings = model.embed_text(
                text_ids=text_ids, 
                text_mask=text_mask, 
                apply_mask=retriever.config.apply_passage_mask,
                extract_cls=retriever.config.extract_cls,
            )
        embeddings = embeddings.cpu().numpy()
        #t3 = time.time()
        #print('embed t3-t2 = ', t3-t2)
        output_data[0].append(ids)
        output_data[1].append(embeddings)
        output_data[2] += len(ids)
         
        total += len(ids)
        
        if total % 10000 == 0:
            logger.info('Encoded passages %d', total)
        
        if output_data[2] >= opt.output_batch_size:
            start_output_threading(output_data, output_part_idx)
            output_part_idx += 1
            output_data = [[], [], 0]
        
        if total == num_passages:
            break
    
    if output_data[2] > 0:
        start_output_threading(output_data, output_part_idx)
        output_part_idx += 1
        output_data = [[], [], 0]
    
    return output_part_idx 


def show_output_stat(num_output_parts):
    num_part = 0
    output_size = 0
    while True:
        out_stat = queue_output_stat.get()
        part_idx = out_stat[0]
        part_size = out_stat[1]
        out_file = out_stat[2]
        logger.info("Passages part %d processed %d. Written to %s", part_idx, part_size, out_file)
        num_part += 1
        output_size += out_stat[1]
        if num_part == num_output_parts:
            break
    logger.info("Total passages processed %d.", output_size)

 
def main(args, is_main):
    global opt
    opt = args
    assert opt.is_student is not None
    assert opt.output_path is not None
    src.slurm.init_distributed_mode(opt)
    assert opt.world_size == 1

    global logger
    output_dir = os.path.dirname(opt.output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    log_file = os.path.join(output_dir, 'log.txt')
    logger = src.util.init_logger(is_main, opt.is_distributed, log_file)
    logger.setLevel(logging.INFO)

    out_files = glob.glob(opt.output_path + '*') 
    if len(out_files) > 0:
        msg_txt = '(%s*) already exists' % opt.output_path
        logger.info(msg_txt)
        msg_info = {
            'state':False,
            'msg':msg_txt
        }
        return msg_info
    output_dir = os.path.dirname(opt.output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir) 

    if opt.is_student:
        model_class = src.student_retriever.StudentRetriever
    else:
        model_class = src.model.Retriever
    #model, _, _, _, _, _ = src.util.load(model_class, opt.model_path, opt)
    retriever = src.util.load_pretrained_retriever(opt.is_student, opt.model_path)
    opt.passage_maxlength = retriever.config.passage_maxlength
    if not opt.is_student:
        if retriever.model.pooler is not None:
            retriever.model.pooler = None
        model = retriever
    else:
        model = retriever.ctx_encoder
     
     
    model.eval()
    model = model.to(opt.device)
    if not opt.no_fp16:
        model = model.half()
    
    global g_passage_count
    g_passage_count = src.util.count_passages(args.passages)


    num_output_parts = embed_passages(model, retriever)
    show_output_stat(num_output_parts)

    msg_info = {
        'state':True,
        'out_file':str(opt.output_path)
    }
    return msg_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_student', type=int, default=None)
    parser.add_argument('--show_progress', type=int, default=True)
    parser.add_argument('--passages', type=str, default=None, help='Path to passages (.jsonl file)')
    parser.add_argument('--output_path', type=str, help='file prefix to store embeddings')
    parser.add_argument('--shard_id', type=int, default=0, help="Id of the current shard")
    parser.add_argument('--num_shards', type=int, default=1, help="Total number of shards")
    parser.add_argument('--per_gpu_batch_size', type=int, default=32, help="Batch size to encode passages")
    parser.add_argument('--output_batch_size', type=int, default=5000000, help="Batch size to output embeddings")
    parser.add_argument('--passage_maxlength', type=int, default=200, help="Maximum number of tokens in a passage")
    parser.add_argument('--model_path', type=str, help="path to directory containing model weights and config file")
    parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    args = parser.parse_args()

    main(args,  is_main=True)


