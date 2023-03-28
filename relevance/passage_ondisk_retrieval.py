# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import os
import numpy as np
import torch
import transformers

import src.slurm
import src.util
import src.model
import src.data

from src.ondisk_index import OndiskIndexer
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

g_title_prefix='title:'
g_passage_prefix='context:'

def retrieve_data(opt, index, data, student_model, teacher_model, tokenizer, f_o):
    batch_size = 1
    dataset = src.data.Dataset(data, ignore_context=True)
    collator = src.data.Collator(opt.question_maxlength, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=1, collate_fn=collator)
    passage_collator = src.data.TextCollator(tokenizer, teacher_model.config.passage_maxlength)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            (index_info, _, _, question_ids, question_mask, _) = batch
            out_emb = student_model.question_encoder.embed_text(
                text_ids=question_ids.to(opt.device).view(-1, question_ids.size(-1)), 
                text_mask=question_mask.to(opt.device).view(-1, question_ids.size(-1)), 
                apply_mask=student_model.config.apply_question_mask,
                extract_cls=student_model.config.extract_cls,
            )
            query_emb = out_emb.cpu().numpy()
            result_lst = index.search(query_emb, top_n=opt.n_docs, n_probe=512, 
                                      min_tables=opt.min_tables, max_retr=opt.max_retr)
            assert(1 == len(result_lst))
            item_result = result_lst[0]
            data_item = data[index_info['index'][0]]
            ctxs_num = len(item_result)
            data_item['ctxs'] =[
                {
                    'id': int(item_result[c]['p_id']),
                    'title': '',
                    'text': item_result[c]['passage'],
                    'score': float(item_result[c]['score']),
                    'tag':item_result[c]['tag']
                } for c in range(ctxs_num)
            ]
            teacher_rerank(opt, teacher_model, passage_collator, question_ids, question_mask, data_item)       
            f_o.write(json.dumps(data_item) + '\n') 

def group_table_passages(item):
    ctx_lst = item['ctxs']
    table_lst = []
    table_dict = {}
    for ctx in ctx_lst:
        table_id = ctx['tag']['table_id']
        if table_id not in table_dict:
            table_dict[table_id] = []
            table_lst.append(table_id)
        table_dict[table_id].append(ctx)

    return table_lst, table_dict 

def sort_passages(opt, model, question_emb, collator, input_ctx_lst):
    input_passages = []
    for ctx in input_ctx_lst:
        passage_info = [ctx['id'], ctx['text'], ctx['title']]
        annoated_passage_info = src.data.TextDataset.annoate_passage(passage_info, g_title_prefix, g_passage_prefix)
        input_passages.append(annoated_passage_info)

    batch_size = 1000
    score_lst = []
    for start_pos in range(0, len(input_passages), batch_size):
        batch_passages = input_passages[start_pos:(start_pos + batch_size)]
        _, text_ids, text_mask = collator(batch_passages)
        with torch.no_grad():
            passage_embeddings = model.embed_text(
                    text_ids=text_ids.to(opt.device), 
                    text_mask=text_mask.to(opt.device), 
                    apply_mask=model.config.apply_passage_mask,
                    extract_cls=model.config.extract_cls,
                )  
            batch_scores = model.calc_score(question_emb, passage_embeddings).view(-1)
            score_lst.append(batch_scores) 
    
    all_scores = torch.cat(score_lst) 
    sorted_idxes = torch.argsort(-all_scores)
    sorted_idx_lst = sorted_idxes.cpu().numpy()
    return sorted_idx_lst


def teacher_rerank(opt, model, passage_collator, question_ids, question_mask, data_item):
    with torch.no_grad():
        question_emb = model.embed_text(
            text_ids=question_ids.to(opt.device).view(-1, question_ids.size(-1)), 
            text_mask=question_mask.to(opt.device).view(-1, question_ids.size(-1)), 
            apply_mask=model.config.apply_question_mask,
            extract_cls=model.config.extract_cls,
        )
   
    ctx_lst = data_item['ctxs']
    sorted_ctx_idxes = sort_passages(opt, model, question_emb, passage_collator, ctx_lst)
    sorted_ctx_lst = [ctx_lst[idx] for idx in sorted_ctx_idxes]
    data_item['ctxs'] = sorted_ctx_lst


def get_model(is_student, model_path, no_fp16):
    model = src.util.load_pretrained_retriever(is_student, model_path)
    model.cuda()
    model.eval()
    if not no_fp16:
        model = model.half()
    return model

def main(opt, index_obj=None):
    src.slurm.init_distributed_mode(opt)
    args = opt
    if os.path.exists(args.output_path):
        print('[%s] already exists' % args.output_path)
        return
    
    src.util.init_logger(is_main=True)
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    data = src.data.load_data(opt.data)
    
    student_model = get_model(True, opt.student_model_path, opt.no_fp16) 
    teacher_model = get_model(False, opt.teacher_model_path, opt.no_fp16) 

    if index_obj is None:
        index = OndiskIndexer(args.index_file, args.passage_file)
    else:
        index = index_obj

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
     
    with open(args.output_path, 'w') as f_o:
        retrieve_data(opt, index, data, student_model, teacher_model, tokenizer, f_o)

    student_model = student_model.cpu()
    teacher_model = teacher_model.cpu()
    logger.info(f'Saved results to {args.output_path}')

def read_passages(data_file):
    passages = []
    passage_file_lst = glob.glob(data_file)
    for passage_file in passage_file_lst:
        part_passages = src.util.load_passages(passage_file)
        passages.extend(part_passages)
    return passages

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_file', type=str)
    parser.add_argument('--passage_file', type=str)
    parser.add_argument('--data', required=True, type=str, default=None, 
                        help=".json file containing question and answers, similar format to reader data")
    parser.add_argument('--output_path', type=str, default=None, help='Results are written to output_path')
    parser.add_argument('--n-docs', type=int, default=1000, help="Number of documents to retrieve per questions")
    parser.add_argument('--student_model_path', type=str)
    parser.add_argument('--teacher_model_path', type=str) 
    parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    parser.add_argument('--question_maxlength', type=int, default=50, help="Maximum number of tokens in a question")
    parser.add_argument('--min_tables', type=int, default=5) 
    parser.add_argument('--max_retr', type=int, default=10000, help='maximum number of vectors to retrieve')

    args = parser.parse_args()
    main(args)


