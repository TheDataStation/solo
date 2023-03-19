# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
from pathlib import Path
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

import src.slurm
import src.util
import src.evaluation
import src.data
from src.model import Retriever
from src.student_retriever import StudentRetriever
from src.options import Options

import pickle
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import datetime
import glob

class Teacher:
    def __init__(self, model):
        self.model = model
        self.model.to(opt.device)
        self.model.eval()
        self.emb_precom_dict = None

    def calc_logits(self, batch, temperature):
        if self.emb_precom_dict is not None:
            question_encoded, passage_encoded = self.get_batch_embs(batch)  
            question_encoded = question_encoded.to(opt.device)
            passage_encoded = passage_encoded.to(opt.device)
        else:
            assert False, 'No embs'
            with torch.no_grad():
                (indexes, question_ids, question_mask, passage_ids, passage_mask, gold_score) = batch 
                question_encoded, passage_encoded = self.model(
                    question_ids=question_ids.cuda(),
                    question_mask=question_mask.cuda(),
                    passage_ids=passage_ids.cuda(),
                    passage_mask=passage_mask.cuda(),
                    encode_only=True,
                )
        
        score = self.model.calc_score(question_encoded, passage_encoded)
        soft_score = score / temperature
        logits = F.log_softmax(soft_score, dim=1)  
        return logits
         
    def precompute_teacher_embeddings(self, train_examples, tokenizer):
        collator = src.data.RetrieverCollator(
            tokenizer, 
            passage_maxlength=opt.passage_maxlength, 
            question_maxlength=opt.question_maxlength,
            all_passages=True,
        )
        dataset = src.data.Dataset(train_examples, n_context=None)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset, 
            sampler=sampler, 
            batch_size=1,
            drop_last=False, 
            collate_fn=collator
        )
        num_batches = len(dataloader)
        self.emb_precom_dict = {}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Teacher precomputing', total=num_batches):
                (index_info, question_ids, question_mask, context_ids, context_mask, meta_lst) = batch
                question_encoded, passage_encoded = self.model(
                    question_ids=question_ids.cuda(),
                    question_mask=question_mask.cuda(),
                    passage_ids=context_ids.cuda(),
                    passage_mask=context_mask.cuda(),
                    encode_only=True,
                ) 
                assert 1 == len(index_info['index'])
                emb_info = {
                    'q_emb':question_encoded.cpu(),
                    'ctx_emb':passage_encoded.cpu(),
                }
                q_id = index_info['q_ids'][0]
                self.emb_precom_dict[q_id] = emb_info
        
        emb_dir = os.path.dirname(opt.teacher_precompute_file)
        if not os.path.isdir(emb_dir):
            os.makedirs(emb_dir)
        
        logger.info('Writing teacher precomputed embeddings') 
        with open(opt.teacher_precompute_file, 'wb') as f_o:
            pickle.dump(self.emb_precom_dict, f_o)
   
    
    def read_teacher_embeddings(self, train_examples, tokenizer):
        emb_file_lst = glob.glob(opt.teacher_precompute_file)
        if len(emb_file_lst) == 0:
            self.precompute_teacher_embeddings(train_examples, tokenizer)
        else:
            self.emb_precom_dict = {}
            logger.info('Reading teacher precomputed embeddings')
            for emb_file in emb_file_lst:
                with open(emb_file, 'rb') as f:
                    part_emb_dict = pickle.load(f)
                    self.emb_precom_dict.update(part_emb_dict)
  
        
    def get_batch_embs(self, batch):
        emb_dict = self.emb_precom_dict
        question_vector_lst = []
        passage_vector_lst = []
        index_info = batch[0]
        q_id_lst = index_info['q_ids']
        meta_dict = batch[-1]
        batch_sample_ctx_idxes = meta_dict['sample_ctx_idxes']
        for batch_idx, q_id in enumerate(q_id_lst):
            emb_data = emb_dict[q_id]
            question_vector = emb_data['q_emb'].view(1, -1)
            question_vector_lst.append(question_vector)
            all_ctx_vector = emb_data['ctx_emb']
            passage_idxes = batch_sample_ctx_idxes[batch_idx]
            passage_emb_lst = [all_ctx_vector[a].view(1, -1) for a in passage_idxes] 
            passage_vector_lst.extend(passage_emb_lst)

        batch_q_vector = torch.cat(question_vector_lst, dim=0)
        batch_ctx_vector = torch.cat(passage_vector_lst, dim=0)
        return (batch_q_vector, batch_ctx_vector)


def train(model, optimizer, scheduler, global_step,
          train_dataset, dev_dataset, opt, collator_train, best_eval_loss, collator_eval):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')
    train_sampler = DistributedSampler(train_dataset) if opt.is_distributed else RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=opt.per_gpu_batch_size, 
        drop_last=True, 
        #num_workers=10, 
        collate_fn=collator_train,
    )
    distill_loss_fn = nn.MSELoss()
    loss, curr_loss = 0.0, 0.0
    epoch = 0
    num_batch = len(train_dataloader)
    while epoch <= opt.max_epoch:
        model.train()
        epoch += 1
        total_examples = 0
        total_correct_count = 0
        for b_idx, batch in enumerate(train_dataloader):
            epoch_step = b_idx + 1
            global_step += 1
           
            teacher_logits = model.teacher.calc_logits(batch, opt.distill_temperature)
            
            (idx, question_ids, question_mask, passage_ids, passage_mask, meta_dict) = batch
            pos_idxes_per_question = torch.tensor(meta_dict['global_pos_idxes']).cuda()
            student_score, student_loss, correct_count = model(
                question_ids=question_ids.cuda(),
                question_mask=question_mask.cuda(),
                passage_ids=passage_ids.cuda(),
                passage_mask=passage_mask.cuda(),
                pos_idxes_per_question=pos_idxes_per_question,
            )
            total_examples += 1
            total_correct_count += correct_count
            
            student_logits = model.calc_logits(student_score, opt.distill_temperature)
            distill_loss = distill_loss_fn(student_logits, teacher_logits) 
            train_loss = distill_loss * opt.distill_weight + student_loss * (1 - opt.distill_weight)
            
            train_loss.backward()
            
            if global_step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()
            
            logger.info('epoch=%d loss=%f step=%d/%d' % (epoch, train_loss.item(), 
                        epoch_step, num_batch))    

        evaluate(model, dev_dataset, collator_eval, opt, epoch)
        src.util.save(model, optimizer, scheduler, global_step, best_eval_loss, 
                      opt, dir_path, f"epoch-{epoch}")
    
    logger.info('best_eval_loss= best_eval_epoch=' % (best_eval_loss, best_eval_epoch))
        

def evaluate(model, dataset, collator, opt, epoch):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=opt.per_gpu_batch_size,
        drop_last=False, 
        #num_workers=10, 
        collate_fn=collator
    )
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    total = 0
    total_correct_count = 0
    eval_loss = .0
    num_batch = len(dataloader)
    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            step = b_idx + 1
            (idx, question_ids, question_mask, context_ids, context_mask, meta_dict) = batch
            pos_idxes_per_question = torch.tensor(meta_dict['global_pos_idxes']).cuda()
            _, loss, correct_count = model(
                question_ids=question_ids.cuda(),
                question_mask=question_mask.cuda(),
                passage_ids=context_ids.cuda(),
                passage_mask=context_mask.cuda(),
                pos_idxes_per_question=pos_idxes_per_question,
            )
            
            total += question_ids.size(0)
            total_correct_count += int(correct_count)
          
            if loss is not None:
                eval_loss += loss.item() 
            
            correct_ratio = total_correct_count / total
            show_eval_metric(epoch, num_batch, step, eval_loss, correct_ratio)
                
        global best_eval_loss
        global best_eval_epoch
        mean_loss = eval_loss / num_batch if eval_loss is not None else None 
        if mean_loss is not None:
            if best_eval_loss is None:
                best_eval_loss = mean_loss
                best_eval_epoch = epoch
            else:
                if mean_loss < best_eval_loss:
                    best_eval_loss = mean_loss
                    best_eval_epoch = epoch
    

def show_eval_metric(epoch, num_batch, step, eval_loss, correct_ratio):
    if eval_loss is not None:
        mean_loss = eval_loss / step 
        logger.info('Eval: epoch=%d loss=%f correct ratio=%f step=%d/%d' % (
            epoch, mean_loss, correct_ratio, step, num_batch))
    else:
        logger.info('Eval: epoch=%d loss=%s correct ratio=%f step=%d/%d' % (
            epoch, 'None', correct_ratio, step, num_batch))


def load_teacher(train_examples, tokenizer):
    assert opt.teacher_precompute_file is not None
    assert opt.teacher_model_path is not None
    teacher_model = Retriever.from_pretrained(opt.teacher_model_path)
    teacher_model.model.pooler = None
    teacher_state_dict = teacher_model.state_dict()
    teacher = Teacher(teacher_model)
    teacher.read_teacher_embeddings(train_examples, tokenizer)
    return teacher, teacher_state_dict

def get_expr_name(tag):
    now_time = datetime.datetime.now()
    expr_name = '%s_%d_%d_%d_%d_%d_%d' % (tag, 
                now_time.year, now_time.month, now_time.day, now_time.hour, now_time.minute, now_time.second)
    return expr_name 

if __name__ == "__main__":
    options = Options()
    options.add_retriever_options()
    options.add_optim_options()
    opt = options.parse()
    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    opt.name = get_expr_name('train' if opt.do_train else 'eval') 
    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    assert (not directory_exists), '%s already exists' % str(dir_path)  

    dir_path.mkdir(parents=True, exist_ok=True)
    if not directory_exists and opt.is_main:
        options.print_options(opt)

    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    #Load data
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
     
    if opt.do_train:
        collator_train = src.data.RetrieverCollator(
            tokenizer, 
            passage_maxlength=opt.passage_maxlength, 
            question_maxlength=opt.question_maxlength,
            sample_pos_ctx=True,
            sample_neg_ctx=True,
            num_neg_ctxs=opt.num_train_neg_ctxs,
        )
        logger.info('loading %s' % opt.train_data)
        train_examples = src.data.load_data(opt.train_data)
        train_dataset = src.data.Dataset(train_examples, opt.n_context)

    collator_eval = src.data.RetrieverCollator(
        tokenizer, 
        passage_maxlength=opt.passage_maxlength, 
        question_maxlength=opt.question_maxlength,
        sample_pos_ctx=False,
        sample_neg_ctx=False,
        num_neg_ctxs=opt.num_eval_neg_ctxs,
    )
    logger.info('loading %s' % opt.eval_data)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size, 
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    global_step = 0
    best_eval_loss = None
    best_eval_epoch = -1
    config = src.model.RetrieverConfig(
        indexing_dimension=opt.indexing_dimension,
        apply_question_mask=not opt.no_question_mask,
        apply_passage_mask=not opt.no_passage_mask,
        extract_cls=opt.extract_cls,
        projection=False,
    )
    
    if opt.do_train:
        model_class = StudentRetriever
        teacher, teacher_state_dict = load_teacher(train_examples, tokenizer)
        if opt.model_path == "none":
            model = model_class(config, teacher_state_dict=teacher_state_dict)
            src.util.set_dropout(model, opt.dropout)
            model = model.to(opt.device)
            optimizer, scheduler = src.util.set_optim(opt, model)
        else:
            assert False, 'Not supported.'
        model.set_teacher(teacher)
    else:
        assert (opt.model_path != 'none') and (opt.model_path is not None)
        model = src.util.load_pretrained_retriever(opt.is_student, opt.model_path)
        logger.info(f"Model loaded from {opt.model_path}")
         
    model = model.to(opt.device)
    if opt.do_train: 
        optimizer, scheduler = src.util.set_optim(opt, model)
        train(model, optimizer, scheduler, global_step, train_dataset, eval_dataset, opt,
              collator_train, best_eval_loss, collator_eval)
    else:
        evaluate(model, eval_dataset, collator_eval, opt, 0) 
    
     
