# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import torch.nn as nn
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
from tqdm import tqdm

def get_group_info(passage_tags):
    group_dict = {}
    for idx, tag_info in enumerate(passage_tags):
        table_id = tag_info['table_id']
        if table_id not in group_dict:
            group_dict[table_id] = []
        group = group_dict[table_id]
        group.append(idx)
    return group_dict 

def compute_table_loss(table_scores, pos_idxes, neg_idxes):
    assert(len(pos_idxes) > 0)
    assert(len(neg_idxes) > 0)
    score_lst = []
    for pos_idx in pos_idxes:
        idx_lst = [pos_idx] + neg_idxes
        item_score = table_scores[idx_lst].view(1, -1)
        score_lst.append(item_score)

    batch_item_score = torch.cat(score_lst, dim=0)
    batch_item_labels = torch.zeros(len(pos_idxes)).long().to(table_scores.device)
    
    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(batch_item_score, batch_item_labels)
    return loss

def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        #num_workers=10,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    num_batch = len(train_dataloader)
    while step < opt.total_steps:
        epoch += 1
        for i, batch in tqdm(enumerate(train_dataloader), total=num_batch):
            
            (idx, labels, _, context_ids, context_mask, batch_passage_tags) = batch
            assert(len(labels) == 1) # bsz must be 1 right now
            
            batch_loss = []
             
            for item_idx, passage_tags in enumerate(batch_passage_tags):
                group_dict = get_group_info(passage_tags) 
                
                group_score_lst = []
                gold_table_lst = train_dataset.get_example(idx[item_idx])['table_id_lst'] 

                pos_idxes = []
                neg_idxes = []
                for table_idx, table_id in enumerate(group_dict):
                    table_label = (table_id in gold_table_lst)
                    if table_label:
                        pos_idxes.append(table_idx)
                    else:
                        neg_idxes.append(table_idx)

                    passage_idxes = group_dict[table_id]
                    group_context_ids = context_ids[item_idx][passage_idxes].unsqueeze(0)
                    group_context_mask = context_mask[item_idx][passage_idxes].unsqueeze(0)
                    group_labels = labels[item_idx].unsqueeze(0)
                    opt_info = {}
                    
                    group_train_loss = model(
                        input_ids=group_context_ids.cuda(),
                        attention_mask=group_context_mask.cuda(),
                        labels=group_labels.cuda(),
                        opt_info=opt_info 
                    )[0]
                    
                    _, num_passages, num_tokens = group_context_ids.shape  
                    encoded_states = opt_info['encoded_states'].view(1, num_passages, num_tokens, -1)
                    
                    group_passage_states = encoded_states[0,:,0,:] # first token state as passage state
                    group_passage_scores = model.group_fn(group_passage_states)
                    group_score = group_passage_scores.max()
                    group_score_lst.append(group_score)
                
                if len(pos_idxes) > 0 and len(neg_idxes) > 0: 
                    table_scores = torch.stack(group_score_lst)
                    table_loss = compute_table_loss(table_scores, pos_idxes, neg_idxes)
                    
                    item_loss = group_train_loss + table_loss
                    batch_loss.append(item_loss)

            if len(batch_loss) == 0:
                model.zero_grad()
                continue
            step += 1
            if len(batch_loss) == 1:
                train_loss = batch_loss[0]
            else:
                train_loss = torch.stack(batch_loss).mean()
             
            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()
            
            if step % opt.checkpoint_steps == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    curr_loss = 0
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)

            if opt.is_main and step % opt.checkpoint_steps == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        #num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    table_match = []
    with torch.no_grad():
        num_batch = len(dataloader)
        for i, batch in tqdm(enumerate(dataloader), total=num_batch):
            (idx, _, _, context_ids, context_mask, batch_passage_tags) = batch
            
            for item_idx, passage_tags in enumerate(batch_passage_tags):
                group_dict = get_group_info(passage_tags)
                
                gold_table_lst = dataset.get_example(idx[item_idx])['table_id_lst']
                
                group_score_lst = []
                pred_table_lst = []
                for table_idx, table_id in enumerate(group_dict):
                    pred_table_lst.append(table_id)
                    passage_idxes = group_dict[table_id]
                    group_context_ids = context_ids[item_idx][passage_idxes].unsqueeze(0)
                    group_context_mask = context_mask[item_idx][passage_idxes].unsqueeze(0)
                    opt_info = {}
                    
                    outputs = model.generate(
                        input_ids=group_context_ids.cuda(),
                        attention_mask=group_context_mask.cuda(),
                        max_length=50,
                        opt_info=opt_info
                    )
                    
                    _, num_passages, num_tokens = group_context_ids.shape
                    encoded_states = opt_info['encoded_states'].view(1, num_passages, num_tokens, -1)
                    group_passage_states = encoded_states[0,:,0,:] # first token state as passage state
                    group_passage_scores = model.group_fn(group_passage_states)
                    group_score = group_passage_scores.max()
                    group_score_lst.append(group_score)
                
                table_scores = torch.stack(group_score_lst).cpu().numpy()
                pred_table_idx = table_scores.argmax()
                pred_top_table = pred_table_lst[pred_table_idx]

                correct = int(pred_top_table in gold_table_lst)
                table_match.append(correct)

                '''
                for k, o in enumerate(outputs):
                    ans = tokenizer.decode(o, skip_special_tokens=True)
                    gold = dataset.get_example(idx[k])['answers']
                    score = src.evaluation.ems(ans, gold)
                    total += 1
                    exactmatch.append(score)

                    gold_table_lst = dataset.get_example(idx[k])['table_id_lst']
                    correct = int(batch_pred_tables[0] in gold_table_lst)
                    table_match.append(correct)
                '''
            accuracy = np.mean(table_match) * 100
            logger.info('p@1=%.2f batch=%d' % (accuracy, i))

    accuracy = np.mean(table_match) * 100
    #exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return accuracy

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    model_class = src.model.FiDT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
        #backward=opt.backward
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
        #backward=opt.backward
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    if not checkpoint_exists and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
