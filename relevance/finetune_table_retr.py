# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model
from tqdm import tqdm
import os
import json
from src.retr_model_mle import RetrModelMLE
from src.retr_model_bnn import RetrModelBNN
from src.general_retr_loss import FusionGeneralRetrLoss 
from src.general_retr_bnn_loss import FusionGeneralRetrBNNLoss
import torch.optim as optim
import time
from src.retr_utils import MetricRecorder, get_top_metrics
import logging
import torch.nn.functional as F
import glob
import datetime


logging.basicConfig(level=logging.ERROR)

Num_Answers = 1
global_steps = 0
best_metric_info = {}

def get_coreset_loader(coreset_method, opt, train_data_dict, train_dataset, train_batch, collator):
    train_qid_lst = None
    if train_batch is not None:
        train_item_idxes = train_batch[0]
        train_qid_lst = [train_dataset.data[a]['id'] for a in train_item_idxes]

    coreset_qid_lst = coreset_method.get_coreset(train_qid_lst)
    data = []
    for qid in coreset_qid_lst:
        data.append(train_data_dict[qid])

    coreset_dataset = src.data.Dataset(data, opt.n_context)
    coreset_sampler = SequentialSampler(coreset_dataset)
    coreset_loader = DataLoader(
        coreset_dataset,
        sampler=coreset_sampler,
        batch_size=opt.per_gpu_eval_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator)
    return coreset_dataset, coreset_loader

def stat_coreset(model, retr_model, coreset_method, step_info, opt, train_data_dict,
                 train_dataset, train_batch, collator, tokenizer):
    coreset_dataset, coreset_loader = get_coreset_loader(coreset_method, opt,
                                                train_data_dict, train_dataset, train_batch, collator)
    coreset_metrics = None
    desc = 'stat train step %d' % step_info['step']

    for coreset_batch in tqdm(coreset_loader, total=len(coreset_loader), desc=desc):
        item_idxes = coreset_batch[0]
        coreset_metrics = evaluate_train(opt, model, retr_model, coreset_dataset, coreset_batch)
        coreset_method.do(coreset_dataset, item_idxes, coreset_metrics, step_info)

    coreset_method.on_checkpoint(step_info)


def init_global():
    global Num_Answers
    global global_steps
    global best_metric_info

    Num_Answers = 1
    global_steps = 0
    best_metric_info = {}

def get_device(cuda):
    device = torch.device(("cuda:%d" % cuda) if torch.cuda.is_available() and cuda >=0 else "cpu")
    return device

def get_loss_fn(opt):
    if not opt.bnn:
        loss_fn = FusionGeneralRetrLoss()
        logger.info('loss function, FusionGeneralRetrLoss')
    else:
        loss_fn = FusionGeneralRetrBNNLoss()
        logger.info('loss function, FusionGeneralRetrBNNLoss')

    return loss_fn

def load_prior_model(opt):
    prior = None
    state_dict = None
    if (opt.prior_model is not None) and (opt.prior_model != 'none'):
        print('loading prior %s' % opt.prior_model)
        state_dict = torch.load(opt.prior_model, map_location=opt.device)
        model_bnn = RetrModelBNN()
        model_bnn.load_state_dict(state_dict)
        prior = {}
        for item in [
                        ['passage_fnt', model_bnn.passage_fnt], 
                        ['table_fnt', model_bnn.table_fnt], 
                        ['feat_l1', model_bnn.feat_l1],
                        ['feat_l2', model_bnn.feat_l2]
                    ]:
            prior[item[0]] = {
                'weight_mu':item[1].weight.mu.detach().to(opt.device),
                'weight_sigma':item[1].weight.sigma.detach().to(opt.device),
                'bias_mu':item[1].bias.mu.detach().to(opt.device),
                'bias_sigma':item[1].bias.sigma.detach().to(opt.device),
            }
    return prior, state_dict

def get_retr_multi_models(opt):
    multi_model_map = {}
    file_pattern = os.path.join(opt.multi_model_dir, '*_step_*_model.pt')
    data_file_lst = glob.glob(file_pattern)
    for data_file in data_file_lst:
        base_name = os.path.basename(data_file)
        step = int(base_name.split('_')[-2])
        opt.fusion_retr_model = data_file
        step_retr_model = get_retr_model(opt, apply_device=False)
        multi_model_map[step] = step_retr_model
    return multi_model_map  
    
def get_retr_model(opt, apply_device=True):
    if not opt.bnn:
        logger.info('Model, RetrModelMLE')
        retr_model = RetrModelMLE()
    else:
        logger.info('Model, RetrModelBNN')
        #import pdb; pdb.set_trace()
        prior, prior_state_dict = load_prior_model(opt)
        retr_model = RetrModelBNN()
        #if (prior_state_dict is not None) and opt.do_train:
        #    retr_model.load_state_dict(prior_state_dict)
        retr_model.set_prior(prior) 

    if opt.fusion_retr_model is not None:
        logger.info('loading pretrained model (%s)' % opt.fusion_retr_model)
        state_dict = torch.load(opt.fusion_retr_model, map_location=opt.device)
        retr_model.load_state_dict(state_dict)

    if apply_device:
        retr_model = retr_model.to(opt.device) 
    return retr_model

def write_predictions(f_o_pred, item, top_idxes, scores):
    qid = item['qid']
    table_id_lst = item['table_id_lst']
    question = item['question']
    passages = item['passages']
    p_id_lst = item['p_id_lst']
    tags = item['tags']
    out_item = {
        'qid':qid,
        'table_id_lst':table_id_lst,
        'question':question,
        'passages':[passages[a] for a in top_idxes],
        'p_id_lst':[p_id_lst[a] for a in top_idxes],
        'tags':[tags[a] for a in top_idxes],
        'scores':[float(scores[a]) for a in top_idxes]
    }
    f_o_pred.write(json.dumps(out_item) + '\n')



def log_metrics(epoc, metric_rec,
                batch_data, batch_score, batch_answers, 
                time_used, total_time,
                itr, num_batch, 
                loss=None, model_tag=None, f_o_pred=None):
    batch_size = len(batch_score)
    batch_sorted_idxes = []
    answer_num_lst = []
    for b_idx in range(batch_size):
        item_scores = batch_score[b_idx]
        answer_scores = item_scores 
        scores = answer_scores.data.cpu().numpy()
        sorted_idxes = np.argsort(-scores)
        item_answer_lst = batch_answers[b_idx]
        sorted_ems = [item_answer_lst[idx]['em'] for idx in sorted_idxes]
        top_metrics = get_top_metrics(sorted_idxes, sorted_ems, batch_data[b_idx], 5)
        metric_rec.update(top_metrics)
        
        batch_sorted_idxes.append(sorted_idxes)
        answer_num_lst.append(len(sorted_idxes))
        if f_o_pred is not None:
            write_predictions(f_o_pred, batch_data[b_idx], sorted_idxes, scores)

    metric_dict = metric_rec.get_mean()
    str_info = ('epoc=%d ' % epoc) if epoc is not None else ''
    if loss is None:
        str_info = ' ' + str_info
    if model_tag is not None:
        str_info += '%s ' % model_tag
    if not loss is None:
        str_info += 'loss=%.6f ' % loss
    
    for max_top in metric_dict:
        str_info += 'p@%d=%.2f ' % (max_top, metric_dict[max_top]['metric_mean'])
    
    str_info += 'time=%.2f total=%.2f %d/%d' % (time_used, total_time, itr, num_batch)
    if loss is not None:
        str_info += ' global_steps=%d' % global_steps
    print(str_info)
    return batch_sorted_idxes

def bnn_predict_2(model, batch_data, fusion_scores, fusion_states, passage_masks):
    retr_scores = model(batch_data, fusion_scores, fusion_states, passage_masks, sample=False)
    return retr_scores

def bnn_predict(model, batch_data, fusion_scores, fusion_states, passage_masks, num_samples=0):
    log_prob_lst = [] 
    for sample_idx in range(num_samples):
        retr_scores = model(batch_data, fusion_scores, fusion_states, passage_masks, sample=True)
        retr_log_probs = F.logsigmoid(torch.stack(retr_scores)).unsqueeze(0)
        log_prob_lst.append(retr_log_probs)
    
    retr_scores = model(batch_data, fusion_scores, fusion_states, passage_masks, sample=False)
    retr_log_probs = F.logsigmoid(torch.stack(retr_scores)).unsqueeze(0)
    log_prob_lst.append(retr_log_probs)
    
    sample_log_probs = torch.cat(log_prob_lst, dim=0)
    bnn_scores = sample_log_probs.mean(dim=0)
     
    return bnn_scores 

def evaluate_train(opt, model, retr_model, dataset, fusion_batch):
    acc_lst = []
    model.eval()
    retr_model.eval()

    scores, score_states, examples, context_mask = get_score_info(model, fusion_batch, dataset)
    batch_data = get_batch_data(examples)
    if not opt.bnn:
        retr_scores = retr_model(batch_data, scores, score_states, context_mask)    
    else:
        retr_scores = bnn_predict(retr_model, batch_data, scores, score_states, context_mask)

    batch_size = len(retr_scores)
    #import pdb; pdb.set_trace()
    for b_idx in range(batch_size):
        item_scores = retr_scores[b_idx]
        scores = item_scores.data.cpu().numpy()
        top_idx = np.argmax(scores)
        correct = batch_data[b_idx]['answers'][top_idx]['em'] 
        acc_lst.append(correct)

    retr_model.train()
    return acc_lst

def evaluate(epoc, model, retr_model, dataset, dataloader, tokenizer, opt, 
             model_tag=None, out_dir=None, model_file=None, num_samples=0):
    #logger.info('Start evaluation')
    model.eval()
    retr_model.eval()
  
    f_o_pred = None
    f_o_metric = None
    if out_dir is not None:
        out_pred_file = os.path.join(out_dir, 'pred_epoch_%d_%s.jsonl' % (epoc, model_tag))
        f_o_pred = open(out_pred_file, 'w')
        out_metric_file = os.path.join(out_dir, 'metric_epoch_%d_%s.jsonl' % (epoc, model_tag))
        f_o_metric = open(out_metric_file, 'w')
   
    metric_rec = MetricRecorder([1, 5]) 
    total_time = .0 
    model.overwrite_forward_crossattention()
    model.reset_score_storage()
    with torch.no_grad():
        num_batch = len(dataloader)
        if opt.sql_batch_no is not None:
            bar_desc = 'data %d epoch %d evaluation' % (opt.sql_batch_no, epoc)
        else:
            bar_desc = 'epoch %d evaluation' % epoc
        for itr, fusion_batch in tqdm(enumerate(dataloader), total=num_batch, desc=bar_desc):
            t1 = time.time()

            scores, score_states, examples, context_mask = get_score_info(model, fusion_batch, dataset)
            batch_data = get_batch_data(examples)
            if not opt.bnn:
                retr_scores = retr_model(batch_data, scores, score_states, context_mask)    
            else:
                retr_scores = bnn_predict(retr_model, batch_data, scores, score_states, 
                                          context_mask, num_samples=num_samples)
            batch_answers = get_batch_answers(batch_data)
             
            t2 = time.time()
            time_span = t2 - t1
            total_time += time_span
           
            log_metrics(epoc, metric_rec, batch_data, retr_scores, batch_answers, time_span, total_time, 
                        (itr + 1), num_batch, loss=None, model_tag=model_tag, f_o_pred=f_o_pred) 
   
    metric_dict = metric_rec.get_mean()
    if opt.sql_batch_no is None:
        str_info =  'Model, %s \n' % opt.fusion_retr_model
        str_info += 'Accuracy, '
        for max_top in metric_dict:
            str_info += 'p@%d=%.2f ' % (max_top, metric_dict[max_top]['metric_mean'])
        logger.info(str_info)
        print(str_info)
    
    eval_metric_info = {}
    for max_top in metric_dict:
        eval_metric_info['p@%d' % max_top] =  metric_dict[max_top]['metric_mean']
    f_o_metric.write(json.dumps(eval_metric_info))

    if f_o_pred is not None:
        f_o_pred.close()
        f_o_metric.close() 
    
    update_best_metric(epoc, metric_rec, model_tag, out_dir, model_file)
    retr_model.train()





def evaluate_multiple_models(model, retr_model_map, dataset, dataloader, tokenizer, opt, out_dir=None):
    model.eval()
    f_o_map = {}
    for step in retr_model_map: 
        out_pred_file = os.path.join(out_dir, 'pred_step_%d.jsonl' % step)
        f_o_pred = open(out_pred_file, 'w')
        f_o_map[step] = f_o_pred
   
    model.overwrite_forward_crossattention()
    model.reset_score_storage()
    with torch.no_grad():
        num_batch = len(dataloader)
        bar_desc = 'multi_model evaluation'
        for itr, fusion_batch in tqdm(enumerate(dataloader), total=num_batch, desc=bar_desc):
            fusion_scores, score_states, examples, context_mask = get_score_info(model, fusion_batch, dataset)
            batch_data = get_batch_data(examples)
            batch_size = len(batch_data)
            for step in retr_model_map:
                retr_model_step = retr_model_map[step].to(opt.device)
                retr_model_step.eval()
                f_o_step = f_o_map[step]
                if not opt.bnn:
                    retr_scores = retr_model_step(batch_data, fusion_scores, score_states, context_mask)    
                else:
                    retr_scores = bnn_predict(retr_model_step, batch_data, fusion_scores,
                                              score_states, context_mask, num_samples=0)
                for b_idx in range(batch_size):
                    item_scores = retr_scores[b_idx]
                    scores = item_scores.data.cpu().numpy()
                    top_idx = np.argmax(scores)
                    correct = batch_data[b_idx]['answers'][top_idx]['em'] 
                    qid = batch_data[b_idx]['qid']
                    out_item_info = {
                        'step':step,
                        'qid':qid,
                        'correct':correct
                    }
                    f_o_step.write(json.dumps(out_item_info) + '\n')
    for step in f_o_map:
        f_o_map[step].close()

        
def update_best_metric(epoc, metric_rec, model_tag, out_dir, model_file):
    metric_dict = metric_rec.metric_dict 
    if 'p@1' not in best_metric_info:
        best_metric_info['N'] = metric_rec.N
        best_metric_info['p@1'] = metric_dict[1]['metric_sum'] # metric_sum is integer and can use =
        best_metric_info['p@5'] = metric_dict[5]['metric_sum']
        best_metric_info['model_file'] = model_file
        best_metric_info['epoch'] = epoc
        best_metric_info['patience_steps'] = 0
    else:
        best_metric_info['patience_steps'] += 1
        cur_best_p_at_1 = best_metric_info['p@1']
        cur_best_p_at_5 = best_metric_info['p@5']
        if metric_dict[1]['metric_sum'] > cur_best_p_at_1:
            best_metric_info['p@1'] = metric_dict[1]['metric_sum']
            best_metric_info['p@5'] = metric_dict[5]['metric_sum']
            best_metric_info['model_file'] = model_file
            best_metric_info['epoch'] = epoc
            best_metric_info['patience_steps'] = 0
        elif metric_dict[1]['metric_sum'] == cur_best_p_at_1:
            if metric_dict[5]['metric_sum'] > cur_best_p_at_5:
                best_metric_info['p@5'] = metric_dict[5]['metric_sum'] 
                best_metric_info['model_file'] = model_file
                best_metric_info['epoch'] = epoc
                best_metric_info['patience_steps'] = 0
    
    best_metric_file = os.path.join(out_dir, 'best_metric_info.json') 
    with open(best_metric_file, 'w') as f_o:
        f_o.write(json.dumps(best_metric_info))

def should_stop_train(opt, coreset_method=None):
    if (coreset_method is None) and ('patience_steps' in best_metric_info):
        return best_metric_info['patience_steps'] > opt.patience_steps    
    else:
        return False

def get_step_info(step, dataset, batch):
    qid_lst = None
    if batch is not None:
        item_idxes = batch[0]
        qid_lst = [dataset.data[a]['id'] for a in item_idxes]
    step_info = {
        'step':step,
        'batch':qid_lst
    }
    return step_info
 
def train(model, retr_model, 
          train_dataset, collator,
          eval_dataset, eval_dataloader, 
          tokenizer, 
          opt, coreset_method=None, train_data_dict=None):

    loss_fn = get_loss_fn(opt)
    learing_rate = 1e-3
    optimizer = optim.Adam(retr_model.parameters(), lr=learing_rate)
    
    global global_steps
    global_steps = 0
    total_time = .0
        
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        #num_workers=10,
        collate_fn=collator
    )
    model.eval()
    if hasattr(model, "module"):
        model = model.module

    model.overwrite_forward_crossattention()
    model.reset_score_storage() 
    assert(Num_Answers == 1) 
    num_batch = len(train_dataloader)
    
    checkpoint_steps = num_batch
    #epoch_ckp_num = int(num_batch / checkpoint_steps)
    #if num_batch % checkpoint_steps:
    #    epoch_ckp_num += 1
    opt.patience_steps = opt.patience_epochs # if it does not improve in 2 continuous epoches, training stops 
   
    if coreset_method is not None:
        coreset_method.init_data(train_dataset.data)
        coreset_method.set_epoch_steps(num_batch)
        step_info = get_step_info(0, None, None)
        stat_coreset(model, retr_model, coreset_method,
                     step_info, opt, train_data_dict, train_dataset, None, collator, tokenizer)
    
    epoc_bar_desc = 'data %d epoch' % opt.sql_batch_no
    for epoc in tqdm(range(opt.max_epoch), desc=epoc_bar_desc):
        metric_rec = MetricRecorder([1, 3, 5])
        bar_desc = 'data %d epoch %d train' % (opt.sql_batch_no, epoc)
        for itr, fusion_batch in tqdm(enumerate(train_dataloader), total=num_batch, desc=bar_desc):
            t1 = time.time()
            
            scores, score_states, examples, context_mask = get_score_info(model, fusion_batch, train_dataset)
            batch_data = get_batch_data(examples)
            opts = {}
            assert(retr_model.training)
            retr_scores = retr_model.sample_forward(batch_data, scores, score_states, 
                                                    context_mask, num_samples=1, opts=opts) 
            batch_answers = get_batch_answers(batch_data) 
            loss = loss_fn(retr_scores, batch_answers, opts=opts)
            optimizer.zero_grad()
            
            #if loss is None:
                #continue
            
            loss.backward()
            optimizer.step()

            global_steps += 1

            if coreset_method is not None:
                step_info = get_step_info(global_steps, train_dataset, fusion_batch)
                stat_coreset(model, retr_model, coreset_method, step_info, opt, train_data_dict,
                            train_dataset, fusion_batch, collator, tokenizer)
            
            t2 = time.time()
            time_span = t2 - t1
            total_time += time_span
           
            log_metrics(epoc, metric_rec, batch_data, retr_scores, batch_answers, time_span, total_time, 
                        (itr + 1), num_batch, loss.item()) 
            
            if (global_steps % checkpoint_steps == 0) and (coreset_method is None):
                model_tag = 'step_%d' % global_steps
                out_dir = os.path.join(opt.checkpoint_dir, opt.name)
                checkpoint_model_file = save_model(out_dir, retr_model, epoc, tag=model_tag, opt=opt) 
               
                if eval_dataset is not None: 
                    evaluate(epoc, model, retr_model,
                             eval_dataset, eval_dataloader,
                             tokenizer, opt, model_tag=model_tag, out_dir=out_dir, 
                             model_file=checkpoint_model_file)
                
                if should_stop_train(opt, coreset_method=coreset_method):
                    break
       
        if should_stop_train(opt, coreset_method=coreset_method):
            logger.info('Training is stopped because of the patience_steps setting')
            break
   
    #best_metric = get_best_metric() 
    #best_summary = 'Best performance, ' + str(get_best_metric())
    return copy.deepcopy(best_metric_info)

def get_best_metric():
    best_metric = copy.deepcopy(best_metric_info)
    N = best_metric['N']
    best_metric['p@1'] = best_metric['p@1'] * 100 / N
    best_metric['p@5'] = best_metric['p@5'] * 100 / N 
    del best_metric['N'] 
    return best_metric

def get_batch_answers(batch_data):
    batch_answers = []
    for item in batch_data:
        answers = item['answers']
        batch_answers.append(answers)
    return batch_answers

def save_model(output_dir, model, epoc, tag='step', opt=None):
    stamp_info = datetime.datetime.now()
    file_name = 'sql_%d_epoc_%d_%s_model_%d_%d_%d.pt' % (opt.sql_batch_no, epoc, tag, 
                                                         stamp_info.month, stamp_info.day, stamp_info.year)
    out_path = os.path.join(output_dir, file_name)
    torch.save(model.state_dict(), out_path) 
    return out_path

def get_batch_data(fusion_examples):
    batch_data = []
    for f_example in fusion_examples:
        gold_table_lst = f_example['table_id_lst']
        ctx_data = f_example['ctxs']
        
        item_answers = []
        for passage_info in ctx_data:
            answer_info = {
                'em':int(passage_info['tag']['table_id'] in gold_table_lst)
            }
            item_answers.append(answer_info)
        
        item_data = {
            'qid':f_example['id'],
            'table_id_lst':gold_table_lst,
            'question':f_example['question'],
            'passages':[a['text'] for a in ctx_data],
            'p_id_lst':[a['id'] for a in ctx_data],
            'answers':item_answers,
            'tags':[a['tag'] for a in ctx_data]
        }
        batch_data.append(item_data)
    
    return batch_data 
    
def get_score_info(model, batch_data, dataset):
    with torch.no_grad():
        (index_info, _, _, context_ids, context_mask, _) = batch_data
        model.reset_score_storage()
        outputs = model.generate(
            input_ids=context_ids.cuda(),
            attention_mask=context_mask.cuda(),
            max_length=50,
            num_beams=Num_Answers,
            num_return_sequences=Num_Answers
        )
        crossattention_scores, score_states = model.get_crossattention_scores(context_mask.cuda())
        num_passages = crossattention_scores.shape[1]
        batch_examples = []
        batch_idxes = index_info['index'] 
        for k, _ in enumerate(outputs):
            example = dataset.data[batch_idxes[k]]
            example['ctxs'] = example['ctxs'][:num_passages]
            batch_examples.append(example)
    context_mask = context_mask.to(crossattention_scores.device)
    return crossattention_scores, score_states, batch_examples, context_mask
    
def set_logger(opt):
    global logger
    logger = logging.getLogger(__name__)
    logger.handlers = []
    logger.setLevel(logging.INFO)

    logger.propagate = False
    if opt.debug:
        console = logging.StreamHandler()
        logger.addHandler(console)

    file_path = os.path.join(opt.checkpoint_dir, opt.name, 'log.txt')
    file_hander = logging.FileHandler(file_path, 'w')
    logger.addHandler(file_hander)

def print_args(opts):
    logger.info(opts)

def main(opt, coreset_method=None):
    init_global()
    
    src.slurm.init_distributed_mode(opt)
    #src.slurm.init_signal_handler()
    
    assert(opt.world_size == 1)

    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)
    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if directory_exists:
        err_msg = '(%s) already exists.' % str(dir_path)
        msg_info = {
            'state':False,
            'msg':err_msg
        }
        return msg_info

    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    global logger
    set_logger(opt)
    print_args(opt)

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer)
   
    device = get_device(opt.cuda)
    opt.device = device

    if not opt.multi_model_eval:   
        retr_model = get_retr_model(opt)
    else:
        retr_model_map = get_retr_multi_models(opt)

    eval_examples = None
    eval_dataset = None
    eval_sampler = None
    eval_dataloader = None
    if (coreset_method is None) and (opt.eval_data is not None): 
        eval_examples = src.data.load_data(
            opt.eval_data, 
            global_rank=opt.global_rank, 
            world_size=opt.world_size,
        )
    
        eval_dataset = src.data.Dataset(
            eval_examples, 
            opt.n_context,
            sort_by_score=False 
        )

        eval_sampler = SequentialSampler(eval_dataset) 
        eval_dataloader = DataLoader(
            eval_dataset, 
            sampler=eval_sampler, 
            batch_size=opt.per_gpu_eval_batch_size,
            num_workers=0, 
            collate_fn=collator_function
        )
    
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    if opt.do_train:
        train_examples = src.data.load_data(
            opt.train_data,
            global_rank=opt.global_rank,
            world_size=opt.world_size,
        )
        train_dataset = src.data.Dataset(train_examples, opt.n_context, sort_by_score=False)

        #import pdb; pdb.set_trace()        
        train_data_dict = {}
        for example in train_examples:
            train_data_dict[example['id']]= example
        
        best_metric = train(model, retr_model, 
                            train_dataset, collator_function,
                            eval_dataset, eval_dataloader, 
                            tokenizer, opt, coreset_method=coreset_method, train_data_dict=train_data_dict)

        msg_info = {
            'state':True,
            'best_metric':best_metric
        }
        return msg_info
    else:
        out_dir = os.path.join(opt.checkpoint_dir, opt.name)
        if not opt.multi_model_eval:
            evaluate(0, model, retr_model,
                    eval_dataset, eval_dataloader,
                    tokenizer, opt, out_dir=out_dir, num_samples=opt.bnn_num_eval_sample)
            msg_info = {
                'state':True,
                'out_dir':out_dir
            }
            return msg_info
        else:
            evaluate_multiple_models(model, retr_model_map, eval_dataset, 
                                     eval_dataloader, tokenizer, opt,
                                     out_dir=out_dir)
            msg_info = {
                'state':True
            }
            return msg_info

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    args = options.parse()
    args.sql_batch_no = 0
    msg_info = main(args)
    if not msg_info.get('state', False):
        print(msg_info['msg'])


