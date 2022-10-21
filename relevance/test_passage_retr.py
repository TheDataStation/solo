# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler


import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model
from tqdm import tqdm
import os
import json

Num_Answers = 1

def evaluate(model, dataset, dataloader, tokenizer, opt):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    
    model.overwrite_forward_crossattention()
    model.reset_score_storage() 
   
    assert(Num_Answers == 1) 
    
    table_pred_results = {1:[], 5:[]}
    count = 0
    with torch.no_grad():
        num_batch = len(dataloader)
        for i, batch in tqdm(enumerate(dataloader), total=num_batch):
            (idx, _, _, context_ids, context_mask) = batch

            model.reset_score_storage()

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50,
                num_beams=Num_Answers,
                num_return_sequences=Num_Answers
            )

            crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

            assert(len(outputs) == (len(idx) * Num_Answers))
            outputs = outputs.reshape(len(idx), Num_Answers, -1)
            for k, answer_code_lst in enumerate(outputs):
                assert(len(answer_code_lst) == 1)
                answer_code = answer_code_lst[0]
                ans = tokenizer.decode(answer_code, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                
                passage_scores = crossattention_scores[k]
                
                top_m = 5
                score_array = passage_scores.cpu().numpy()
                top_passage_idxes = np.argpartition(-score_array, range(top_m))[:top_m]
                
                for threshold in [1, top_m]:
                    top_threshold_idxes = top_passage_idxes[:threshold]
                    if threshold == 1:
                        top_p_id = top_threshold_idxes[0]
                        write_preds(example['id'], top_p_id, example['ctxs'][top_p_id]['text']) 


def write_preds(qid, top_passage_idx, top_passage):
    out_item = {
        'qid':qid,
        'top_passage_id':int(top_passage_idx),
        'top_passage':top_passage
    }
    f_o_preds.write(json.dumps(out_item) + '\n')

def show_precision(count, table_pred_results):
    str_info = 'count = %d' % count
    for threshold in table_pred_results:
        precision = np.mean(table_pred_results[threshold]) * 100
        str_info += 'p@%d = %.2f ' % (threshold, precision)
    logger.info(str_info)

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)
   
    global f_o_preds 
    out_preds_file = os.path.join(opt.checkpoint_dir, opt.name, 'preds.jsonl')
    f_o_preds = open(out_preds_file, 'w')

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size,
        backward=opt.backward
    )
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context, 
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=0, 
        collate_fn=collator_function
    )
    
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("Start eval")
    evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)
    
    f_o_preds.close()

    #logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

