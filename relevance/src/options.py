# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=1000)
        self.parser.add_argument('--total_steps', type=int, default=1000)
        self.parser.add_argument('--scheduler_steps', type=int, default=None, 
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--fixed_lr', action='store_true')

    def add_eval_options(self):
        self.parser.add_argument('--write_results', action='store_true', help='save results')
        self.parser.add_argument('--write_crossattention_scores', action='store_true', 
                        help='save dataset with cross-attention scores')

    def add_reader_options(self):
        self.parser.add_argument('--train_data', type=str, default=None, help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default=None, help='path of eval data')
        self.parser.add_argument('--model_size', type=str, default='base')
        self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')
        self.parser.add_argument('--text_maxlength', type=int, default=300, 
                        help='maximum number of tokens in text segments (question+passage)')
        
        
        self.parser.add_argument('--question_maxlength', type=int, default=50, 
                        help='maximum number of tokens in questions')


        self.parser.add_argument('--answer_maxlength', type=int, default=-1, 
                        help='maximum number of tokens used to train the model, no truncation if -1')
        self.parser.add_argument('--no_title', action='store_true', 
                        help='article titles not included in passages')
        self.parser.add_argument('--n_context', type=int)

    def add_retriever_options(self):
        self.parser.add_argument('--train_data', type=str, default='none', help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default='none', help='path of eval data')
        self.parser.add_argument('--indexing_dimension', type=int, default=768)
        self.parser.add_argument('--no_projection', action='store_true', 
                        help='No addition Linear layer and layernorm, only works if indexing size equals 768')
        self.parser.add_argument('--question_maxlength', type=int, default=40, 
                        help='maximum number of tokens in questions')
        self.parser.add_argument('--passage_maxlength', type=int, default=200, 
                        help='maximum number of tokens in passages')
        self.parser.add_argument('--no_question_mask', action='store_true')
        self.parser.add_argument('--no_passage_mask', action='store_true')
        self.parser.add_argument('--extract_cls', action='store_true')
        self.parser.add_argument('--no_title', action='store_true', 
                        help='article titles not included in passages')
        self.parser.add_argument('--n_context', type=int, default=1)
        
        self.parser.add_argument('--teacher_model_path', type=str, default='none', help='path for teacher model')
        self.parser.add_argument('--teacher_precompute_file', type=str)
        self.parser.add_argument('--distill_temperature', type=float, default=1)
        self.parser.add_argument('--distill_weight', type=float, default=0.65)
        self.parser.add_argument('--is_student', type=int, default=True)

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--name', type=str, help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./output/', help='models are saved here')
        self.parser.add_argument('--model_path', type=str, default='none', help='path for retraining')
      
        self.parser.add_argument('--bnn', type=int, help='Bayesian NN') 
        self.parser.add_argument('--bnn_num_eval_sample', type=int, default=6)
        self.parser.add_argument('--fusion_retr_model', type=str) 
        self.parser.add_argument('--prior_model', type=str) 
        self.parser.add_argument('--max_epoch', type=int, default=30) 
        self.parser.add_argument('--ckp_steps', type=int) 
        self.parser.add_argument('--retr_model_type', type=str) 
        self.parser.add_argument('--do_train', action="store_true")
        self.parser.add_argument('--cuda', type=int, default=0)
        self.parser.add_argument('--train_qas_file', type=str)
        self.parser.add_argument('--eval_qas_file', type=str)
        self.parser.add_argument('--eval_in_train', type=int, default=1)
        self.parser.add_argument('--multi_model_eval', type=int, default=0)
        self.parser.add_argument('--multi_model_dir', type=str)

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
        self.parser.add_argument('--maxload', type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        # training parameters

        self.parser.add_argument('--sql_batch_no', type=int)
        self.parser.add_argument('--patience_steps', type=int, default=10)
        self.parser.add_argument('--eval_freq', type=int, default=1000)
        self.parser.add_argument('--save_freq', type=int, default=1000)

    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir)/ opt.name
        model_dir = expr_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(expr_dir/'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()
        return opt


def get_options(use_reader=False,
                use_retriever=False,
                use_optim=False,
                use_eval=False):
    options = Options()
    if use_reader:
        options.add_reader_options()
    if use_retriever:
        options.add_retriever_options()
    if use_optim:
        options.add_optim_options()
    if use_eval:
        options.add_eval_options()
    return options.parse()
