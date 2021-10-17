import json
import os
import random
import argparse
from table2txt.graph_strategy.strategy import Strategy
from table2txt.graph_strategy.complete_graph import CompleteGraph
import pytorch_lightning as pl
from webnlg.finetune import main as generate_text_from_graph, SummarizationModule

def read_tables(data_file):
    table_lst = []
    M = 3
    with open(data_file) as f:
        for line in f:
            table = json.loads(line)
            row_lst = table['rows']
            num_rows = len(row_lst)
            num_sample = min(M, num_rows)
            sample_rows = random.sample(row_lst, num_sample)
            table['rows'] = sample_rows 
            table_lst.append(table)
    return table_lst

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_tables', type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    return args

def get_strategy_lst():
    stg_lst = []
    stg_1 = CompleteGraph()
    stg_lst.append(stg_1)
    return stg_lst

def evaluate_strategy(args, stg):
    graph = stg.generate()
    generate_text_from_graph(args)  

def main():
    random.seed(10)
    args = get_args()
    table_lst = read_tables(args.input_tables)
    for table in table_lst:
        for stg in stg_lst:
            evaluate_strategy(args, stg)

if __name__ == '__main__':
    main()
