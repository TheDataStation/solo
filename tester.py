import os
import argparse
import datetime
import glob
import shutil
import json
import finetune_table_retr as model_tester
from trainer import read_config, read_tables, retr_triples, get_train_date_dir
from src.ondisk_index import OndiskIndexer

def main(args, table_data=None, index_obj=None):
    config = read_config()
    test_query_dir = os.path.join(args.work_dir, 'data', args.dataset, args.query_dir, 'test')
    retr_test_dir = os.path.join(test_query_dir, 'rel_graph')
    retr_file = os.path.join(retr_test_dir, 'fusion_retrieved_tagged.jsonl')
    con_opt = '2' # retrieve new top tables by default
    if (table_data is None) and os.path.isfile(retr_file):
        str_msg = 'A set of top tables are already retrieved from Index. \n' + \
                  'If the Index does not change, and also trainer.config does not change, ' + \
                  'you can choose option 1 which takes less time. \n' + \
                  '1 - use existing top tables \n' + \
                  '2 - retrieve new top tables \n' + \
                  'q - exit \n'
        
        con_opt = input(str_msg)
        while (con_opt not in ['1', '2', 'q']):
            print('type 1, 2 or q')
            con_opt = input(str_msg)
        if con_opt == 'q':
            return
         
    if table_data is None: 
        table_dict = read_tables(args.work_dir, args.dataset)
    else:
        table_dict = table_data
    if con_opt == '2':
        if os.path.isdir(retr_test_dir):
            shutil.rmtree(retr_test_dir)
        retr_triples('test', args.work_dir, args.dataset, test_query_dir, table_dict, False, config, index_obj=index_obj)
    test_args = get_test_args(args.work_dir, args.dataset, retr_test_dir, config, args)
    msg_info = model_tester.main(test_args)
    
    if not msg_info['state']:
        print(msg_info['msg'])
        return None
    return msg_info['out_dir'] 

def get_index_obj(work_dir, dataset):
    index_dir = os.path.join(work_dir, 'index/on_disk_index_%s_rel_graph' % dataset)
    index_file = os.path.join(index_dir, 'populated.index')
    passage_file = os.path.join(index_dir, 'passages.jsonl')

    index = OndiskIndexer(index_file, passage_file)
    return index

def get_date_dir(train_model_dir):
    test_dir = 'test_' + os.path.basename(train_model_dir)
    return test_dir

def get_model_file(file_pattern):
        file_lst = glob.glob(file_pattern)
        if len(file_lst) == 0:
            err_msg = 'There is no model file in (%s)' % file_pattern
            raise ValueError(err_msg)
        file_lst.sort(key=os.path.getmtime)
        recent_file = file_lst[-1]
        print('loading recent model file (%s)' % recent_file) 
        return recent_file

def get_train_best_model(train_model_dir):
    best_metric_file = os.path.join(train_model_dir, 'best_metric_info.json')
    with open(best_metric_file) as f:
        best_metric_info = json.load(f)
    model_file = best_metric_info['model_file']
    base_file = os.path.basename(model_file)
    model_file = os.path.join(train_model_dir, base_file)
    print('loading model file (%s)' % model_file) 
    return model_file 

def get_test_args(work_dir, dataset, retr_test_dir, config, args):
    file_name = 'fusion_retrieved_tagged.jsonl'
    eval_file = os.path.join(retr_test_dir, file_name)
    checkpoint_dir = os.path.join(work_dir, 'open_table_discovery/output', dataset)
    checkpoint_name = get_date_dir(args.train_model_dir)

    if args.train_model_dir is None: 
        ret_model_file_pattern = os.path.join(work_dir, 'models', dataset, '*.pt') 
        retr_model = get_model_file(ret_model_file_pattern) 
    else:
        retr_model = get_train_best_model(args.train_model_dir)     
    test_args = argparse.Namespace(sql_batch_no=None,
                                    do_train=False,
                                    model_path=os.path.join(work_dir, 'models/tqa_reader_base'),
                                    fusion_retr_model=retr_model,
                                    train_data=None,
                                    eval_data=eval_file,
                                    n_context=int(config['retr_top_n']),
                                    per_gpu_batch_size=1,
                                    per_gpu_eval_batch_size=1,
                                    cuda=0,
                                    name=checkpoint_name,
                                    checkpoint_dir=checkpoint_dir,
                                    bnn=args.bnn,
                                    prior_model=None,
                                    text_maxlength=int(config['text_maxlength']),
                                    bnn_num_eval_sample=15,
                                    multi_model_eval=0
                                    ) 
    return test_args 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--query_dir', type=str, default='query')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--train_model_dir', type=str, required=True)
    parser.add_argument('--bnn', type=int, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)

