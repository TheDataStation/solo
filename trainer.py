import argparse
import json
import os
import shutil
import subprocess
from tqdm import tqdm
import uuid
from table2question import table2sql, gen_fusion_query
import passage_ondisk_retrieval
from table2txt.retr_utils import process_train, process_dev
import finetune_table_retr as model_trainer
import datetime
import torch
from enum import Enum

class ConfirmOption(Enum):
    UseExisting = 1
    CreateNew = 2
    Exit = 3

def read_config():
    with open('./system.config') as f:
       config = json.load(f)
    return config

def get_sql_data_dir(work_dir, dataset):
    data_dir = os.path.join(work_dir, 'open_table_discovery/table2question/dataset', dataset, 'sql_data')
    return data_dir

def read_stat_info(data_dir):
    data_file = os.path.join(data_dir, 'stat_info.json')
    with open(data_file) as f:
        stat_info = json.load(f)
    return stat_info

def get_data_state(data_dir):
    state_file = os.path.join(data_dir, 'state.json')
    if not os.path.isfile(state_file):
        return None
    with open(state_file) as f:
        state_info = json.load(f)
    return state_info

def update_data_state(work_dir, dataset, data_itr, sql_dict):
    data_dir = get_sql_data_dir(work_dir, dataset)
    state_file = os.path.join(data_dir, 'state.json')
    state_info = {
        'data_ready':True,
        'data_itr':data_itr,
        'sql_num':len(sql_dict),
    }
    with open(state_file, 'w') as f_o:
        f_o.write(json.dumps(state_info))

    write_sql_dict(data_dir, sql_dict)

def write_sql_dict(out_dir, sql_dict):
    out_file = os.path.join(out_dir, 'sql_keys.jsonl')
    with open(out_file, 'w') as f_o:
        for sql_key in sql_dict:
            item = {
                'key':sql_key
            }
            f_o.write(json.dumps(item) + '\n')

def read_sql_dict(data_state, data_dir):
    sql_dict = {}
    data_file = os.path.join(data_dir, 'sql_keys.jsonl')
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line) 
            sql_dict[item['key']] = 1
    assert(data_state['sql_num'] == len(sql_dict))
    return sql_dict

def read_train_tables(data_dir, dataset):
    train_tables = []
    data_file = os.path.join(data_dir, 'train_tables.jsonl')
    train_table_dict = {}
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
        train_table_dict[item['table_id']] = 1
    
    table_file = '../data/%s/tables/tables.jsonl' % dataset
    table_dict = {}
    with open(table_file) as f:
        for line in f:
            item = json.loads(line)
            table_id = item['tableId']
            table_dict[table_id] = item
            if table_id in train_table_dict:
                train_tables.append(item)
    
    return train_tables, table_dict

def get_sql_args(work_dir, dataset, config):
    sql_args = argparse.Namespace(work_dir=work_dir,
                                  dataset=dataset,
                                  table_file='tables.jsonl',
                                  experiment='sql_data',
                                  dev_table_pct=float(config['dev_table_pct']),
                                  num_dev_queries=int(config['dev_n'])
                                 )
    return sql_args 

def get_fusion_query_args(work_dir, dataset, question_dir):
    query_args = argparse.Namespace(work_dir=work_dir,
                                    dataset=dataset,
                                    question_dir=question_dir
                                   )
    return query_args

def get_retr_args(work_dir, dataset, question_dir, out_retr_dir, config):
    student_model_path = os.path.join(work_dir, 'models/student_tqa_retriever_step_29500')
    teacher_model_path = os.path.join(work_dir, 'models/tqa_retriever')
    index_dir = os.path.join(work_dir, 'index/on_disk_index_%s_rel_graph' % dataset) 
    index_file = os.path.join(index_dir, 'populated.index')
    passage_file = os.path.join(index_dir, 'passages.jsonl')
    query_file = os.path.join(question_dir, 'fusion_query.jsonl')
    output_path = os.path.join(out_retr_dir, 'fusion_retrieved.jsonl')
    top_n = int(config['retr_top_n'])
    min_tables = int(config['min_tables'])
    max_retr = int(config['max_retr'])
    question_maxlength = int(config['question_maxlength'])
    retr_args = argparse.Namespace( 
                                    student_model_path=student_model_path,
                                    teacher_model_path=teacher_model_path,
                                    index_dir=index_dir,
                                    index_file=index_file,
                                    passage_file=passage_file,
                                    data=query_file,
                                    output_path=output_path,
                                    n_docs=top_n,
                                    min_tables=min_tables,
                                    max_retr=max_retr,
                                    question_maxlength=question_maxlength,
                                    no_fp16=False
                                   )
    return retr_args

def get_train_date_dir():
    a = datetime.datetime.now()
    train_dir = 'train_%d_%d_%d_%d_%d_%d_%d' % (a.year, a.month, a.day, a.hour, a.minute, a.second, a.microsecond) 
    return train_dir

def get_train_args(train_itr, work_dir, dataset, checkpoint_dir, 
                   retr_train_dir, retr_eval_dir, config, prior_model):
    file_name = 'fusion_retrieved_tagged.jsonl'
    train_file = os.path.join(retr_train_dir, file_name)
    eval_file = os.path.join(retr_eval_dir, file_name)
    
    checkpoint_name = 'train_%d' % (train_itr)
     
    train_args = argparse.Namespace(sql_batch_no=train_itr,
                                    do_train=True,
                                    model_path=os.path.join(work_dir, 'models/tqa_reader_base'),
                                    train_data=train_file,
                                    eval_data=eval_file,
                                    n_context=int(config['train_n_context']),
                                    per_gpu_batch_size=int(config['train_batch_size']),
                                    per_gpu_eval_batch_size=int(config['eval_batch_size']),
                                    cuda=0,
                                    debug=config['debug'],
                                    name=checkpoint_name,
                                    checkpoint_dir=checkpoint_dir,
                                    max_epoch=int(config['max_epoch']),
                                    patience_epochs=int(config['patience_epochs']),
                                    patience_datasets=int(config['patience_datasets']),
                                    bnn=int(config['bnn']),
                                    text_maxlength=int(config['text_maxlength']),
                                    multi_model_eval=0,
                                    prior_model=prior_model,
                                    fusion_retr_model=None,
                                    ) 
    return train_args

def count_lines(data_file):
    count = 0
    if not os.path.exists(data_file):
        return count
    with open(data_file) as f:
        for line in f:
            count += 1
    return count

def sql2question(mode, sql_dir, work_dir, dataset, train_itr=None):
    torch.cuda.empty_cache()
    if train_itr is None:
        print('translating %s sql to question' % mode)
    else:
        print('translating data %d sql to question' % train_itr)
    target_dir = os.path.join(work_dir, 'open_table_discovery', 'sql2question/sql2nlg/data', dataset, 'sql_data')
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    
    template_dir = os.path.join(work_dir, 'open_table_discovery/table2question/template')

    part_name = '%s_%s' % (mode, uuid.uuid4())
    part_dir = os.path.join(target_dir, part_name)
    if os.path.isdir(part_dir):
        shutil.rmtree(part_dir)
    shutil.copytree(template_dir, part_dir) 
    
    sql_src_file = os.path.join(sql_dir, 'test_unseen.source')
    part_src_file = os.path.join(part_dir, 'test_unseen.source')
    if os.path.exists(part_src_file):
        os.remove(part_src_file)
    shutil.copy(sql_src_file, part_dir)

    sql_tar_file = os.path.join(sql_dir, 'test_unseen.target')
    part_tar_file = os.path.join(part_dir, 'test_unseen.target')
    if os.path.exists(part_tar_file):
        os.remove(part_tar_file)
    shutil.copy(sql_tar_file, part_dir)
         
    cmd = 'cd %s/open_table_discovery/sql2question ;' % work_dir + \
          ' . %s/pyenv/sql2question/bin/activate ;' % work_dir + \
          ' ./decode_sql2nlg.sh t5-base %s/models/sql2nlg-t5-base_2022_01_21.ckpt' % work_dir + \
          ' 0 ' + dataset + ' sql_data ' + part_name
    os.system(cmd) 

    out_dir = os.path.join(work_dir, 'open_table_discovery', 'sql2question/sql2nlg/outputs/test_model', 
                           dataset, 'sql_data', part_name)
    out_question_file = os.path.join(out_dir, 'val_outputs/test_unseen_predictions.txt.debug')

    count_sql = count_lines(sql_src_file)
    count_question = count_lines(out_question_file)
    assert(count_sql == count_question)
    
    sql_question_file = os.path.join(sql_dir, 'questions.txt')
    if os.path.exists(sql_question_file):
        err_msg = '(%s) already exists, do you want to replace it (y/n)? ' % sql_question_file
        raise ValueError(err_msg) 
    
    shutil.copy(out_question_file, sql_question_file)
    shutil.rmtree(out_dir)
   
    query_args = get_fusion_query_args(work_dir, dataset, sql_dir) 
    gen_fusion_query.main(query_args)

def read_meta(meta_file):
    meta_map = {}
    with open(meta_file) as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            meta_map[qid] = item
    return meta_map
 
def get_sql_triples(question_dir, data_file):
    meta_file = os.path.join(question_dir, 'meta.txt')
    meta_map = read_meta(meta_file)
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
    return null 

def retr_triples(mode, work_dir, dataset, question_dir, table_dict, is_train, config, index_obj=None):
    print('retrieving %s table triples' % mode)
    out_retr_dir = os.path.join(question_dir, 'rel_graph')
    os.mkdir(out_retr_dir)
    retr_args = get_retr_args(work_dir, dataset, question_dir, out_retr_dir, config) 
    passage_ondisk_retrieval.main(retr_args, index_obj=index_obj)
    
    process_func = None
    if is_train:
        process_func = process_train
    else:
        process_func = process_dev
   
    retr_data = [] 
    data_file = os.path.join(out_retr_dir, 'fusion_retrieved.jsonl') 
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            retr_data.append(item)

    strategy = 'rel_graph'
    rerank_top_n = int(config['rerank_retr_top_n'])
    updated_retr_data = process_func(retr_data, rerank_top_n, table_dict, strategy)
    out_file = os.path.join(out_retr_dir, 'fusion_retrieved_tagged.jsonl') 
    with open(out_file, 'w') as f:
        for item in tqdm(updated_retr_data):
            f.write(json.dumps(item) + '\n')

def read_tables(work_dir, dataset):
    table_file = os.path.join(work_dir, 'data', '%s/tables/tables.jsonl' % dataset)
    table_dict = {}
    with open(table_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['tableId']
            table_dict[table_id] = item
    return table_dict

def get_train_itr_desp(data_itr):
    if data_itr == 0:
        return '(data 0)'
    else:
        return '(data 0-%d)' % data_itr

def confirm(args):
    data_dir = get_sql_data_dir(args.work_dir, args.dataset)
    data_state = get_data_state(data_dir)
    data_ready = False
    if (data_state is not None) and data_state['data_ready']:
        data_ready = True
    opt = None
    if data_ready:
        str_msg = 'Training %s already exists, type \n' % get_train_itr_desp(data_state['data_itr']) + \
                  '1 - Continue with existing data \n' + \
                  '2 - train with new data (existing training data will be deleted) \n' + \
                  'q - exit\n' 
        opt_str = input(str_msg)
        if opt_str == '1':
            opt = ConfirmOption.UseExisting
        elif opt_str == '2':
            if os.path.isdir(data_dir):
                shutil.rmtree(data_dir)
            data_state = None
            opt = ConfirmOption.CreateNew
        elif opt_str == 'q':
            opt = ConfirmOption.Exit
    else:
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
        opt = ConfirmOption.CreateNew
                 
    return opt, data_state

def remove_train_data_dir(train_sql_dir):
    if os.path.isdir(train_sql_dir):
        shutil.rmtree(train_sql_dir) 

def main():
    args = get_args()
    con_opt, data_state = confirm(args)
    while con_opt is None:
        print('type 1, 2 or q')
        con_opt = confirm(args)
    if con_opt == ConfirmOption.Exit:
        return
    config = read_config()
    sql_dict = None
    stat_info = None
    train_tables = None
    table_dict = None
    if con_opt == ConfirmOption.CreateNew: 
        sql_args = get_sql_args(args.work_dir, args.dataset, config)
        msg_info = table2sql.init_data(sql_args)
        if not msg_info['state']:
            print(msg_info['msg'])
            return

        sql_data_dir = msg_info['sql_data_dir']
        sql_dict = msg_info['sql_dict']
        train_tables = msg_info['train_tables']
        stat_info = msg_info['stat_info']
       
        table_dict = read_tables(args.work_dir, args.dataset)
         
        dev_sql_dir = os.path.join(sql_data_dir, 'dev')
        sql2question('dev', dev_sql_dir, args.work_dir, args.dataset)

        top_n = int(config['retr_top_n'])
        min_tables = int(config['min_tables'])
        max_retr = int(config['max_retr'])
        retr_triples('dev', args.work_dir, args.dataset, dev_sql_dir, table_dict, False, config)
    else:
        sql_data_dir = get_sql_data_dir(args.work_dir, args.dataset)
        dev_sql_dir = os.path.join(sql_data_dir, 'dev')

    checkpoint_dir = os.path.join(args.work_dir, 'open_table_discovery/output', args.dataset, get_train_date_dir())
    assert(not os.path.isdir(checkpoint_dir))
      
    train_file_lst = [] 
    prior_model = None
    best_metric = None
    
    existing_data_itr = -1
    if data_state != None:
        existing_data_itr = data_state['data_itr'] 
    train_itr = -1
    while True:
        train_itr += 1
        num_train_queries = int(config['train_incr_size'])
        assert(num_train_queries > 0)
         
        mode = 'train_%d' % train_itr
        train_sql_dir = os.path.join(sql_data_dir, mode)
        if train_itr > existing_data_itr: # need to create more questions
            if sql_dict is None:
                sql_dict = read_sql_dict(data_state, sql_data_dir)
                train_tables, table_dict = read_train_tables(sql_data_dir, args.dataset)         
                stat_info = read_stat_info(sql_data_dir)
                table2sql.init_worker()
            
            remove_train_data_dir(train_sql_dir)

            table2sql.generate_queries(train_sql_dir, mode, train_tables, num_train_queries, stat_info, sql_dict) 
            
            sql2question(mode, train_sql_dir, args.work_dir, args.dataset, train_itr) 
            
            retr_triples(mode, args.work_dir, args.dataset, train_sql_dir, table_dict, True, config)
            
            update_data_state(args.work_dir, args.dataset, train_itr, sql_dict)
      
        if best_metric is None:
            prior_model = None
        else:
            prior_model = best_metric['model_file']
        train_args = get_train_args(train_itr, args.work_dir, args.dataset, checkpoint_dir, 
                                    os.path.join(train_sql_dir, 'rel_graph'), 
                                    os.path.join(dev_sql_dir, 'rel_graph'), 
                                    config, prior_model)
        if not os.path.isfile(train_args.train_data):
            print('No train data file (%s)' % train_args.train_data)
            break
        msg_info = model_trainer.main(train_args)
        torch.cuda.empty_cache()
        if not msg_info['state']:
            print(msg_info['msg'])
            break 
       
        train_metric = msg_info['best_metric'] 
        if best_metric is None:
            best_metric = train_metric
            best_metric['train_itr'] = train_itr
            best_metric['patience_itr'] = 0
        else:
            update_best_metric(best_metric, train_metric, train_itr)

        if best_metric['patience_itr'] > config['patience_datasets']:
            break

    show_best_metric(train_args.checkpoint_dir, best_metric, args.work_dir, args.dataset)

def show_best_metric(checkpoint_dir, best_metric, work_dir, dataset):
    p_at_1 = best_metric['p@1'] * 100 / best_metric['N']
    p_at_5 = best_metric['p@5'] * 100 / best_metric['N']
    model_file = best_metric['model_file']

    best_model_dir = os.path.join(checkpoint_dir, 'best_model')
    os.mkdir(best_model_dir)
    metric_info = {'p@1':p_at_1, 'p@5':p_at_5}
    metric_file = os.path.join(best_model_dir, 'metric.json')
    with open(metric_file, 'w') as f_o:
        f_o.write(json.dumps(metric_info))
   
    model_base_name = os.path.basename(best_metric['model_file']) 
    best_model_file = os.path.join(best_model_dir, model_base_name)
    assert(not os.path.isfile(best_model_file)) 
    shutil.copy(best_metric['model_file'], best_model_file)

    deploy_dir = os.path.join(work_dir, 'models', dataset)
    if not os.path.isdir(deploy_dir):
        os.makedirs(deploy_dir)

    deploy_file = os.path.join(deploy_dir, model_base_name)
    if os.path.isfile(deploy_file):
        updated_model_base_name = ('%s_' % str(uuid.uuid4())) + model_base_name
        deploy_file = os.path.join(deploy_dir, updated_model_base_name)
    shutil.copy(best_metric['model_file'], deploy_file) 
    
    print('Evaluation P@1=%.2f P@5=%.2f' % (p_at_1, p_at_5))
    print('Best model %s ' % deploy_file)

def update_best_metric(best_metric, train_metric, train_itr):
    best_metric['patience_itr'] += 1 
    if train_metric['p@1'] > best_metric['p@1']:
        best_metric['p@1'] = train_metric['p@1']
        best_metric['p@5'] = train_metric['p@5']
        best_metric['model_file'] = train_metric['model_file']
        best_metric['train_itr'] = train_itr
        best_metric['patience_itr'] = 0

    elif train_metric['p@1'] == best_metric['p@1']:
        if train_metric['p@5'] > best_metric['p@5']:
            best_metric['p@5'] = train_metric['p@5']
            best_metric['model_file'] = train_metric['model_file']
            best_metric['train_itr'] = train_itr     
            best_metric['patience_itr'] = 0        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    #train()

