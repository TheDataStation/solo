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

def read_config():
    with open('./trainer.config') as f:
       config = json.load(f)
    return config

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
    model_path = os.path.join(work_dir, 'models/tqa_retriever')
    index_dir = os.path.join(work_dir, 'data', dataset, 'index/on_disk_index_%s_rel_graph' % dataset) 
    index_file = os.path.join(index_dir, 'populated.index')
    passage_file = os.path.join(index_dir, 'passages.jsonl')
    query_file = os.path.join(question_dir, 'fusion_query.jsonl')
    output_path = os.path.join(out_retr_dir, 'fusion_retrieved.jsonl')
    top_n = int(config['retr_top_n'])
    min_tables = int(config['min_tables'])
    max_retr = int(config['max_retr'])
    question_maxlength = int(config['question_maxlength'])
    retr_args = argparse.Namespace(model_path=model_path,
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

def get_train_args(train_itr, work_dir, dataset, retr_train_dir, retr_eval_dir, config):
    file_name = 'fusion_retrieved_tagged.jsonl'
    train_file = os.path.join(retr_train_dir, file_name)
    eval_file = os.path.join(retr_eval_dir, file_name)
    
    checkpoint_dir = os.path.join(work_dir, 'fusion_in_decoder/output')
    checkpoint_name = '%s_rel_graph_sql_data_train_%d' % (dataset, train_itr)
     
    train_args = argparse.Namespace(sql_batch_no=train_itr,
                                    do_train=True,
                                    model_path=os.path.join(work_dir, 'models/tqa_reader_base'),
                                    train_data=train_file,
                                    eval_data=eval_file,
                                    n_context=int(config['retr_top_n']),
                                    per_gpu_batch_size=int(config['train_batch_size']),
                                    cuda=0,
                                    name=checkpoint_name,
                                    checkpoint_dir=checkpoint_dir,
                                    max_epoch=int(config['max_epoch']),
                                    patience_steps=int(config['patience_steps']),
                                    text_maxlength=int(config['text_maxlength']),
                                    fusion_retr_model=None
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

def sql2question(mode, sql_dir, work_dir, dataset):
    print('translating %s sql to question' % mode)
    target_dir = os.path.join(work_dir, 'plms_graph2text/sql2nlg/data', dataset, 'sql_data')
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
         
    cmd = 'cd %s/plms_graph2text ;' % work_dir + \
          ' . ~/pyenv/plms_graph2text/bin/activate ;' + \
          ' ./decode_sql2nlg.sh t5-base %s/models/sql2nlg-t5-base_2022_01_21.ckpt' % work_dir + \
          ' 0 ' + dataset + ' sql_data ' + part_name
    os.system(cmd) 

    out_dir = os.path.join(work_dir, 'plms_graph2text/sql2nlg/outputs/test_model', 
                           dataset, 'sql_data', part_name)
    out_question_file = os.path.join(out_dir, 'val_outputs/test_unseen_predictions.txt.debug')

    count_sql = count_lines(sql_src_file)
    count_question = count_lines(out_question_file)
    assert(count_sql == count_question)
    
    sql_question_file = os.path.join(sql_dir, 'questions.txt')
    if os.path.exists(sql_question_file):
        err_msg = '(%s) already exists, do you want to replace it (y/n)? '
        raise ValueError(err_msg) 
    
    shutil.copy(out_question_file, sql_question_file)
    shutil.rmtree(out_dir)
   
    query_args = get_fusion_query_args(work_dir, dataset, sql_dir) 
    gen_fusion_query.main(query_args)
     
    #import pdb; pdb.set_trace()    
    #result = subprocess.check_output(cmd, shell=True, text=True)
    #print(result)  

def retr_triples(mode, work_dir, dataset, question_dir, table_dict, is_train, config):
    print('retrieving %s table triples' % mode)
    out_retr_dir = os.path.join(question_dir, 'rel_graph')
    os.mkdir(out_retr_dir)
    retr_args = get_retr_args(work_dir, dataset, question_dir, out_retr_dir, config) 
    passage_ondisk_retrieval.main(retr_args)
    
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
    top_n = int(config['retr_top_n'])
    min_tables = int(config['min_tables'])
    updated_retr_data = process_func(retr_data, top_n, table_dict, strategy, min_tables)
    out_file = os.path.join(out_retr_dir, 'fusion_retrieved_tagged.jsonl') 
    with open(out_file, 'w') as f:
        for item in tqdm(updated_retr_data):
            f.write(json.dumps(item) + '\n')

    os.remove(data_file) 

def read_tables(work_dir, dataset):
    table_file = os.path.join(work_dir, 'data', '%s/tables/tables.jsonl' % dataset)
    table_dict = {}
    with open(table_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['tableId']
            table_dict[table_id] = item
    return table_dict

def main():
    args = get_args()
    config = read_config()
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
    
    train_itr = -1
    while True:
        train_itr += 1
        train_config = read_config()
        num_train_queries = 0
        if train_itr <= 1:
            num_train_queries = int(train_config['train_start_n'])
        else:
            num_train_queries = int(train_config['train_step_n'])
        if num_train_queries <= 0:
            break

        mode = 'train_%d' % train_itr
        train_sql_dir = os.path.join(sql_data_dir, mode)
        table2sql.generate_queries(train_sql_dir, mode, train_tables, num_train_queries, stat_info, sql_dict) 
        
        sql2question(mode, train_sql_dir, args.work_dir, args.dataset) 
        retr_triples(mode, args.work_dir, args.dataset, train_sql_dir, table_dict, True, config)
        
        train_args = get_train_args(train_itr, args.work_dir, args.dataset, 
                                    os.path.join(train_sql_dir, 'rel_graph'), 
                                    os.path.join(dev_sql_dir, 'rel_graph'), 
                                    config)
        
        model_trainer.main(train_args)
        
        break

def train():
    args = get_args()
    config = read_config()
    train_itr = 0
    train_sql_dir = '/home/cc/code/open_table_discovery/table2question/dataset/nq_tables_example/sql_data/train_0'
    dev_sql_dir = '/home/cc/code/open_table_discovery/table2question/dataset/nq_tables_example/sql_data/dev'

    train_args = get_train_args(train_itr, args.work_dir, args.dataset, 
                                os.path.join(train_sql_dir, 'rel_graph'), 
                                os.path.join(dev_sql_dir, 'rel_graph'), 
                                config)
    
    msg_info = model_trainer.main(train_args)
    if not msg_info['state']:
        print(msg_info['msg'])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    #train()

