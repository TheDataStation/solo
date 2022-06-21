import argparse
import json
import os
from table2question import table2sql

def read_config():
    with open('./trainer.config') as f:
       config = json.load(f)
    return config

def get_sql_args(work_dir, dataset):
    sql_args = argparse.Namespace(work_dir=work_dir,
                                  dataset=dataset,
                                  table_file='tables.jsonl',
                                  experiment='sql_data',
                                  dev_table_pct=0.2,
                                  num_dev_queries=2000
                                 )
    return sql_args 
    

def main():
    args = get_args()
    sql_args = get_sql_args(args.work_dir, args.dataset)
    msg_info = table2sql.init_data(sql_args)
    if not msg_info['state']:
        print(msg_info['msg'])
        return

    sql_data_dir = msg_info['sql_data_dir']
    sql_dict = msg_info['sql_dict']
    train_tables = msg_info['train_tables']
    stat_info = msg_info['stat_info']
    train_itr = 0
   
    while True:
        train_itr += 1
        config = read_config()
        num_train_queries = 0
        if train_itr <= 1:
            num_train_queries = config['start_n']
        else:
            num_train_queries = config['step_n']
        if num_train_queries <= 0:
            break

        mode = 'train_%d' % train_itr
        train_sql_dir = os.path.join(sql_data_dir, mode)
        table2sql.generate_queries(train_sql_dir, mode, train_tables, num_train_queries, stat_info, sql_dict) 
        
        break

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

