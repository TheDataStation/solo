import json
import csv
import argparse
import os
from tqdm import tqdm
from multiprocessing import Pool as ProcessPool
import uuid
import glob
import random

def get_out_file(args):
    data_dir = os.path.join(args.work_dir, 'data/%s/tables' % args.dataset)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    data_file = os.path.join(data_dir, 'tables.jsonl')
    return data_file

def read_meta(meta_file):
    if not os.path.exists(meta_file):
        return None
    with open(meta_file) as f:
        meta_data = json.load(f)
    return meta_data

def read_table(arg_info):
    csv_file = arg_info['data_file']
    meta_file = arg_info['meta_file']
    file_name = os.path.basename(os.path.splitext(csv_file)[0])
    table_title = '' 
    table_id = ''
    table_meta = read_meta(meta_file)
    if table_meta is not None:
        table_id = table_meta['table_id']
        table_title = table_meta['title']
    if table_title == '':
        if arg_info['file_name_title']:
            table_title = file_name
    if table_id == '':
        table_id = file_name + ' - ' + str(uuid.uuid4())
    table = {
        'columns':None,
        'rows':[],
        'tableId':table_id,
        'documentTitle':table_title,
    }
    col_name_lst = None
    with open(csv_file) as f:
        reader = csv.reader(f, delimiter=',')
        row_data = table['rows']
        for row, item in enumerate(reader):
            if row == 0:
                col_name_lst = item
            else:
                assert(len(item) == len(col_name_lst))
                cells = [{'text':a} for a in item]
                cell_info = {'cells':cells}
                row_data.append(cell_info)
    
    if table_meta is not None:
        meta_col_names = table_meta.get('col_names', None)
        if meta_col_names is not None:
            if(len(col_name_lst) == len(meta_col_names)):
                col_name_lst = meta_col_names
    
    table['columns'] = [{'text':col_name} for col_name in col_name_lst]
    
    return table

def main(args):
    out_file = get_out_file(args)
    if os.path.exists(out_file):
        msg_text = '(%s) already exists' % out_file
        msg_info = {
            'state':False,
            'msg':msg_text
        }
        return msg_info

    f_o = open(out_file, 'w')
    dataset_dir = os.path.join(args.work_dir, 'data', args.dataset)
    csv_file_pattern = os.path.join(dataset_dir, 'tables_csv', '**', '*.csv')
    csv_file_lst = glob.glob(csv_file_pattern, recursive=True)
   
    num_wokers = min(os.cpu_count(), 10) 
    work_pool = ProcessPool(num_wokers)
    arg_info_lst = []
    
    for csv_file in csv_file_lst:
        meta_file = os.path.splitext(csv_file)[0] + '.meta.json'
        args_info = {
            'data_file':csv_file,
            'meta_file':meta_file,
            'file_name_title':args.file_name_title
        }
        arg_info_lst.append(args_info)

    multi_process = True
    if multi_process:    
        for table in tqdm(work_pool.imap_unordered(read_table, arg_info_lst), total=len(arg_info_lst)):
            if table is None:
                continue
            output_table(table, args, f_o)
    else:
        for arg_info in tqdm(arg_info_lst):
            table = read_table(arg_info)
            if table is None:
                continue
            output_table(table, args, f_o)
     
    f_o.close()

    msg_info = {
        'state':True,
    }
    return msg_info

def output_table(table, args, f_o):
    if args.table_sample_rows is not None:
        row_data = table['rows']
        num_rows = len(row_data)
        table['rows'] = random.sample(row_data, min(num_rows, args.table_sample_rows))
    f_o.write(json.dumps(table) + '\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--file_name_title', type=int, default=1)
    parser.add_argument('--table_sample_rows', type=int, default=None)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    msg_info = main(args)
    if not msg_info['state']:
        print(msg_info['msg'])


