import json
import csv
import argparse
import os
from tqdm import tqdm
from multiprocessing import Pool as ProcessPool
import uuid
import glob

def get_out_file(args):
    data_dir = os.path.join(args.work_dir, 'data/%s/tables' % args.dataset)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    data_file = os.path.join(data_dir, 'tables.jsonl')
    return data_file

def read_meta(meta_file):
    table_title = ''
    table_id = ''
    title_prefix = 'title='
    id_prefix = 'id='
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            for line in f:
                text = line.strip()
                if text.startswith(title_prefix):
                    pos = len(title_prefix)
                    table_title = text[pos:]
                elif text.startswith(id_prefix):
                    pos = len(id_prefix)
                    table_id = text[pos:]
    return (table_title, table_id)

def read_table(file_info, args):
    csv_file = file_info['data_file']
    meta_file = file_info['meta_file']
    file_name = os.path.basename(os.path.splitext(csv_file)[0])
    table_title, table_id = read_meta(meta_file)
    if table_title == '':
        if args.file_name_title:
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
                table['columns'] = [{'text':col_name} for col_name in col_name_lst]
            else:
                assert(len(item) == len(col_name_lst))
                cells = [{'text':a} for a in item]
                cell_info = {'cells':cells}
                row_data.append(cell_info)
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
        meta_file = os.path.splitext(csv_file)[0] + '.meta'
        args_info = {
            'data_file':csv_file,
            'meta_file':meta_file,
        }
        arg_info_lst.append(args_info)
    
    for table in tqdm(work_pool.imap_unordered(read_table, arg_info_lst), total=len(arg_info_lst)):
        f_o.write(json.dumps(table) + '\n')
    
    f_o.close()

    msg_info = {
        'state':True,
    }
    return msg_info

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--file_name_title', type=int, default=1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    msg_info = main(args)
    if not msg_info['state']:
        print(msg_info['msg'])


