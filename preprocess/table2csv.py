import json
import csv
import argparse
import os
from tqdm import tqdm
import random

def read_small_table_set(data_file):
    if data_file is None:
        return None
    table_lst = []
    with open(data_file) as f:
        for line in f:
            table_id = line.rstrip()
            table_lst.append(table_id)
    table_set = set(table_lst)
    return table_set

def read_table(args):
    data_file = '../data/%s/tables/tables.jsonl' % args.dataset
    small_table_set = read_small_table_set(args.input_tables)
    
    table_data_lst = []
    with open(data_file) as f:
        for line in f:
            table_data = json.loads(line)
            table_id = table_data['tableId']
            if (small_table_set is not None) and (table_id not in small_table_set):
                continue
            table_data_lst.append(table_data)
    out_table_list = table_data_lst
    if args.sample_size is not None:
        table_data_sample = random.sample(table_data_lst, args.sample_size)
        out_table_list = table_data_sample
    return out_table_list 

def main():
    args = get_args()
    table_id_dict = {}
    table_seq_no = 0
    table_seq_no = 0

    table_dir = os.path.join(args.output_dir)
    if not os.path.isdir(table_dir):
        os.makedirs(table_dir)
    else:
        if len(os.listdir(table_dir)) > 0:
            print(f'out dir {table_dir} not empty')
            return
    all_out_files = []
    for table in tqdm(read_table(args)):
        table_seq_no += 1
        table_title = table['documentTitle']
        table_id = table['tableId']
        title_processed = table_title.replace('/', ' ')
        pos = table_id.rindex('-')
        updated_table_id = table_id[pos+1:].strip()
        file_name = f'{title_processed} - ID({updated_table_id}).csv'.strip()
        file_name = update_file_name(file_name)
        out_file = os.path.join(table_dir, file_name)
        all_out_files.append(file_name)
        table_seq_no += 1
        with open(out_file, 'w') as f_o:
            columns = table['columns']
            writer = csv.writer(f_o)
            col_names = [col_info['text'] for col_info in columns]
            writer.writerow(col_names)
            row_data = table['rows']
            for row_item in row_data:
                cells = row_item['cells']
                cell_values = [a['text'] for a in cells] 
                writer.writerow(cell_values)

    for out_file in all_out_files:
        if out_file[0] == '.':
            print(out_file)

def update_file_name(file_name):
    pos = 0
    while pos < len(file_name):
        if file_name[pos] == '.':
            pos += 1
        else:
            break
    return file_name[pos:]  

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--input_tables', type=str)
    parser.add_argument('--sample_size', type=int)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()


