import json
import csv
import argparse
import os
from tqdm import tqdm

def read_small_table_set(data_file):
    table_lst = []
    with open(data_file) as f:
        for line in f:
            table_id = line.rstrip()
            table_lst.append(table_id)
    table_set = set(table_lst)
    return table_set

def read_table(args):
    data_file = '/home/cc/data/%s/tables/tables.jsonl' % args.dataset
    small_table_set = read_small_table_set()
    with open(data_file) as f:
        for line in f:
            table_data = json.loads(line)
            table_id = table_data['tableId']
            if table_id not in small_table_set:
                continue
            yield table_data

def main():
    args = get_args()
    table_id_dict = {}
    for table in tqdm(read_table()):
        table_id = table['tableId']
        if '/' in table_id:
            table_id = table_id.replace('/', '[left-slash]')

        out_file = os.path.join(args.output_dir, '%s.csv' % table_id)
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--input_tables', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()


