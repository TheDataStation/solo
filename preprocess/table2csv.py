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
    data_file = '/home/cc/code/data/%s/tables/tables.jsonl' % args.dataset
    #small_table_set = read_small_table_set(args.input_tables)
    with open(data_file) as f:
        for line in f:
            table_data = json.loads(line)
            table_id = table_data['tableId']
            #if table_id not in small_table_set:
            #    continue
            yield table_data

def main():
    args = get_args()
    table_id_dict = {}
    table_seq_no = 0
    for table in tqdm(read_table(args)):
        table_dir = os.path.join(args.output_dir, 'table_%d' % table_seq_no)
        os.mkdir(table_dir)

        table_title = table['documentTitle']
        table_id = table['tableId']

        out_file = os.path.join(table_dir, '%d.csv' % table_seq_no)
        meta_file = os.path.join(table_dir, '%d.meta' % table_seq_no)
        with open(meta_file, 'w') as f_m:
            f_m.write("title=%s\n" % table_title)
            f_m.write("id=%s\n" % table_id)

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--input_tables', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()


