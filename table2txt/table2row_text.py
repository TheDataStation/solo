import json
from tqdm import tqdm
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--expr', type=str, required=True)
    args = parser.parse_args()
    return args

def read_tables(args):
    table_lst = []
    table_file = '/home/cc/data/%s/tables/tables.jsonl' % args.dataset
    with open(table_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_lst.append(item)
    return table_lst

def gen_row_text(table, col_name_lst, row_info):
    cell_lst = row_info['cells']
    out_text = table['documentTitle']
    for col, cell in enumerate(cell_lst):
        col_name = col_name_lst[col]
        cell_text = cell['text']
        out_text += ' , ' + col_name + ' ' + cell_text
    return out_text


def main():
    args = get_args()
    out_dir = os.path.join('dataset', args.dataset, args.expr)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, 'passages.jsonl')
    if os.path.exists(out_file):
        print('output file (%s) already exists.' % out_file)
        return

    table_lst = read_tables(args)
    passage_seq_id = 0
    with open(out_file, 'w') as f_o:
        for table in tqdm(table_lst):
             col_info_lst = table['columns']
             col_name_lst = [a['text'] for a in col_info_lst]
             row_info_lst = table['rows']
             for row, row_info in enumerate(row_info_lst):
                row_text = gen_row_text(table, col_name_lst, row_info)
                tag_info = {
                    'table_id':table['tableId'],
                    'row':row
                }
                passage_info = {
                    'p_id':passage_seq_id,
                    'passage':row_text,
                    'tag':tag_info
                }
                passage_seq_id += 1
                f_o.write(json.dumps(passage_info) + '\n')
                 

if __name__ == '__main__':
    main()
