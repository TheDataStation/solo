import json
import csv
from tqdm import tqdm

def get_tables():
    table_id_lst = []
    with open('/home/cc/data/nq/tables/tables_small.jsonl') as f:
        for line in f:
            table_id = line.rstrip()
            table_id_lst.append(table_id)

    table_id_set = set(table_id_lst)
    return table_id_set

def read_table_repr(table_id_set):
    data_file = '/home/cc/code/tapas/models/tapas_nq_hn_retriever_large/tables/predict_results_0.tsv'
    with open(data_file) as f:
        reader = csv.reader(f, delimiter='\t')
        for row, item in enumerate(reader):
            if row == 0:
                continue
            table_id = item[2]
            if table_id in table_id_set:
                yield item


def main():
    table_id_set = get_tables()
    out_file = '/home/cc/code/tapas/models/tapas_nq_hn_retriever_large/tables_small/predict_results_0.tsv'
    with open(out_file, 'w') as f_o:
        writer = csv.writer(f_o, delimiter='\t')
        columns = ['query_id', 'query_rep', 'table_id', 'table_rep']
        writer.writerow(columns)
        for item in tqdm(read_table_repr(table_id_set)):
            writer.writerow(item)

if __name__ == '__main__':
    main()
