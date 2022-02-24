import json
from tqdm import tqdm
import csv

dataset = 'nq_tables'

def read_tables():
    table_dict = {}
    data_file = f'./{dataset}/tables/tables.jsonl'
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['tableId']
            parts = table_id.split('_')
            assert(len(parts) == 2)
            table_id_title = parts[0]
            if table_id_title not in table_dict:
                table_dict[table_id_title] = []
            data_lst = table_dict[table_id_title]
            data_lst.append(item)

    return table_dict

def find_answer(answer, table):
    row_data = table['rows']
    for row_item in row_data:
        row_cells = row_item['cells']
        for cell in row_cells:
            if answer in cell['text']:
                return True
    return False

def process_data(mode, data_file, table_dict):
    out_csv_file = '%s_log.csv' % mode
    f_o_csv = open(out_csv_file, 'w')
    csv_writer = csv.writer(f_o_csv, delimiter=',')
    csv_writer.writerow(['qid', 'question', 'table_id_lst', 'answer', 'other_table_id'])

    qid_set = set()
    table_set = set()
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)  
            table_id_lst = item['table_id_lst']
            answers = item['answers']
           
            table_id_title_lst = [] 
            for table_id in table_id_lst:
                table_set.add(table_id)
                parts = table_id.split('_')
                table_id_title = parts[0]
                table_id_title_lst.append(table_id_title)
            
            table_id_title_set = set(table_id_title_lst)
            assert(len(table_id_title_set) == 1)
            
            table_id_title = list(table_id_title_set)[0]
            data_lst = table_dict[table_id_title]
            
            table_id_lst_text = ' \n '.join(table_id_lst)

            for data_table in data_lst:
                if data_table['tableId'] in table_id_lst:
                    continue
                for answer in answers:
                    if find_answer(answer, data_table):
                        qid_set.add(item['qid'])
                        table_set.add(data_table['tableId'])
                        csv_row = [item['qid'], item['question'], table_id_lst_text, answer, data_table['tableId']]
                        csv_writer.writerow(csv_row)
                         
    f_o_csv.close()
    print('questions %d ' % len(qid_set))

    ref_table_lst = list(table_set)
    with open('%s_table_id_lst.txt' % mode, 'w') as f_o:
        for table_id in ref_table_lst:
            f_o.write(table_id + '\n')

def main():
    table_dict = read_tables()
    process_data('dev', '/home/cc/data/nq_tables/interactions/dev_qas.jsonl', table_dict)

if __name__ == '__main__':
    main()
