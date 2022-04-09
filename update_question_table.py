import json
from tqdm import tqdm
import csv
from fabric_qa.utils import f1_score

def read_tables(dataset):
    table_dict = {} 
    table_title_dict = {}
    data_file = '/home/cc/data/%s/tables/tables.jsonl' % dataset
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['tableId']
            table_dict[table_id] = item
            table_title = get_table_title(item) 
            if table_title not in table_title_dict:
                table_title_dict[table_title] = []
            data_lst = table_title_dict[table_title]
            data_lst.append(item)

    return (table_dict, table_title_dict)


def find_gold_answer_cells(answer, table):
    answer_cells = []
    row_data = table['rows']
    col_data = table['columns']
    for row_item in row_data:
        row_cells = row_item['cells']
        for col, cell in enumerate(row_cells):
            if answer in cell['text']:
                cell_text = cell['text'].strip()
                col_text = col_data[col]['text'].strip()
                cell_col_text = 'col_name:' + col_text + ' cell_text:' + cell_text
                answer_cells.append(cell_col_text)  
    return answer_cells 


def text_matched(gold_answer, other_answer):
    score = f1_score(other_answer, gold_answer)
    return score >= 0.9


def find_answer(answer, gold_table_lst, other_table):
    debug_mode = False
    if debug_mode:
        gold_table_id_lst = [a['tableId'] for a in gold_table_lst]
        if answer == 'Russia (' and '2014 Winter Olympics medal table_84AAFABE2421A55B' in gold_table_id_lst:
            if other_table['tableId'] == '2014 Winter Olympics medal table_4C1729274FC23617':
                import pdb; pdb.set_trace()
                print('debug')
    
    other_answer_cells = find_gold_answer_cells(answer, other_table)
    for gold_table in gold_table_lst:
        gold_anser_cells = find_gold_answer_cells(answer, gold_table)
        for gold_answer in gold_anser_cells:
            for other_answer in other_answer_cells:
                if text_matched(gold_answer, other_answer):
                    return True 

    return False

def get_table_title(table):
    title = table['documentTitle'].strip().lower()
    return title

def process_data(dataset, mode, data_file, table_dict, table_title_dict):
    out_csv_file = '%s_%s_log.csv' % (dataset, mode)
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
           
            table_title_lst = [] 
            for table_id in table_id_lst:
                table_set.add(table_id)
                table_title = get_table_title(table_dict[table_id])
                table_title_lst.append(table_title)
            
            table_title_set = set(table_title_lst)
            assert(len(table_title_set) == 1)
            
            table_title = list(table_title_set)[0]
            data_lst = table_title_dict[table_title]
            
            table_id_lst_text = ' \n '.join(table_id_lst)

            gold_table_lst = [table_dict[a] for a in table_id_lst]
            for data_table in data_lst:
                if data_table['tableId'] in table_id_lst:
                    continue
                for answer in answers:
                    if find_answer(answer, gold_table_lst, data_table):
                        qid_set.add(item['qid'])
                        table_set.add(data_table['tableId'])
                        csv_row = [item['qid'], item['question'], table_id_lst_text, answer, data_table['tableId']]
                        csv_writer.writerow(csv_row)
                         
    f_o_csv.close()
    print('questions %d ' % len(qid_set))

    ref_table_lst = list(table_set)
    with open('%s_%s_table_id_lst.txt' % (dataset, mode), 'w') as f_o:
        for table_id in ref_table_lst:
            f_o.write(table_id + '\n')

def main():
    dataset = 'nq_tables'
    mode = 'dev'
    table_dict, table_title_dict = read_tables(dataset)
    data_file = '/home/cc/data/%s/interactions/%s_qas.jsonl' % (dataset, mode)
    process_data(dataset, mode, data_file, table_dict,  table_title_dict)

def get_labeled_tables():
    labeled_data = {} 
    with open('./test_other_table.csv') as f:
        reader = csv.reader(f, delimiter=',')     
        for row, item in enumerate(reader):
            if row == 0:
                continue
            qid = item[0]
            label = item[-1]
            if label != 'Y':
                continue
            other_table_id = item[-2]
            if qid not in labeled_data:
                labeled_data[qid] = []
            
            table_id_lst = labeled_data[qid]
            table_id_lst.append(other_table_id)  
    return labeled_data

def read_retr_data():
    data_file = '/home/cc/code/open_table_discovery/table2txt/dataset/nq_tables/rel_graph/fusion_retrieved_test.jsonl'
    retr_data = {}
    item_data = []
    with open(data_file) as f:
        for line in tqdm(f):
             item = json.loads(line)
             qid = item['id']
             retr_data[qid] = item
             item_data.append(item)
    return retr_data, item_data 
    
def update_retr_data():
    out_file = '/home/cc/code/open_table_discovery/table2txt/dataset/nq_tables/rel_graph/fusion_retrieved_test.jsonl_other_table'
    f_o = open(out_file, 'w')
    labeled_data = get_labeled_tables()
    table_dict, _ = read_tables('nq_tables')
    retr_data, item_data = read_retr_data()
    for qid in labeled_data:
        ret_item = retr_data[qid]
        other_table_id_lst = labeled_data[qid]
        gold_table_id_lst = ret_item['table_id_lst']
        for other_table_id in other_table_id_lst:
            assert(other_table_id in table_dict)
            if other_table_id not in gold_table_id_lst:
                gold_table_id_lst.append(other_table_id)
            
    for item in item_data: 
        f_o.write(json.dumps(item) + '\n')
    f_o.close()

if __name__ == '__main__':
    main()
    #update_retr_data()


