# take input from GTR result, https://github.com/FeiWang96/GTR/blob/main/data/webquerytable/preprocess.py
import json
from tqdm import tqdm

def get_table_title(caption, sub_caption):
    title = caption
    if sub_caption != '':
        title += ' , ' + sub_caption
    return title


def read_query_rels(query_rels_file):
    query_rel_dict = {}
    with open(query_rels_file) as f:
        for line in f:
            item_info = line.strip()
            data = item_info.split('\t')
            qid = data[0]
            table_id = data[2]
            label = int(data[3])
            if qid not in query_rel_dict:
                query_rel_dict[qid] = {'table_id_lst':[]}
            table_id_lst = query_rel_dict[qid]['table_id_lst']
            if label == 1:
                table_id_lst.append(table_id)
    return query_rel_dict


def get_revised_question(question):
    updated_question = question + ' is what ?'
    return updated_question


def process_query(query_file, query_rel_dict, out_file):
    f_o = open(out_file, 'w')
    with open(query_file) as f:
        for line in tqdm(f):
            item_info = line.strip()
            data = item_info.split('\t')
            qid = data[0]
            question = data[1]
            revised_question = get_revised_question(question)
            out_item = {
                'qid': qid,
                'question': revised_question,
                'table_id_lst': query_rel_dict[qid]['table_id_lst'],
                'answers': ['']
            }
            f_o.write(json.dumps(out_item) + '\n')
    f_o.close()


def process_tables(table_file, out_file):
    f_o = open(out_file, 'w')
    with open(table_file) as f:
        table_data = json.load(f)
    
    for table_id in tqdm(table_data):
        item_info = table_data[table_id]
        caption = item_info['caption']
        sub_caption = item_info['subcaption']
        title = get_table_title(caption, sub_caption)
        content = item_info['table_array']
        
        out_item_info = {}
        columns = content[0]
        out_item_info['columns'] = [{'text': col_name} for col_name in columns]  
        out_row_data = []

        for idx, content_item in enumerate(content):
            if idx == 0:
                continue
            out_cells = [{'text': cell_text} for cell_text in content_item]
            out_row_item = {'cells': out_cells}
            out_row_data.append(out_row_item)

        out_item_info['rows'] = out_row_data 
        out_item_info['tableId'] = table_id
        out_item_info['documentTitle'] = title 
        f_o.write(json.dumps(out_item_info)) 
     
    f_o.close()


def main():
    table_file = '/home/cc/code/GTR/data/webquerytable/tables.json'
    out_file = '/home/cc/data/webquerytable/tables/tables.jsonl'
    process_tables(table_file, out_file)

    query_rels_file = '/home/cc/code/GTR/data/webquerytable/qtrels.txt'
    query_rel_dict = read_query_rels(query_rels_file)

    dev_query_file = '/home/cc/code/GTR/data/webquerytable/dev_query.txt'
    dev_out_file = '/home/cc/data/webquerytable/interactions/dev_qas.jsonl'
    process_query(dev_query_file, query_rel_dict, dev_out_file)

    test_query_file = '/home/cc/code/GTR/data/webquerytable/test_query.txt'
    test_out_file = '/home/cc/data/webquerytable/interactions/test_qas.jsonl'
    process_query(test_query_file, query_rel_dict, test_out_file)


if __name__ == '__main__':
    main()



