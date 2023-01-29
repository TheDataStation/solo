from flask import Flask, request, render_template
import os
import random
import json
from tqdm import tqdm
import argparse
import tester
import uuid
import shutil

app = Flask(__name__)
app.template_folder = ''
work_dir = None
data_dir = None
table_dict = {}
dataset = None
example_questions = None
index_obj = None

@app.route('/', methods=('GET', 'POST'))
def index():
    table_data_lst = []
    file_name = 'table.html'
    if request.method == 'POST':
        #import pdb; pdb.set_trace()
        input_data = request.get_json()
        table_data_lst = query(input_data)
        file_name = 'top_tables.html'
    
    output = render_template(file_name,
                           example_question_lst=example_questions,
                           table_data_lst=table_data_lst
                          )
    return output

def create_query_file(qry_data_dir, question):
    file_name = 'fusion_query.jsonl'
    data_file = os.path.join(qry_data_dir, file_name)
    query_data = {
        "id":0,
        "question":question,
        "table_id_lst":['N/A'],
        "answers":["N/A"],
        "ctxs": [{"title": "", "text": "This is a example passage."}]
    }
    with open(data_file, 'w') as f_o:
        f_o.write(json.dumps(query_data) + '\n')

def get_top_tables(out_dir):
    data_file = os.path.join(out_dir, 'pred_epoch_0_None.jsonl')
    with open(data_file) as f:
        item = json.load(f)
    out_table_lst = []
    out_table_set = set()
    tag_lst = item['tags']
    for tag in tag_lst:
        table_id = tag['table_id']
        if table_id not in out_table_set:
            out_table_set.add(table_id)
            out_table_lst.append(table_id)
            if len(out_table_lst) >= 5:
                break
    print(out_table_lst)
    return out_table_lst

def query(input_data):
    #import pdb; pdb.set_trace()
    question = input_data['question']
    qry_folder = 'demo_query_%s' % str(uuid.uuid4())
    query_dir = os.path.join(data_dir, dataset, qry_folder)
    test_args = create_test_args(qry_folder)
    qry_data_dir = os.path.join(query_dir, 'test')
    if not os.path.isdir(qry_data_dir):
        os.makedirs(qry_data_dir)
    create_query_file(qry_data_dir, question)
    out_dir = tester.main(test_args, table_data=table_dict, index_obj=index_obj)
    top_table_lst = get_top_tables(out_dir)
    table_data_lst = [table_dict[table_id] for table_id in top_table_lst]
    rank = 0
    for table_data in table_data_lst:
        rank += 1
        table_data['search_rank'] = rank
        
    shutil.rmtree(query_dir)
    shutil.rmtree(out_dir)
     
    return table_data_lst

def load_tables():
    data_file = os.path.join(data_dir, dataset, 'tables/tables.jsonl')
    with open(data_file) as f:
        for line in tqdm(f):
            table_data = json.loads(line)
            table_id = table_data['tableId']
            table_dict[table_id] = table_data

def load_index(args):
    args.table_repre = 'rel_graph'
    global index_obj
    index_obj = tester.get_index_obj(args.work_dir, dataset, args)

def load_example_questions():
    question_lst = []
    data_file = os.path.join(data_dir, dataset, 'query/test/fusion_query.jsonl')
    with open(data_file) as f:
        for line in tqdm(f):
            q_info = json.loads(line)
            question = q_info['question']
            question_lst.append(question)
    
    global example_questions
    example_questions = random.sample(question_lst, 100)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    return args 

def create_test_args(query_dir):
    test_args = argparse.Namespace(
        work_dir=work_dir,
        dataset=dataset,
        query_dir=query_dir,
        table_repre='rel_graph',
        train_model_dir=None,
        bnn=1,
    )
    return test_args

def app_init():
    print('app_init')
    random.seed(0)
    global work_dir
    global data_dir
    global dataset
    args = get_args()
    work_dir = args.work_dir
    data_dir = os.path.join(work_dir, 'data')
    dataset = args.dataset

    load_tables()
    load_example_questions()
    load_index(args)

if __name__ == '__main__':
    app_init()
    app.run(debug=True, use_reloader=False)
