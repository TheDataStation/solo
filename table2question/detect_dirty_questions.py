from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import argparse
from IPython.display import display
import json
from tqdm import tqdm

def read_tables(args):
    print('loading tables')
    table_map = {}
    data_file = '%s/data/%s/tables/tables.jsonl' % (args.work_dir, args.dataset)
    with open(data_file) as f:
        for line in tqdm(f):
            table_data = json.loads(line)
            table_id = table_data['tableId'] 
            columns = table_data['columns']
            df_cols = [a['text'] for a in columns]
            rows = table_data['rows']
            df_data = []
            for row_info in rows:
                cell_info = row_info['cells']
                df_cells = [a['text'] for a in cell_info] 
                df_data.append(df_cells)
            
            df_table = pd.DataFrame(df_data, columns=df_cols) 
            table_map[table_id] = {'data':table_data, 'df':df_table}
    return table_map

def read_meta(args):
    print('loading meta')
    meta_data = []
    meta_file = './dataset/%s/%s/meta.txt' % (args.dataset, args.sql_expr) 
    with open(meta_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            meta_data.append(item)
    return meta_data

def read_query(args):
    print('loading queries')
    query_data = []
    query_file = './dataset/%s/%s/fusion_query.jsonl' % (args.dataset, args.sql_expr)
    with open(query_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            query_data.append(item)
    return query_data

def load_model():
    print('loading table QA model')
    model_name = "google/tapas-large-finetuned-wikisql-supervised"
    model = TapasForQuestionAnswering.from_pretrained(model_name)
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    return model, tokenizer 

def main():
    args = get_args()
   
    log_dir = './dataset/%s/%s/log_data' % (args.dataset, args.sql_expr) 
    model, tokenizer = load_model()

    table_map = read_tables(args) 
    query_data = read_query(args)
    meta_data = read_meta(args)
    for idx, query_info in tqdm(enumerate(query_data), total=len(query_data)):
        meta_info = meta_data[idx]
        assert(query_info['id'] == meta_info['qid'])
         
        queries = [query_info['question']]
        table_id = query_info['table_id_lst'][0]
        table = table_map[table_id]['df']
        table_detail_data = table_map[table_id]['data']
        inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
        outputs = model(**inputs)
        predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
            inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
        )

        # let's print out the results:
        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
        aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

        answers = []
        for coordinates in predicted_answer_coordinates:
            if len(coordinates) == 1:
                # only a single cell:
                answers.append(table.iat[coordinates[0]])
            else:
                # multiple cells
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(table.iat[coordinate])
                answers.append(", ".join(cell_values))

        table_title = table_detail_data['documentTitle']
        print(table_title)
        display(table)
        table_csv_file = '%s/%s_table.csv' % (log_dir, query_info['id'])
        table.to_csv(table_csv_file) 

        print("")
        for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
            print(query)
            if predicted_agg == "NONE":
                print("Predicted answer: " + answer)
            else:
                print("Predicted answer: " + predicted_agg + " > " + answer)

        input('next')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--sql_expr', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
