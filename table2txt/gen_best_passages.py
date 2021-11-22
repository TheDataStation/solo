import json
import os
from tqdm import tqdm

Stg_CompleteGraph = 'CompleteGraph'
Stg_GraphNoCaption = 'GraphNoCaption' 

def read_stg_cfg():
    stg_cfg_dict = {}
    part_lst = ['a', 'b', 'c']
    data_dir = '/home/cc/code/open_table_discovery/table2txt/graph_strategy/output/'
    for part in part_lst:
        data_file = os.path.join(data_dir, 'tables_500_a%s_best_stg.jsonl' % part)
        with open(data_file) as f:
            for line in f:
                item = json.loads(line)
                table_id = item['table_id']
                stg = item['strategy']
                stg_cfg_dict[table_id] = stg
    return stg_cfg_dict

def read_passages(stg):
    table_passage_dict = {}
    data_file = '/home/cc/code/open_table_discovery/table2txt/dataset/nq_tables/%s/graph_passages.json' % stg
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['tag']
            if table_id not in table_passage_dict:
                table_passage_dict[table_id] = []

            passage_lst = table_passage_dict[table_id]
            item['id'] = ('%s_%s' % (stg, item['id']))
            passage_lst.append(json.dumps(item))

    return table_passage_dict

def get_table_lst():
    table_id_lst = []
    with open('/home/cc/data/nq_tables/tables/tables_500.jsonl') as f:
        for line in f:
            item = json.loads(line)
            table_id = item['tableId']
            table_id_lst.append(table_id)
    return table_id_lst

def main():
    table_id_lst = get_table_lst()
    stg_cfg_dict = read_stg_cfg()
    stg_passage_dict = {} 
    stg_passage_dict[Stg_CompleteGraph] = read_passages(Stg_CompleteGraph)
    stg_passage_dict[Stg_GraphNoCaption] = read_passages(Stg_GraphNoCaption)
   
    out_file = './dataset/nq_tables/Composite/graph_passages.json'
    with open(out_file, 'w') as f_o:  
        for table_id in tqdm(table_id_lst):
            if table_id not in stg_cfg_dict:
                stg = Stg_CompleteGraph
            else:
                stg = stg_cfg_dict[table_id]
            
            table_passage_dict = stg_passage_dict[stg]
            passage_lst = table_passage_dict[table_id]
            for line in passage_lst:
                f_o.write(line + '\n') 


if __name__ == '__main__':
    main()
