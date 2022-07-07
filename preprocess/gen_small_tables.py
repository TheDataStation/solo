import json
import random


def get_gold_tables(tag):
    data_file = '/home/cc/data/nq/interactions/' + tag + '.jsonl'
    table_id_lst = []
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            table_id = item['table']['tableId']
            table_id_lst.append(table_id)
    table_set = set(table_id_lst)
    return table_set

def get_all_tables():
    data_file = '/home/cc/data/nq/tables/tables.jsonl'
    table_id_lst = []
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            table_id = item['tableId']
            table_id_lst.append(table_id)
    table_set = set(table_id_lst)
    return table_set

def main():
    dev_gold_table_set = get_gold_tables('dev')
    test_gold_table_set = get_gold_tables('test')
    all_table_set = get_all_tables()
    
    other_table_set = all_table_set - dev_gold_table_set - test_gold_table_set
    other_table_lst = list(other_table_set)
    M = 20000
    example_other_table_lst = random.sample(other_table_lst, M)
    
    example_table_set = dev_gold_table_set.union(test_gold_table_set, set(example_other_table_lst))
    
    example_table_lst = list(example_table_set)
    with open('/home/cc/data/nq/tables/tables_small.jsonl', 'w') as f_o:
        for example_table in example_table_lst:
            f_o.write(example_table + '\n')



if __name__ == '__main__':
    main()
