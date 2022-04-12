from tqdm import tqdm
from table2txt.graph_strategy.rel_tags import RelationTag
from table2txt.table2tokens import tag_slide_tokens

def tag_data_text(data, table_dict, strategy):
    tag_func = None
    if strategy == 'rel_graph':
        tag_func = tag_rel_graph
    elif strategy == 'slide':
        tag_func = tag_slide_tokens
    else:
        raise ValueError('strategy (%s) not supported')

    for item in tqdm(data):
        tag_func(item, table_dict)

def tag_rel_graph(item, table_dict):
    passage_info_lst = item['ctxs']
    for passage_info in passage_info_lst:
        tag_info = passage_info['tag']
        table_id = tag_info['table_id']
        row = tag_info['row']
        sub_col = tag_info['sub_col']
        obj_col = tag_info['obj_col']
        table_data = table_dict[table_id]
        title = table_data['documentTitle']
        
        if sub_col is None:
            sub_name = ''
            sub = ''
        else:
            sub_name = table_data['columns'][sub_col]['text']
            sub = table_data['rows'][row]['cells'][sub_col]['text']

        obj_name = table_data['columns'][obj_col]['text']
        obj = table_data['rows'][row]['cells'][obj_col]['text'] 
        tagged_text = RelationTag.get_tagged_text(title, sub_name, sub, obj_name, obj)        
        passage_info['text'] = tagged_text

def process_train(train_data, top_n, table_dict, strategy):
    updated_train_data = []
    for item in tqdm(train_data):
        gold_table_lst = item['table_id_lst']
        ctxs = item['ctxs'][:top_n]
        labels = [int(a['tag']['table_id'] in gold_table_lst) for a in ctxs]
        if (max(labels) < 1) or (min(labels) > 0):
            continue
        item['ctxs'] = ctxs
        updated_train_data.append(item)

    tag_data_text(updated_train_data, table_dict, strategy)
    return updated_train_data

def process_dev(dev_data, top_n, table_dict, strategy):
    updated_dev_data = []
    for item in tqdm(dev_data):
        ctxs = item['ctxs'][:top_n]
        item['ctxs'] = ctxs
        updated_dev_data.append(item)
    tag_data_text(updated_dev_data, table_dict, strategy)
    return updated_dev_data 

