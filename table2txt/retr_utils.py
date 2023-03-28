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

def type_process(col):
    ret_col = col
    if isinstance(col, str):
        if col == 'None':
            ret_col = None
        else:
            ret_col = int(col)
    return ret_col 

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
        
        sub_col = type_process(sub_col)
        obj_col = type_process(obj_col) 
         
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

def group_passages(passage_lst):
    table_dict = {}
    table_lst = []
    for passage_info in passage_lst:
        table_id = passage_info['tag']['table_id']
        if table_id not in table_dict:
            table_dict[table_id] = []
            table_lst.append(table_id)
        sub_lst = table_dict[table_id]
        sub_lst.append(passage_info)
    return table_lst, table_dict 


def truncate_passages(item, top_n, max_ctx_per_table):
    passage_lst = item['ctxs']
    table_lst, table_dict = group_passages(passage_lst)
    top_tables = table_lst
    
    ctx_used_lst = []
    ctx_not_used_lst = []
    for table_id in top_tables:
        table_ctx_lst = table_dict[table_id]
        ctxs_used = table_ctx_lst[:max_ctx_per_table]
        ctx_used_lst.extend(ctxs_used)
        ctxs_not_used = table_ctx_lst[len(ctxs_used):]
        ctx_not_used_lst.extend(ctxs_not_used) 
    
    if len(ctx_used_lst) < top_n:
        num_more_ctxs = top_n - len(ctx_used_lst)
        ctx_used_lst += ctx_not_used_lst[:num_more_ctxs]
    else:
        ctx_used_lst = ctx_used_lst[:top_n]

    item['ctxs'] = ctx_used_lst


def collect_passages(item):
    ctx_lst = item['ctxs']
    pos_lst = []
    neg_lst = []
    gold_table_lst = item['table_id_lst']
    for passage_info in ctx_lst:
        if passage_info['tag']['table_id'] in gold_table_lst:
            pos_lst.append(passage_info)
        else:
            neg_lst.append(passage_info)
    return pos_lst, neg_lst

def process_train(train_data, top_n, table_dict, strategy, max_ctx_per_table=10):
    updated_train_data = []
    for item in tqdm(train_data):
        truncate_passages(item, top_n, max_ctx_per_table)
        gold_table_lst = item['table_id_lst']
        ctxs = item['ctxs']
        labels = [int(a['tag']['table_id'] in gold_table_lst) for a in ctxs]
        
        if max(labels) < 1: # all negatives
            continue
        
        if min(labels) > 0: # all positives
            continue

        assert(len(item['ctxs']) == top_n) 
        updated_train_data.append(item)

    tag_data_text(updated_train_data, table_dict, strategy)
    return updated_train_data


def process_dev(dev_data, top_n, table_dict, strategy, max_ctx_per_table=5):
    updated_dev_data = []
    for item in tqdm(dev_data):
        truncate_passages(item, top_n, max_ctx_per_table)
        ctxs = item['ctxs']
        item['ctxs'] = ctxs
        updated_dev_data.append(item)
    tag_data_text(updated_dev_data, table_dict, strategy)
    return updated_dev_data 

