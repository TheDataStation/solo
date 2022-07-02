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


def update_min_tables(item, top_n, min_tables):
    assert(top_n >= 25)
    passage_lst = item['ctxs']
    top_passage_lst = passage_lst[:top_n]
    table_lst = [a['tag']['table_id'] for a in top_passage_lst]
    table_set = set(table_lst)
    top_n_tables = len(table_set)
    if top_n_tables >= min_tables:
        item['ctxs'] = top_passage_lst
        return
    
    table_lst, table_dict = group_passages(passage_lst)
    if len(table_lst) < min_tables:
        item['ctxs'] = top_passage_lst
        return
    
    min_passages = 5
    num_added = 0
    for idx in range(top_n_tables, min_tables):
        table_id = table_lst[idx]
        sub_lst = table_dict[table_id]
        top_sub_lst = sub_lst[:min_passages]
        table_dict[table_id] = top_sub_lst
        num_added += len(top_sub_lst)

    top_n_table_sub_total = 0
    for idx in range(top_n_tables):
        table_id = table_lst[idx]
        top_n_table_sub_total += len(table_dict[table_id])
         
    num_subtract = top_n_table_sub_total - (top_n - num_added)
    assert(num_subtract > 0)
    
    idx = top_n_tables - 1
    while (idx >= 0) and (num_subtract > 0):
        table_id = table_lst[idx]
        sub_lst = table_dict[table_id]
        num_subtract_sub = min(num_subtract, len(sub_lst) - min_passages)
        if num_subtract_sub > 0:
            sub_top_n = len(sub_lst) - num_subtract_sub
            sub_lst = sub_lst[:sub_top_n]
            table_dict[table_id] = sub_lst 
            num_subtract -= num_subtract_sub
        idx -= 1
    
    top_passage_lst = []
    for table_id in table_lst:
        top_passage_lst.extend(table_dict[table_id])
        if len(top_passage_lst) >= top_n:
            break
    
    top_passage_lst = top_passage_lst[:top_n]
    assert(len(top_passage_lst) == top_n)
    item['ctxs'] = top_passage_lst

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

def process_train(train_data, top_n, table_dict, strategy, min_tables):
    updated_train_data = []
    for item in tqdm(train_data):
        #pos_lst, neg_lst = collect_passages(item)
        update_min_tables(item, top_n, min_tables)
        gold_table_lst = item['table_id_lst']
        ctxs = item['ctxs']
        labels = [int(a['tag']['table_id'] in gold_table_lst) for a in ctxs]
        
        if max(labels) < 1: # all negatives
            continue
            #if len(pos_lst) > 0:
            #    M = min(top_n // 2, len(pos_lst))
            #    item['ctxs'] = pos_lst[:M] + ctxs[:(len(ctxs)-M)]
            #else:
            #    continue
        
        if min(labels) > 0: # all positives
            continue
            #if len(neg_lst) > 0:
            #    M = min(top_n // 2, len(neg_lst))
            #    item['ctxs'] = ctxs[:(len(ctxs)-M)] + neg_lst[:M]
            #else:
            #    continue

        assert(len(item['ctxs']) == top_n) 
        updated_train_data.append(item)

    tag_data_text(updated_train_data, table_dict, strategy)
    return updated_train_data

def process_dev(dev_data, top_n, table_dict, strategy, min_tables):
    updated_dev_data = []
    for item in tqdm(dev_data):
        update_min_tables(item, top_n, min_tables)
        ctxs = item['ctxs']
        item['ctxs'] = ctxs
        updated_dev_data.append(item)
    tag_data_text(updated_dev_data, table_dict, strategy)
    return updated_dev_data 

