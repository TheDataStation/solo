import os
import json
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--expr', type=str)
    parser.add_argument('--passage_file', type=str)

    args = parser.parse_args()
    return args

def get_train_file(args):
    data_dir = os.path.join('./dataset', args.dataset, 'sql_all_per_10', args.expr)
    data_file = os.path.join(data_dir, 'fusion_retrieved_train_100_passages.jsonl')
    out_file = os.path.join(data_dir, 'fusion_retrieved_train.jsonl')
    return data_file, out_file

def get_meta_data(args):
    meta_data = []
    meta_file = os.path.join('./dataset', args.dataset, 'sql_all_per_10', 'meta.txt') 
    with open(meta_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            meta_data.append(item)
    return meta_data

def get_passage_key(table_id, row, sub_col, obj_col):
    key = '%s_%d_%s_%s' % (table_id, row, str(sub_col), str(obj_col))
    return key

def read_passages(args):
    passage_data = {}
    with open(args.passage_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['tag']['table_id']
            row = item['tag']['row']
            sub_col = item['tag']['sub_col']
            obj_col = item['tag']['obj_col']
            key = get_passage_key(table_id, row, sub_col, obj_col)
            passage_data[key] = item
    return passage_data

def get_question_meta(qid, table_id, meta_data):
    index = int(qid.split('_')[1])
    meta_item = meta_data[index]
    sel_col = meta_item['sql']['sel']
    cond_lst = meta_item['sql']['conds']
    cond_cols = [cond[0] for cond in cond_lst]
    row = meta_item['row']
    ref_cols = list(set([sel_col] + cond_cols))
    ref_cols.sort()

    assert(table_id == meta_item['table_id'])
    meta_info = {
        'row':row,
        'ref_cols':ref_cols
    }
    return meta_info

def get_postive_passages(qid, table_id, meta_data, passage_data):
    meta_info = get_question_meta(qid, table_id, meta_data)
    row = meta_info['row']
    ref_cols = meta_info['ref_cols']
    pos_passage_info_lst = []
    
    for obj_col in ref_cols:
        sub_col = None
        key = get_passage_key(table_id, row, sub_col, obj_col)
        if key in passage_data:
            pos_passage_info = passage_data[key]
            pos_passage_info_lst.append(pos_passage_info)
        else:
            print('there is no key, %s' % key)
             
    N = len(ref_cols)
    for idx in range(N-1):
        sub_col = ref_cols[idx]
        obj_col = ref_cols[idx+1]
        key = get_passage_key(table_id, row, sub_col, obj_col)
        if key in passage_data:
            pos_passage_info = passage_data[key]
            pos_passage_info_lst.append(pos_passage_info)
        else:
            print('there is no key, %s' % key)
    return pos_passage_info_lst[:20]

def main():
    args = get_args()
    train_file, out_train_file = get_train_file(args)
    meta_data = get_meta_data(args)
    passage_data = read_passages(args)
    f_o = open(out_train_file, 'w')
    with open(train_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            qid = item['id']
            gold_tables = item['table_id_lst']
            table_id = gold_tables[0]
            ctx_lst = item['ctxs']
            retr_correct_labels = [(a['tag']['table_id'] in gold_tables) for a in ctx_lst]
            ret_pos_id_set = set()
            retr_pos_lst = []
            retr_neg_lst = []
            for idx, correct in enumerate(retr_correct_labels):
                if correct:
                    retr_pos_lst.append(ctx_lst[idx])
                    ret_pos_id_set.add(int(ctx_lst[idx]['id']))
                else:
                    retr_neg_lst.append(ctx_lst[idx])
            
            new_pos_ctx_lst = []
            pos_passage_info_lst = get_postive_passages(qid, table_id, meta_data, passage_data) 
            for passage_info in pos_passage_info_lst:
                p_id = int(passage_info['p_id'])
                if p_id not in ret_pos_id_set:
                    new_pos_ctx = {
                        'id':p_id,
                        'title':'',
                        'text':passage_info['passage'],
                        'tag':passage_info['tag']
                    }
                    new_pos_ctx_lst.append(new_pos_ctx)
            
            updated_pos_lst = new_pos_ctx_lst + retr_pos_lst
            updated_pos_lst = updated_pos_lst[:50]
            num_neg = len(ctx_lst) - len(updated_pos_lst)
            updated_neg_lst = retr_neg_lst[:num_neg]
            
            if len(updated_pos_lst) == 0:
                print('question %s is ignored, no pos' % qid)
                continue

            if len(updated_neg_lst) == 0:
                print('question %s is ignored, no neg' % qid)
                continue

            item['ctxs'] = updated_pos_lst + updated_neg_lst

            f_o.write(json.dumps(item) + '\n')
    
    f_o.close()

if __name__ == '__main__':
    main()
