import json
import argparse
from tqdm import tqdm
import os

tag_title = '[T]'
tag_sub_name = '[SC]'
tag_sub = '[S]'
tag_obj_name = '[OC]'
tag_obj = '[O]'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--part_no', type=str, required=True)
    args = parser.parse_args()
    return args

def read_tables(dataset):
    table_data = {}
    table_file = '/home/cc/code/data/%s/tables/tables.jsonl' % dataset
    with open(table_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['tableId']
            table_data[table_id] = item
    return table_data


def main():
    args = get_args()
    table_dict = read_tables(args.dataset)
    dataset_dir = '/home/cc/code/open_table_discovery/table2question/dataset'
    data_part_dir = os.path.join(dataset_dir, args.dataset, 'sql_data/train_0/rel_graph/data_parts')
    file_name = '%s.jsonl' % args.part_no
    data_file = os.path.join(data_part_dir, file_name)
   
    out_file_name_tagged = '%s_tag_1.jsonl' % (args.part_no) 
    out_file_name_no_tag = '%s_tag_0.jsonl' % (args.part_no)
    
    out_file_tagged = os.path.join(data_part_dir, out_file_name_tagged)
    out_file_no_tag = os.path.join(data_part_dir, out_file_name_no_tag)
    
    if os.path.isfile(out_file_tagged):
        raise ValueError('%s already exists' % out_file_tagged)
   
    if os.path.isfile(out_file_no_tag):
        raise ValueError('%s already exists' % out_file_no_tag)
    
    f_o_tagged = open(out_file_tagged, 'w')
    f_o_no_tag = open(out_file_no_tag, 'w')
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            qid = item['id']
            ctx_lst = item['ctxs']
            for ctx_info in ctx_lst:
                passage = ctx_info['text']
                table_id = ctx_info['tag']['table_id']
                passage_untagged = untag_passage(qid, ctx_info['id'], passage, ctx_info, table_dict[table_id])
                ctx_info['text'] = passage_untagged

            f_o_tagged.write(line)
            f_o_no_tag.write(json.dumps(item) + '\n') 
     
    f_o_tagged.close()
    f_o_no_tag.close()             
    
def check_less_than_two(passage, tag):
    count = 0
    start = 0
    offset = passage.find(tag, start)
    while offset >= 0:
        count += 1
        start = offset + 1
        offset = passage.find(tag, start)
    if count >= 2:
        return False
    return True 

def untag_passage(qid, pid, passage, ctx_info, table_info):
    row = ctx_info['tag']['row']
    sub_col = ctx_info['tag']['sub_col']
    obj_col = ctx_info['tag']['obj_col']

    title = table_info['documentTitle']
    out_passage = title + ' , '

    if sub_col is not None:
        out_passage += table_info['columns'][sub_col]['text'] + ' , '
        out_passage += table_info['rows'][row]['cells'][sub_col]['text'] + ' , '

    out_passage += table_info['columns'][obj_col]['text'] + ' , '
    out_passage += table_info['rows'][row]['cells'][obj_col]['text'] + ' . '

    return out_passage


if __name__ == '__main__':
    main()
