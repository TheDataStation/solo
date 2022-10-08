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
    parser.add_argument('--data_size', type=int, required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    dataset_dir = '/home/cc/code/open_table_discovery/table2question/dataset'
    data_part_dir = os.path.join(dataset_dir, args.dataset, 'sql_data/train_0/rel_graph/data_parts')
    file_name = '%s.jsonl' % args.part_no
    data_file = os.path.join(data_part_dir, file_name)
   
    out_file_name_tagged = '%s_tag_1_%d.jsonl' % (args.part_no, args.data_size) 
    out_file_name_no_tag = '%s_tag_0_%d.jsonl' % (args.part_no, args.data_size)
    
    out_file_tagged = os.path.join(data_part_dir, out_file_name_tagged)
    out_file_no_tag = os.path.join(data_part_dir, out_file_name_no_tag)
    
    if os.path.isfile(out_file_tagged):
        raise ValueError('%s already exists' % out_file_tagged)
   
    if os.path.isfile(out_file_no_tag):
        raise ValueError('%s already exists' % out_file_no_tag)
    
    f_o_tagged = open(out_file_tagged, 'w')
    f_o_no_tag = open(out_file_no_tag, 'w')
    data_size_count = 0 
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            ctx_lst = item['ctxs']
            good_item = True
            for ctx_info in ctx_lst:
                passage = ctx_info['text']
                passage_untagged = untag_passage(passage)
                if passage_untagged is None:
                    good_item = False
                    break         
                ctx_info['text'] = passage_untagged
            if good_item:
                f_o_tagged.write(line)
                f_o_no_tag.write(json.dumps(item) + '\n')
                data_size_count += 1
                if data_size_count >= args.data_size:
                    break
                 
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

def untag_passage(passage):
    ok_1 = check_less_than_two(passage, tag_title)
    ok_2 = check_less_than_two(passage, tag_sub_name)
    ok_3 = check_less_than_two(passage, tag_sub)
    ok_4 = check_less_than_two(passage, tag_obj_name)
    ok_5 = check_less_than_two(passage, tag_obj)
    
    if ok_1 and ok_2 and ok_3 and ok_4 and ok_5:
        updated_passage = passage 
        updated_passage = updated_passage.replace(tag_title, '', 1) 
        updated_passage = updated_passage.replace(tag_sub_name, ' , ', 1) 
        updated_passage = updated_passage.replace(tag_sub, ' , ', 1) 
        updated_passage = updated_passage.replace(tag_obj_name, ' , ', 1) 
        updated_passage = updated_passage.replace(tag_obj, ' , ', 1) 
        return updated_passage 
    else:
        return None

if __name__ == '__main__':
    main()
