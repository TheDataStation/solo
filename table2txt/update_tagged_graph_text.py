import json
from tqdm import tqdm
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--synthetic', type=int, required=True)
    args = parser.parse_args()
    return args

def read_passages(dataset, expr, passage_file):
    passage_dict = {}
    passage_dir = '/home/cc/code/fusion_in_decoder/data/on_disk_index_%s_%s' % (dataset, expr)
    data_file = os.path.join(passage_dir, passage_file)
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            p_id = int(item['p_id'])
            passage_dict[p_id] = item['passage'].strip()
    return passage_dict

def get_annotated_text(args):
    rel_graph_dict = read_passages(args.dataset, 'rel_graph', 'passages.jsonl')
    graph_text_dict = read_passages(args.dataset, 'graph_text', 'merged_passages.jsonl')
    annotated_text_dict = {}
    for p_id in tqdm(rel_graph_dict):
        rel_graph = rel_graph_dict[p_id]
        graph_text = graph_text_dict[p_id]
        if len(graph_text) > len(rel_graph):
            idx = graph_text.index(rel_graph)
            assert(idx == 0)
            pos = len(rel_graph)
            annotated_text = graph_text[pos:].strip()
            if (rel_graph[-1] != '.') and (annotated_text[0] == '.'):
                annotated_text = annotated_text[1:]
            annotated_text_dict[p_id] = annotated_text.strip()
        else:
            assert(rel_graph == graph_text)

    return annotated_text_dict         

def main():
    args = get_args()
    if args.synthetic:
        data_dir = '../table2question/dataset/%s/sql_data/%s/graph_text/' % (args.dataset, args.mode)
        file_name = 'fusion_retrieved_tagged.jsonl'
        out_file_name = 'fusion_retrieved_tagged_merged.jsonl'
        out_file = os.path.join(data_dir, out_file_name)
    else:
        data_dir = './dataset/%s/graph_text' % args.dataset
        file_name = 'fusion_retrieved_%s_tagged.jsonl' % args.mode
        out_file = os.path.join(data_dir, 'fusion_retrieved_%s_tagged_merged.jsonl' % args.mode)
    
    if os.path.exists(out_file):
        print('(%s) already exists.' % out_file)
        return
    f_o = open(out_file, 'w')
    
    annotated_text_dict = get_annotated_text(args)
    
    data_file = os.path.join(data_dir, file_name)
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            ctx_lst = item['ctxs']
            for ctx in ctx_lst:
                p_id = int(ctx['id'])
                passage = ctx['text']
                if p_id in annotated_text_dict:
                    annotated_text = annotated_text_dict[p_id]
                    updated_passage = passage + ' [G] ' + annotated_text
                else:
                    updated_passage = passage
                ctx['text'] = updated_passage

            f_o.write(json.dumps(item) + '\n')
    
    f_o.close()

if __name__ == '__main__':
    main()

