import argparse
import os
from tqdm import tqdm
import glob
from table2txt import table2graph
import generate_passage_embeddings as passage_encoder
from src import ondisk_index

def get_graph_args(work_dir, dataset):
    graph_args = argparse.Namespace(work_dir=work_dir, 
                                    dataset=dataset,
                                    experiment='rel_graph',
                                    table_file='tables.jsonl',
                                    strategy='RelationGraph'
                                    )
    return graph_args

def get_encoder_args(model_path):
    encoder_args = argparse.Namespace(passages=None, 
                                      output_path=None,
                                      shard_id=0,
                                      num_shards=1,
                                      per_gpu_batch_size=32,
                                      passage_maxlength=200,
                                      model_path=model_path,
                                      no_fp16=False
                                     )
    return encoder_args

def get_index_args(work_dir, dataset, emb_file):
    index_args = argparse.Namespace(work_dir=work_dir,
                                    dataset=dataset,
                                    experiment='rel_graph',
                                    emb_file=emb_file
                                    )
    return index_args 

def main():
    args = get_args()
    print('Linearizing table rows')
    graph_args = get_graph_args(args.work_dir, args.dataset)
    msg_info = table2graph.main(graph_args)
    graph_ok = msg_info['state']
    if not graph_ok:
        return
    
    graph_file = msg_info['out_file']
    part_file_lst = split_graphs(graph_file, args.batch_size)
    encoder_model = os.path.join(args.work_dir, 'models/tqa_retriever')
    emd_file_suffix = '_embeddings'
    out_emd_file_lst = []
    for part_file in part_file_lst:
        print('Encoding %s' % part_file)
        encoder_args = get_encoder_args(encoder_model)
        encoder_args.passages = part_file
        encoder_args.output_path = part_file + emd_file_suffix
        out_emd_file_lst.append(encoder_args.output_path)
        passage_encoder.main(encoder_args, is_main=False) 
        os.remove(part_file)
    
    index_args = get_index_args(args.work_dir, args.dataset, '*' + emd_file_suffix + '_*')
    msg_info = ondisk_index.main(index_args)
    if not msg_info['state']:
        print(msg_info['msg'])
    for out_emd_file in out_emd_file_lst:
        cmd = 'rm %s_*' % out_emd_file
        os.system(cmd)
               
def split_graphs(graph_file, batch_size):
    out_file_prefix = graph_file + '_part_'
    cmd = 'split -l %d %s %s' % (batch_size, graph_file, out_file_prefix)
    os.system(cmd)
    part_file_lst = glob.glob(out_file_prefix + '*')
    part_file_lst.sort()
    assert(len(part_file_lst) > 0)
    return part_file_lst 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=10000)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

