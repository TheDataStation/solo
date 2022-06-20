import argparse
import os
from tqdm import tqdm
import glob
from table2txt import table2graph
import table_from_csv
import generate_passage_embeddings as passage_encoder
from src import ondisk_index
import shutil

def get_csv_args(work_dir, dataset):
    csv_args = argparse.Namespace(work_dir=work_dir,
                                  dataset=dataset
                                 )
    return csv_args 

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

def confirm(args):
    dataset_dir = os.path.join(args.work_dir, 'data', args.dataset)
    tables_file = os.path.join(dataset_dir, 'tables/tables.jsonl')
    passage_file = os.path.join(args.work_dir, 'open_table_discovery/table2txt/dataset', 
                                args.dataset, 'rel_graph/passages.jsonl') 
    index_dir = os.path.join(dataset_dir, 'index/on_disk_index_%s_rel_graph' % args.dataset)     
    
    table_exists = os.path.exists(tables_file)
    passage_exists = os.path.exists(passage_file)
    index_exists = os.path.exists(index_dir)
    if table_exists or index_exists: 
        confirmed = input('Tables or Index already exists, do you want to continue(y/n)? ')
        if confirmed.lower().strip() == 'y':
            if table_exists:
                os.remove(tables_file)
            if passage_exists:
                os.remove(passage_file)
            if index_exists:
                shutil.rmtree(index_dir)
            return True
        else:
            return False
    else:
        return True

def main():
    args = get_args()
    if not confirm(args):
        return
     
    print('Importing tables')
    csv_args = get_csv_args(args.work_dir, args.dataset)
    msg_info = table_from_csv.main(csv_args)
    if not msg_info['state']:
        print(msg_info['msg'])
        return

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
    index_dir = msg_info['index_dir']
    assert(os.path.isdir(index_dir))
    shutil.move(graph_file, index_dir)
    for out_emd_file in out_emd_file_lst:
        cmd = 'rm %s_*' % out_emd_file
        os.system(cmd)
               
def split_graphs(graph_file, batch_size):
    out_file_prefix = graph_file + '_part_'
    part_file_lst = glob.glob(out_file_prefix + '*')
    if len(part_file_lst) > 0:
        cmd = 'rm ' + out_file_prefix + '*'
        os.system(cmd)
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
    parser.add_argument('--batch_size', type=int, default=5000000)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

