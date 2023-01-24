import argparse
import os
from tqdm import tqdm
import glob
from table2txt import table2graph
import table_from_csv
import generate_passage_embeddings as passage_encoder
from src import ondisk_index
import shutil
import json
from trainer import read_config
from multiprocessing import Pool as ProcessPool

StateImportCSV = 'import_csv'
StateGenTriples = 'gen_triples'
StateEncode = 'encode'
StateIndex = 'index'

def get_state_file(dataset):
    return 'index_state_%s.json' % dataset 

def read_state(state_file):
    if os.path.isfile(state_file):
        with open(state_file) as f:
            state_info = json.load(f)
    else:
        state_info = {
            StateImportCSV:False,
            StateGenTriples:False,
            StateEncode:False,
            StateIndex:False
        }
    return state_info

def update_state(state_info, state_key, state, state_file):
    state_info[state_key] = state
    with open(state_file, 'w') as f_o:
        f_o.write(json.dumps(state_info))

def get_csv_args(work_dir, dataset, config):
    csv_args = argparse.Namespace(work_dir=work_dir,
                                  dataset=dataset,
                                  file_name_title=config['file_name_title'],
                                  table_sample_rows=config['table_sample_rows']
                                 )
    return csv_args 

def get_graph_args(work_dir, dataset, config):
    graph_args = argparse.Namespace(work_dir=work_dir, 
                                    dataset=dataset,
                                    experiment='rel_graph',
                                    table_file='tables.jsonl',
                                    strategy='RelationGraph',
                                    table_chunk_size=config['table_chunk_size'],
                                    table_import_batch=config['table_import_batch']
                                    )
    return graph_args

def get_encoder_args(model_path, show_progress=True):
    encoder_args = argparse.Namespace(passages=None, 
                                      output_path=None,
                                      shard_id=0,
                                      num_shards=1,
                                      per_gpu_batch_size=32,
                                      passage_maxlength=200,
                                      model_path=model_path,
                                      no_fp16=False,
                                      show_progress=show_progress
                                     )
    return encoder_args

def get_index_args(work_dir, dataset, emb_file):
    index_args = argparse.Namespace(work_dir=work_dir,
                                    dataset=dataset,
                                    experiment='rel_graph',
                                    emb_file=emb_file
                                    )
    return index_args 

def exists_tables_csv(dataset_dir):
    csv_file_pattern = os.path.join(dataset_dir, 'tables_csv', '**', '*.csv')
    csv_file_lst = glob.glob(csv_file_pattern, recursive=True)
    return len(csv_file_lst) > 0

def confirm(args):
    dataset_dir = os.path.join(args.work_dir, 'data', args.dataset)
    tables_file = os.path.join(dataset_dir, 'tables/tables.jsonl')
    passage_file = os.path.join(args.work_dir, 'open_table_discovery/table2txt/dataset', 
                                args.dataset, 'rel_graph/passages.jsonl') 
    index_dir = os.path.join(args.work_dir, 'index/on_disk_index_%s_rel_graph' % args.dataset)     
   
    tables_csv_exists = exists_tables_csv(dataset_dir)
    args.tables_csv_exists = tables_csv_exists 
    table_exists = os.path.exists(tables_file)
    passage_exists = os.path.exists(passage_file)
    index_exists = os.path.exists(index_dir)
    if tables_csv_exists:
        if table_exists:
            os.remove(tables_file)
    if passage_exists:
        os.remove(passage_file)
    if index_exists: 
        confirmed = input('Index already exists. If continue, index will be rebuilt. \n' +
                          'Do you want to continue(y/n)? ')
        if confirmed.lower().strip() == 'y':
            shutil.rmtree(index_dir)
            return True
        else:
            return False
    else:
        return True

def main():
    args = get_args()
    pipe_sate_file = get_state_file(args.dataset)
    pipe_state_info = read_state(pipe_sate_file)
    if not confirm(args):
        return
     
    config = read_config()
    
    if args.tables_csv_exists:
        print('\nImporting tables')
        csv_args = get_csv_args(args.work_dir, args.dataset, config)
        msg_info = table_from_csv.main(csv_args)
        if not msg_info['state']:
            update_state(pipe_state_info, StateImportCSV, False, pipe_sate_file)
            print(msg_info['msg'])
            return
        else:
            update_state(pipe_state_info, StateImportCSV, True, pipe_sate_file)
    
    print('\nGenerating triples')
    graph_args = get_graph_args(args.work_dir, args.dataset, config)
    msg_info = table2graph.main(graph_args)
    graph_ok = msg_info['state']
    if not graph_ok:
        update_state(pipe_state_info, StateGenTriples, False, pipe_sate_file)
        return
    else:
        update_state(pipe_state_info, StateGenTriples, True, pipe_sate_file)
    
    print('\nEncoding triples')
    emb_file_suffix = '_embeddings'
    graph_file = msg_info['out_file']
    out_emb_file_lst = encode_triples(args.work_dir, graph_file, config['encode_batch'], emb_file_suffix)
    update_state(pipe_state_info, StateEncode, True, pipe_sate_file)
    
    print('\nCreating disk index')
    index_args = get_index_args(args.work_dir, args.dataset, '*' + emb_file_suffix + '_*')
    msg_info = ondisk_index.main(index_args)
    if not msg_info['state']:
        update_state(pipe_state_info, StateIndex, False, pipe_sate_file)
        print(msg_info['msg'])
    else:
        update_state(pipe_state_info, StateIndex, True, pipe_sate_file)

    index_dir = msg_info['index_dir']
    assert(os.path.isdir(index_dir))
    shutil.move(graph_file, index_dir)
    for out_emb_file in out_emb_file_lst:
        cmd = 'rm %s_*' % out_emb_file
        os.system(cmd)
    
    print('\nIndexing done')

def encode_triples(work_dir, graph_file, batch_size, emb_file_suffix):
    part_file_lst = split_graphs(graph_file, batch_size)
    encoder_model = os.path.join(work_dir, 'models/tqa_retriever')
    out_emb_file_lst = []
    part_info_lst = []
    for part_idx, part_file in enumerate(part_file_lst):
        part_info = {
            'file_name':part_file,
            'encoder_model':encoder_model,
            'emb_file_suffix':emb_file_suffix,
            'show_progress':(part_idx == 0)
        }
        part_info_lst.append(part_info)
    
    multi_process = True
    if multi_process:
        num_workers = len(part_info_lst)
        work_pool = ProcessPool(num_workers)
        for out_emb_file in tqdm(work_pool.imap_unordered(encode_part_trples, part_info_lst), total=len(part_info_lst)):
            out_emb_file_lst.append(out_emb_file)
    else:
        for part_info in part_info_lst:
            out_emb_file = encode_part_trples(part_info)
            out_emb_file_lst.append(out_emb_file)
    return out_emb_file_lst

def encode_part_trples(part_info):
    part_file = part_info['file_name']
    encoder_model = part_info['encoder_model']
    emb_file_suffix = part_info['emb_file_suffix']

    print('Encoding %s' % os.path.basename(part_file))
    encoder_args = get_encoder_args(encoder_model, part_info['show_progress'])
    encoder_args.passages = part_file
    encoder_args.output_path = part_file + emb_file_suffix
    passage_encoder.main(encoder_args, is_main=False) 
    os.remove(part_file)
    return encoder_args.output_path
 
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
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

