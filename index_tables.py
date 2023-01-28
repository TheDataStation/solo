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
import math

StateImportCSV = 'import_csv'
StateGenTriples = 'gen_triples'
StateEncode = 'encode'
StateIndex = 'index'
EmbFileSuffix = '_embeddings'

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
    passage_dir = os.path.join(args.work_dir, 'open_table_discovery/table2txt/dataset', 
                                args.dataset, 'rel_graph')
    passage_file = os.path.join(passage_dir, 'passages.jsonl') 
    emb_file = os.path.join(passage_dir, '*_embeddings_*')
    index_dir = os.path.join(args.work_dir, 'index/on_disk_index_%s_rel_graph' % args.dataset)     
 
    check_data_lst = [] 
    if (args.pipe_step is None) or (args.pipe_step == ''): 
        tables_csv_exists = exists_tables_csv(dataset_dir)
        args.tables_csv_exists = tables_csv_exists 
        table_exists = os.path.exists(tables_file)
        passage_exists = os.path.exists(passage_file)
        emb_file_lst = glob.glob(emb_file) 
        emb_exists = len(emb_file_lst) > 0
        
        if tables_csv_exists:
            if table_exists:
                check_data = {'name': 'Tables imported', 'file_lst': [tables_file]}
                check_data_lst.append(check_data) 
        if passage_exists:
            check_data = {'name':'Triples', 'file_lst': [passage_file]}
            check_data_lst.append(check_data) 
        if emb_exists:
            check_data = {'name':'Triple embeddings', 'file_lst':emb_file_lst}
            check_data_lst.append(check_data) 
         
    index_exists = os.path.exists(index_dir)
    if index_exists:
        check_data = {'name':'Index', 'dir':index_dir}
         
    if len(check_data_lst) > 0:
        check_data_desc = get_check_data_desc(check_data_lst) 
        confirmed = input('%s already exists. If continue, the data will be removed and recreated. \n' % check_data_desc +
                          'Do you want to continue(y/n)? ')
        if confirmed.lower().strip() == 'y':
            clear_checked_data(check_data_lst)
            return True
        else:
            return False
    else:
        return True

def clear_checked_data(check_data_lst):
    for check_data in check_data_lst:
        file_lst = check_data.get('file_lst', [])
        for file_path in file_lst:
            os.remove(file_path) 
        data_dir = check_data.get('dir', None)
        if data_dir is not None:
            shutil.rmtree(data_dir)

def get_check_data_desc(check_data_lst):
    desc = '('
    for offset, check_data in enumerate(check_data_lst):
        desc += check_data['name']
        if offset < len(check_data_lst) - 1:
            desc += ' , '
    desc += ')'
    return desc

def main():
    args = get_args()
    pipe_sate_file = get_state_file(args.dataset)
    pipe_state_info = read_state(pipe_sate_file)
    if not confirm(args):
        return
    config = read_config()
    if (args.pipe_step is not None) and (args.pipe_step != ''):
        if args.pipe_step != 'emb_to_index':
            print('arg pipe_step only support "emb_to_index"')
            return
        else:
            pipe_triple_file = './table2txt/dataset/%s/rel_graph/passages.jsonl' % args.dataset
            create_index(pipe_state_info, pipe_sate_file, args, pipe_triple_file, EmbFileSuffix)
            return
             
    if args.tables_csv_exists:
        import_table_msg = '\nImporting tables'
        if config['table_sample_rows'] is not None:
            import_table_msg += '(Sample rows)'
        print(import_table_msg)
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
    num_triples = msg_info['num_triples'] 
   
    print('\nComputing space for encoding') 
    triple_file = msg_info['out_file']
    if not check_encode_space(triple_file):
        print('Encoding is stopped.')
        return
     
    print('\nEncoding triples')
    encode_triples(args.work_dir, triple_file, num_triples, config['num_encode_workers'], EmbFileSuffix)
    update_state(pipe_state_info, StateEncode, True, pipe_sate_file)
   
    #Creating index  
    create_index(pipe_state_info, pipe_sate_file, args, triple_file, EmbFileSuffix) 

def create_index(pipe_state_info, pipe_sate_file, args, triple_file, emb_file_suffix):
    emb_file_pattern = 'table2txt/dataset/%s/rel_graph/*%s_00' % (args.dataset, emb_file_suffix)
    out_emb_file_lst = glob.glob(emb_file_pattern) 
    if len(out_emb_file_lst) == 0:
        print('There is no triple embedding files')
        return

    print('\nComputing space for creating disk index')
    if not check_index_space(out_emb_file_lst):
        print('Creating disk index is stopped.')
        return
    
    print('\nCreating disk index')
    index_args = get_index_args(args.work_dir, args.dataset, '*' + emb_file_suffix + '_*')
    msg_info = ondisk_index.main(index_args)
    if pipe_state_info is not None:
        if not msg_info['state']:
            update_state(pipe_state_info, StateIndex, False, pipe_sate_file)
            print(msg_info['msg'])
        else:
            update_state(pipe_state_info, StateIndex, True, pipe_sate_file)

    index_dir = msg_info['index_dir']
    assert(os.path.isdir(index_dir))
    shutil.move(triple_file, index_dir)
    for out_emb_file in out_emb_file_lst:
        os.remove(out_emb_file)
    
    print('\nIndexing done')
     

def encode_triples(work_dir, graph_file, num_triples, num_encode_workers, emb_file_suffix):
    part_file_lst = split_triples(graph_file, num_triples, num_encode_workers)
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
        work_pool = ProcessPool(num_encode_workers)
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
 
def split_triples(triple_file, num_triples, num_workers):
    out_file_prefix = triple_file + '_part_'
    part_file_lst = glob.glob(out_file_prefix + '*')
    if len(part_file_lst) > 0:
        cmd = 'rm ' + out_file_prefix + '*'
        os.system(cmd)
    batch_size = int(num_triples / num_workers) + (1 if num_triples % num_workers else 0)
    cmd = 'split -l %d %s %s' % (batch_size, triple_file, out_file_prefix)
    os.system(cmd)
    part_file_lst = glob.glob(out_file_prefix + '*')
    part_file_lst.sort()
    assert(len(part_file_lst) > 0)
    return part_file_lst 

def get_unit_size():
    return (1024.0 * 1024.0 * 1024.0) 

def get_file_size(triple_file):
    file_gb = os.path.getsize(triple_file) / get_unit_size() 
    return file_gb

def get_free_space(triple_file):
    _, _, free_space = shutil.disk_usage(triple_file)
    free_gb = free_space / get_unit_size()
    return free_gb

def check_encode_space(triple_file):
    triple_size = get_file_size(triple_file)
    free_space = get_free_space(triple_file)
    needed_space = triple_size * 6
    return check_space(free_space, needed_space, 'Encoding')   

def check_index_space(out_emb_file_lst):
    emb_size = 0
    file_name_lst = []
    for emb_file in out_emb_file_lst:
        file_name = emb_file
        if not file_name.endswith('_00'):
            file_name = emb_file + '_00'
        emb_part_size = get_file_size(file_name)
        emb_size += emb_part_size
        file_name_lst.append(file_name)

    free_space = get_free_space(file_name_lst[0])
    needed_space = emb_size * 3.6
    return check_space(free_space, needed_space, 'Creating disk index') 

def check_space(free_space, needed_space, stage):
    if free_space < needed_space:
        msg = 'Free disk space (%.2f GB) is less than recommended %.2f GB. %s will fail. Continue?(y/n)' % \
              (free_space, needed_space, stage)
        user_opt = ''
        while user_opt not in ['y', 'n']:
            user_opt = input(msg)
            user_opt = user_opt.strip().lower()
        return user_opt == 'y'
    else:
        return True 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pipe_step', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

