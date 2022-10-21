import json
import glob
import argparse

def get_all_metrics():
    file_pattern = 'metric_step_*.jsonl'
    data_file_lst = glob.glob(file_pattern)
    metric_info_map = {}
    for data_file in data_file_lst:
        parts = data_file.split('_')
        step = int(parts[2].split('.')[0])
        with open(data_file) as f:
            metric_info = json.load(f)
        metric_info_map[step] = metric_info
    return metric_info_map    

def get_all_setps():
    metric_info_map = get_all_metrics()
    file_pattern = '*.pt'
    data_file_lst = glob.glob(file_pattern) 
    step_info_lst = []
    
    step_info_map = {}
    for data_file in data_file_lst:
        offset = data_file.rindex('sql_0')
        base_name = data_file[offset:]
        parts = base_name.split('_')
        epoch = int(parts[3])
        step = int(parts[5])
        step_info = {
            'epoch':epoch,
            'step':step,
            'metric':metric_info_map[step]
        }
        step_info_lst.append(step_info)
        step_info_map[step] = step_info 
    sorted_lst = sorted(step_info_lst, key=lambda a: a['step'])

    return sorted_lst, step_info_map 

def verify_best_step(args):
    data_file = 'best_metric_info.json'
    with open(data_file) as f:
        metric_info = json.load(f)
    model_file = metric_info['model_file']
    #sql_0_epoc_6_step_1600_model.pt
    offset = model_file.rindex('sql_0')
    base_name = model_file[offset:]
    parts = base_name.split('_')
    best_epoch = int(parts[3])
    best_step = int(parts[5])
    print('best_epoch =', best_epoch, 'best_step =', best_step)
    
    patience_steps = int((args.N / 4 / 50) * 2)
    print('patience_steps = ', patience_steps)
    all_step_info, step_info_map = get_all_setps() 
    
    seq_no = 0
    best_info = all_step_info[0]
    best_seq = 0
    actual_try_steps = 0    
    for step_info in all_step_info[1:]:
        seq_no += 1
        C_1 = step_info['metric']['p@1'] > best_info['metric']['p@1']
        C_2 = (step_info['metric']['p@1'] == best_info['metric']['p@1']) and (step_info['metric']['p@5'] > best_info['metric']['p@5']) 
        if seq_no - best_seq > patience_steps:
            print('stop step = %d' % step_info['step'])
            break
        if C_1 or C_2:
            actual_try_steps = seq_no - best_seq
            best_info = step_info
            best_seq = seq_no
            

    print('actual_try_seqs = ', actual_try_steps)
    
    print('updated best info \n ', best_info)

    print('previous best info \n', step_info_map[best_step])
     
    assert(best_info == step_info_map[best_step]) 
    print('ok')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    verify_best_step(args) 

if __name__ == '__main__':
    main()
