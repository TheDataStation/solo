import json
from table2question.sql_data import SqlQuery
from tqdm import tqdm
import re

def read_tables(table_file):
    table_dict = {}
    with open(table_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['id']
            table_dict[table_id] = item
    return table_dict

def process(mode, sql_file, table_file, out_dir):
    out_data = []
    table_dict = read_tables(table_file)
    with open(sql_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['table_id']
            table_info = table_dict[table_id]
            question = item['question']
            sql_info = item['sql'] 
            sql_text = get_sql_text(table_info, sql_info)
            out_item = {
                'sql_text':sql_text,
                'question':question
            }
            out_data.append(out_item)
    
    write_out_data(mode, out_dir, out_data)
     
def write_out_data(mode, out_dir, out_data):
    mode_part_name = mode
    if mode == 'dev':
        mode_part_name = 'val'

    src_file = out_dir + '/' + '%s.source' % mode_part_name
    tgt_file = out_dir + '/' + '%s.target' % mode_part_name
    tgt_eval_file = out_dir + '/' + '%s.target_eval' % mode_part_name
    
    f_o_src = open(src_file, 'w')
    f_o_tgt = open(tgt_file, 'w')
    f_o_tgt_eval = open(tgt_eval_file, 'w')
    
    for out_item in out_data:
        sql_text = out_item['sql_text']
        question = out_item['question']
        f_o_src.write(sql_text + '\n')
        f_o_tgt.write(question + '\n')
        
        eval_question = ' '.join(re.split('(\W)', question))
        eval_question = ' '.join(eval_question.split())
        eval_question = eval_question.lower()
        f_o_tgt_eval.write(eval_question + '\n')
    
    f_o_src.close()
    f_o_tgt.close()
    f_o_tgt_eval.close()     

def get_sql_text(table_info, sql_info):
    sel_col_idx = sql_info['sel']
    agg_op_idx = sql_info['agg']
    conds = sql_info['conds']
    
    col_names = table_info['header']
    sel_col_name = col_names[sel_col_idx]
    
    agg_op = SqlQuery.agg_ops[agg_op_idx]
    if agg_op != '':
        agg_op_tag = SqlQuery.get_src_tag(agg_op)
    else:
        agg_op_tag = ''

    cond_text_lst = []
    for cond_info in conds:
        col_idx, op_idx, cond_value = cond_info
        cond_op = SqlQuery.cond_ops[op_idx]
        cond_op_tag = SqlQuery.get_src_tag(cond_op)
        if col_idx is not None:
            cond_col_name = col_names[col_idx]
            if cond_col_name == '':
                print(sql_info)
            cond_text = '%s %s %s' % (cond_col_name, cond_op_tag, cond_value)
        else:
            cond_text = 'about %s %s' % (cond_op_tag, cond_value)
        cond_text_lst.append(cond_text)
   
    sel_op_tag = SqlQuery.get_src_tag(SqlQuery.sel_op) 
    where_op_tag = SqlQuery.get_src_tag(SqlQuery.where_op)
    and_op_tag = ' ' + SqlQuery.get_src_tag(SqlQuery.and_op) + ' '
    cond_expr = and_op_tag.join(cond_text_lst)
    if agg_op_tag != '':
        sql_text = ' '.join([sel_op_tag, agg_op_tag, sel_col_name, where_op_tag, cond_expr])
    else:
        sql_text = ' '.join([sel_op_tag, sel_col_name, where_op_tag, cond_expr])
    return sql_text

def get_files(mode):
    sql_file = '/home/cc/data/wikisql/%s.jsonl' % mode
    table_file = '/home/cc/data/wikisql/%s.tables.jsonl' % mode
    
    return sql_file, table_file

def main():
    for mode in ['train', 'dev']:
        train_sql_file, train_table_file = get_files(mode)
        process(mode, train_sql_file, train_table_file, './output') 


if __name__ == '__main__':
    main()
