import json
import re


def read_table_file(table_lst, data_file, table_filter_set):
    with open(data_file) as f:
        for line in tqdm(f):
            table = json.loads(line)
            table_id = table['tableId']
            if table_filter_set is not None:
                if table_id not in table_filter_set:
                    continue
            table_lst.append(table)
    return table_lst

def read_template(template_text):
    span_info_lst = []
   
    start_tag = '[E'
    end_tag = '[/E'
    N = len(template_text)
    pos_1 = 0
        pos_2 = pos_1 + len(start_tag)    
        if template_text[pos_1:pos_2] == start_tag:
            while pos_2 < N and template_text[pos_2] != ']':
                pos_2 += 1 
            ent_idx_text = template_text[(pos_1+2):(pos_2+1) 
            ent_idx = int(ent_idx_text) 
            
            pos_3 = pos_2:
            while template_text[]
