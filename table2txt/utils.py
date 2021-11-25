import json
import re
from tqdm import tqdm

Tuple_Start_Tag = r'\[T\]'

Ent_Start_Tag = r'\[E\d+\]'

Ent_End_Tag = r'\[/E\d+\]'

Rel_Tag = r'\[R\]'

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

def get_match_lst(pattern, text):
    match_lst = []
    p = re.compile(pattern)
    for m in p.finditer(text):
        span = m.span()
        pos_1 = span[0]
        while pos_1 < span[1]:
            if text[pos_1] == 'E':
                break
            pos_1 += 1
        pos_1 += 1
        pos_2 = span[1] - 1
        ent_idx = int(text[pos_1:pos_2])
        item_info = {
            'ent_idx': ent_idx,
            'span': span,
            'match': m
        }
        match_lst.append(item_info)
    return match_lst

def get_entity_end_pos(start_match, template_text, max_pos, opts):
    pos = start_match['span'][1]
    end_tag = '[/E%d]' % start_match['ent_idx']
    end_tag_len = len(end_tag)
    while pos < max_pos:
        if template_text[pos:(pos+end_tag_len)] != end_tag:
            pos += 1
        else:
            break
    
     
    if '[/E0]' in template_text[start_match['span'][1]:max_pos]:
        opts['E0'] = True
   
    if pos < max_pos:
        end_pos = pos + end_tag_len
        return end_pos 
    else:
        return None

def read_template(table_data, meta_info, template_text):
    start_match_lst = get_match_lst(Ent_Start_Tag, template_text) 
    
    span_info_lst = []
    pos = start_match_lst[0]['span'][0]
    if pos > 0:
        span_text = template_text[:pos]
        span_info = {
            'is_template': False,
            'text': span_text,
            'span': [0, pos]
        }
        span_info_lst.append(span_info)

    for start_idx, start_match in enumerate(start_match_lst):
        if start_idx > 0:
            pos_1 = span_info_lst[-1]['span'][1]
            pos_2 = start_match['span'][0]
            span_text = template_text[pos_1:pos_2]
            span_info = {
                'is_template': False,
                'text': span_text,
                'span': [pos_1, pos_2]
            }
            span_info_lst.append(span_info)
        
        ent_idx = start_match['ent_idx']
        if start_idx < len(start_match_lst) - 1:
            max_pos = start_match_lst[start_idx+1]['span'][0]
        else:
            max_pos = len(template_text)
        
        opts = {} 
        entity_end_pos = get_entity_end_pos(start_match, template_text, max_pos, opts)
        if entity_end_pos is not None:
            span_pos_1 = start_match['span'][0]
            span_pos_2 = entity_end_pos
            span_info = {
                'is_template': True,
                'ent_idx': ent_idx,
                'span': [span_pos_1, span_pos_2]
            }
        else:
            e0_case_1 = (start_match['ent_idx'] == 0)
            e0_case_2 = opts.get('E0', False)
            e_0_case = e0_case_1 or e0_case_2
            if (False):
                print('cannot find end tag for %s at [%d, %d]' % (
                        start_match['match'].group(),
                        start_match['span'][0],
                        start_match['span'][1] 
                        ))
            span_info = {
                'is_template': False,
                'text' : ' ',
                'span': start_match['span']
            }
        span_info_lst.append(span_info)
    
    pos = span_info_lst[-1]['span'][1]
    if pos < len(template_text):
        span_text = template_text[pos:]
        span_info = {
            'is_template': False,
            'text': span_text,
            'span': [pos, len(template_text)]
        }
        span_info_lst.append(span_info)
   
    for span_info in span_info_lst:
        if not span_info['is_template']:
            span_text = span_info['text']
            for tag in [Ent_Start_Tag, Ent_End_Tag, r'\[/E0\]']:
                to_replace = re.compile(tag)
                span_text = to_replace.sub(' ', span_text)
            span_info['text'] = span_text

    return span_info_lst
   
   
     
