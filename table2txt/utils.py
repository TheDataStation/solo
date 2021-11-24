import json
import re
from tqdm import tqdm

Tuple_Start_Tag = r'\[T\]'
Ent_Start_Tag = r'\[E\d+\]'
Ent_End_Tag = r'\[/E\d+\]'
Rel_tag = r'\[R\]'

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
    match_dict = {}
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
            'ent_idx':ent_idx,
            'span':span
        }
        match_lst.append(item_info)
        if ent_idx not in match_dict:
            match_dict[ent_idx] = item_info
        else:
            print('%s is duplicate' % m.group())
        
    return match_lst, match_dict

def read_template(template_text):
    start_match_lst, start_match_dict = get_match_lst(Ent_Start_Tag, template_text) 
    end_match_lst, end_match_dict = get_match_lst(Ent_End_Tag, template_text)
    
    span_info_lst = []
    pos = start_match['span'][0]
    if pos > 0:
        span_text = template_text[:pos]
        span_info = {
            'is_template': False,
            'text':span_text
        }
        span_info_lst.append(span_info)

    for start_idx, start_match in enumerate(start_match_lst):
        if start_idx > 0:
            pos_1 = start_match_lst[start_idx-1]['span'][1]
            pos_2 = start_match['span'][0]
            span_text = template_text[pos_1:pos_2]
            span_info = {
                'is_template': False,
                'text': span_text
            }
            span_info_lst.append(span_info)
        
        ent_idx = start_match['ent_idx']
        end_match = end_match_dict.get(ent_idx, None)
        if end_match is not None:
            span_pos_1 = start_match['span'][0]
            span_pos_2 = end_match['span'][1]
            span_info = {
                'is_template': True,
                'ent_idx': ent_idx,
                'span': [span_pos_1, span_pos_2]
            }
        else:
            span_info = {
                'is_template': False,
                'text' : ' '
            }
        span_info_lst.append(span_info)
    
    pos = start_match_lst[-1]['span'][1]
    if pos < len(template_text):
        span_text = template_text[pos:]
        span_info = {
            'is_template': False,
            'text': span_text
        }
        span_info_lst.append(span_info)
   
    for span_info in span_info_lst:
        if not span_info['is_template']:
            span_text = span_info['text']
            for tag in [Ent_Start_Tag, Ent_End_Tag]:
                to_replace = re.compile(tag)
                span_text = to_replace.sub(' ', span_text)
            span_info['text'] = span_text

    return span_info_lst
   
   
     
