import json
import re
from tqdm import tqdm
from webnlg.data.template_data import TemplateTag

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

def get_tag_span(start_tag, end_tag, template_text):
    pos_1 = template_text.find(start_tag)
    pos_2 = template_text.find(end_tag)
    if pos_1 < 0 or pos_2 < 0:
        raise ValueError('%s or %s not found' % (start_tag, end_tag))

    span_start = pos_1
    span_end = pos_2 + len(end_tag)
    span = (span_start, span_end)
    return span 

def read_template(table_data, meta_info, template_text):
    span_info_lst = []
    sub_span = get_tag_span(TemplateTag.Subject_Start, TemplateTag.Subject_End, template_text)
    obj_span = get_tag_span(TemplateTag.Object_Start, TemplateTag.Object_End, template_text)
    sub_first = sub_span[1] < obj_span[0]
    obj_first = obj_span[1] < sub_span[0]
    assert(sub_first or obj_first)
   
    sub_span_info = {
        'is_template':True,
        'span':sub_span,
        'type':'sub'
    }

    obj_span_info = {
        'is_template':True,
        'span':obj_span,
        'type':'obj'
    }
    
    ent_span_lst = []
    if sub_first:
        ent_span_lst = [sub_span_info, obj_span_info]
    else:
        ent_span_lst = [obj_span_info, sub_span_info]
   
    left_most_ent_pos = ent_span_lst[0]['span'][0] 
    if left_most_ent_pos > 0:
        span_info_1 = {
            'is_template':False,
            'span':[0, left_most_ent_pos-1]
        }
        # span from 0 to left most entity
        span_info_lst.append(span_info_1)
    # span of left most entity
    span_info_lst.append(ent_span_lst[0])
    # span of the region between two entities  
    pos_1 = ent_span_lst[0]['span'][1] + 1
    pos_2 = ent_span_lst[1]['span'][0] - 1
    if pos_1 <= pos_2:
        span_info_2 = {
            'is_template':False,
            'span':[pos_1, pos_2]
        }
        span_info_lst.append(span_info_2)
    # span of the right most entity
    span_info_lst.append(ent_span_lst[1])
    
    right_most_ent_pos = ent_span_lst[1]['span'][1]
    if right_most_ent_posi + 1 < len(template_text):
        span_info_3 = {
            'is_template':False,
            'span':[right_most_ent_posi + 1, len(template_text) - 1]
        }
        span_info_lst.append(span_info_3)

    return span_info_lst
   
def remove_ent_tags(span_info_lst):
    for span_info in span_info_lst:
        if not span_info['is_template']:
            span_text = span_info['text']
            for tag in [Ent_Start_Tag, Ent_End_Tag, r'\[/E0\]']:
                to_replace = re.compile(tag)
                span_text = to_replace.sub(' ', span_text)
            span_info['text'] = span_text
     
     
