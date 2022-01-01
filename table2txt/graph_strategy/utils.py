import json
import re
from tqdm import tqdm
from webnlg.data.template_data import TemplateTag

def get_tag_span(start_tag, end_tag, template_text, template_meta=None):
    pos_1 = template_text.find(start_tag)
    pos_2 = template_text.find(end_tag)
    if pos_1 < 0 or pos_2 < 0:
        #str_msg = '%s or %s not found' % (start_tag, end_tag)
        #print(str_msg)
        return None

    span_start = pos_1
    span_end = pos_2 + len(end_tag)
    span = (span_start, span_end)
    return span 

def read_template(template_text, template_meta=None):
    span_info_lst = []
    sub_span = get_tag_span(TemplateTag.Subject_Start, TemplateTag.Subject_End, template_text, template_meta)
    obj_span = get_tag_span(TemplateTag.Object_Start, TemplateTag.Object_End, template_text, template_meta)
    
    if (sub_span is None) or (obj_span is None):
        return None

    sub_first = sub_span[1] < obj_span[0]
    obj_first = obj_span[1] < sub_span[0]
    
    if not (sub_first or obj_first):
        return None
   
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
    if right_most_ent_pos + 1 < len(template_text):
        span_info_3 = {
            'is_template':False,
            'span':[right_most_ent_pos + 1, len(template_text) - 1]
        }
        span_info_lst.append(span_info_3)

    return span_info_lst
   
