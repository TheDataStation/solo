import json
import sys
import numpy as np
from xml.dom import minidom
import glob
from pathlib import Path
import re
import unidecode
import os
from fuzzywuzzy import fuzz 
import spacy
from tqdm import tqdm
from template_data import TemplateTag 

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    d = [m.group(0) for m in matches]
    new_d = []
    for token in d:
        token = token.replace('(', '')
        token_split = token.split('_')
        for t in token_split:
            #new_d.append(t.lower())
            new_d.append(t)
    return new_d

def get_nodes(n):
    n = n.strip()
    n = n.replace('(', '')
    n = n.replace('\"', '')
    n = n.replace(')', '')
    #n = n.replace(',', ' , ')
    n = n.replace('_', ' ')
    n = n.replace('-', ' ')
    #n = ' '.join(re.split('(\W)', n))
    n = unidecode.unidecode(n)
    #n = n.lower()
    return n

def get_relation(n):
    n = n.replace('(', '')
    n = n.replace(')', '')
    n = n.strip()
    n = n.split()
    n = "_".join(n)
    return n

def process_comma(text):
    char_array = [] 
    for idx, ch in enumerate(text):
        if ch == ',':
            if text[idx - 1].isdigit() and text[idx + 1].isdigit():
                continue
            else:
                char_array.append(' ')
                char_array.append(',')
                char_array.append(' ')
        else:
            char_array.append(ch)

    text_2 = ''.join(char_array)

    return text_2

def text_preprocess(text):
    text_1 = process_comma(text)

    text_1 = text_1.replace('(', ' ')
    text_1 = text_1.replace(')', ' ')

    tokens = text_1.split()
    input_text = ' '.join(tokens)
    doc = nlp(input_text)
    new_text = ' '.join([a.text for a in doc])
    return new_text

def process_triples(mtriples):
    rel_triple_lst = [] 
    for m in mtriples:
        ms = text_preprocess(m.firstChild.nodeValue)
        ms = ms.strip().split(' | ')
        assert(len(ms)) == 3
        n1 = ms[0]
        n2 = ms[2]
        nodes1 = get_nodes(n1)
        nodes2 = get_nodes(n2)

        edge = get_relation(ms[1])

        edge_split = camel_case_split(edge)
        edges = ' '.join(edge_split)
        
        sub = {}
        sub['tokens'] = nodes1.split()
        sub['role'] = 'sub' 

        rel = {}
        rel['tokens'] = edges.split()
        rel['role'] = 'rel'

        obj = {}
        obj['tokens'] = nodes2.split()
        obj['role'] = 'obj'

        triple_info = {
            'sub': sub,
            'rel': rel,
            'obj': obj
        }
        rel_triple_lst.append(triple_info)

    nodes, ent_info_dict = triple2node(rel_triple_lst)
    return nodes, ent_info_dict


def triple2node(rel_triple_lst):
    nodes = []
    ent_idx = 1
    ent_info_dict = {}
    nodes.append(TemplateTag.Para_Start)
    for triple in rel_triple_lst:
        nodes.append(TemplateTag.Triple_Start)
        for key in ['sub', 'obj']:
            ent_src_text = ' '.join(triple[key]['tokens'])
            ent_text = ent_src_text.lower()
            if ent_text not in ent_info_dict:
                ent_info_dict[ent_text] = {'text':ent_src_text, 'entity_id':ent_idx, 'role':triple[key]['role'] }
                ent_idx += 1
            #triple[key]['idx'] = ent_info_dict[ent_text]['ent_idx']
            #tag_idx = triple[key]['idx']
            #triple[key]['start_tag'] = '[E%d]' % tag_idx
            #triple[key]['end_tag'] = '[/E%d]' % tag_idx
        
        sub = triple['sub']
        sub_toks = []
        sub_toks.append(TemplateTag.Subject_Start)
        sub_toks.extend(sub['tokens'])
        sub_toks.append(TemplateTag.Subject_End)

        rel = triple['rel']
        rel_toks = []
        rel_toks.append(TemplateTag.Rel_Start)
        rel_toks.extend(rel['tokens'])
        rel_toks.append(TemplateTag.Rel_End)
        
        obj = triple['obj']
        obj_toks = []
        obj_toks.append(TemplateTag.Object_Start)
        obj_toks.extend(obj['tokens'])
        obj_toks.append(TemplateTag.Object_End)
       
        nodes.extend(sub_toks)
        nodes.extend(rel_toks)
        nodes.extend(obj_toks)

        nodes.append(TemplateTag.Triple_End)
        
    nodes.append(TemplateTag.Para_End)
    return nodes, ent_info_dict 

Ent_Not_found = 0
Ent_Found = 0

def get_data_dev_test(file_, train_cat, dataset):
    datapoints = []
    cats = set()

    xmldoc = minidom.parse(file_)
    entries = xmldoc.getElementsByTagName('entry')
    cont = 0
    for e in entries:
        cat = e.getAttribute('category')
        cats.add(cat)

        if cat not in train_cat and dataset == 'test_seen':
            continue

        # if cat in train_cat and dataset == 'test_unseen':
        #     continue

        mtriples = e.getElementsByTagName('mtriple')
        nodes, ent_info_dict = process_triples(mtriples)

        lexs = e.getElementsByTagName('lex')

        surfaces = []
        for l in lexs:
            #l = l.firstChild.nodeValue.strip().lower()
            l_text = l.firstChild.nodeValue.strip()
            updated_l_text = text_preprocess(l_text) 
            template = get_template_output(updated_l_text, ent_info_dict)
            if template is None:
                continue
            new_doc = ' '.join(re.split('(\W)', updated_l_text))
            new_doc = ' '.join(new_doc.split())
            # new_doc = tokenizer.tokenize(new_doc)
            # new_doc = ' '.join(new_doc)
            surfaces.append((template, template.lower()))
        
        if len(surfaces) == 0:
            #print('No template found')
            global Ent_Not_found
            Ent_Not_found += 1
            continue

        global Ent_Found
        Ent_Found += 1
        cont += 1
        datapoints.append((nodes, surfaces))

    return datapoints, cats, cont

def get_data(file_):
    datapoints = []
    cats = set()

    xmldoc = minidom.parse(file_)
    entries = xmldoc.getElementsByTagName('entry')
    cont = 0
    for e in entries:
        cat = e.getAttribute('category')
        cats.add(cat)

        cont += 1

        mtriples = e.getElementsByTagName('mtriple')
        nodes, ent_info_dict = process_triples(mtriples)

        lexs = e.getElementsByTagName('lex')

        for l in lexs:
            #l = l.firstChild.nodeValue.strip().lower()
            l_text = l.firstChild.nodeValue.strip()
            updated_l_text = text_preprocess(l_text)
            template = get_template_output(updated_l_text, ent_info_dict)
            if template is None:
                continue
            new_doc = ' '.join(re.split('(\W)', updated_l_text))
            new_doc = ' '.join(template.split())
            #new_doc = tokenizer.tokenize(new_doc)
            #new_doc = ' '.join(new_doc)
            datapoints.append((nodes, (template, template.lower())))

    return datapoints, cats, cont

def str_is_number(text):
    return all([a == '.' or a.isdigit() for a in text])

def fuzzy_match_entity(ent_text, target_text):
    idx = -1
    span_len = 0
    if str_is_number(ent_text):
        new_ent_text = str(int(float(ent_text)))
        idx = target_text.find(new_ent_text)
        span_len = len(new_ent_text)
    else:
        ent_token_lst = ent_text.split()
        if ',' in ent_token_lst:
            updated_ent_text_1 = ent_text.replace(',', ' ')
            updated_tokens = updated_ent_text_1.split()
            updated_ent_text_2 = ' '.join(updated_tokens)
            idx = target_text.find(updated_ent_text_2)
            if idx > -1:
                span_len = len(updated_ent_text_2)  
                return idx, span_len

            ent_token_lst = updated_ent_text_2.split()

        target_token_lst = target_text.split()
        step = len(ent_token_lst)
        N = len(target_token_lst)
        pos = 0
        while (pos + step <= N):
            sub_target_text = ' '.join(target_token_lst[pos:(pos+step)])
            score = fuzz.ratio(ent_text, sub_target_text)
            if score >= 85:
                idx = target_text.index(sub_target_text)
                span_len = len(sub_target_text)
                break
            pos += 1
         
    return idx, span_len

def get_entity_tags(role):
    if role == 'sub':
        return (TemplateTag.Subject_Start, TemplateTag.Subject_End)
    elif role == 'obj':
        return (TemplateTag.Object_Start, TemplateTag.Object_End)
    else:
        raise ValueError('No supported role [%s]' % role)

def get_template_output(target_text, ent_info_dict):
    out_info_dict = {}
    out_text = target_text

    ent_text_lst =[(ent, len(ent)) for ent in ent_info_dict]
    ent_text_lst = sorted(ent_text_lst, key=lambda x: x[1], reverse=True)
    for ent_text, _ in ent_text_lst: # for each entity in a triple, find the match in target text 
        out_text_lower = out_text.lower()
        role = ent_info_dict[ent_text]['role']
        ent_idx = ent_info_dict[ent_text]['entity_id'] 
        idx = out_text_lower.find(ent_text)
        span_len = len(ent_text) 
        if idx == -1:
            idx, span_len = fuzzy_match_entity(ent_text, out_text_lower)
            if idx == -1:
                return None

        out_ent = out_text[idx:(idx+span_len)]
        ent_to_replace = out_ent
         
        template_text = '[e_n_t_%d][/e_n_t_%d]' % (ent_idx, ent_idx)
        start_tag, end_tag = get_entity_tags(role)
        out_info_dict[template_text] = '%s %s %s' % (start_tag, ent_info_dict[ent_text]['text'], end_tag)
        to_replaced = re.compile(re.escape(ent_to_replace), re.IGNORECASE)
        #replace the entity in target (out) text with template text 
        out_text = to_replaced.sub(template_text, out_text)
    
    for template_text in out_info_dict:
        out_text = out_text.replace(template_text, out_info_dict[template_text])
    return out_text

def main():
    folder = sys.argv[1]
    datasets = ['train', 'dev', 'test_both', 'test_seen', 'test_unseen']
    
    global nlp
    nlp = spacy.load("en_core_web_sm")

    train_cat = set()
    dataset_points = []
    for d in tqdm(datasets):
        cont_all = 0
        datapoints = []
        all_cats = set()
        if 'unseen' in d:
            d_set = 'test'
            files = [folder + '/' + d_set + '/testdata_unseen_with_lex.xml']
        elif 'test' in d:
            d_set = 'test'
            files = [folder + '/' + d_set + '/testdata_with_lex.xml']
        else:
            files = Path(folder + '/' + d).rglob('*.xml')

        files = sorted(list(files))

        for idx, filename in tqdm(enumerate(files), total=len(files)):
            #print(filename)
            filename = str(filename)

            if d == 'train':
                datapoint, cats, cont = get_data(filename)
            else:
                datapoint, cats, cont = get_data_dev_test(filename, train_cat, d)
            cont_all += cont
            all_cats.update(cats)
            datapoints.extend(datapoint)
        if d == 'train':
            train_cat = all_cats
        print(d, len(datapoints))
        print('cont', cont_all)
        print('len cat', len(all_cats))
        print('cat', all_cats)
        dataset_points.append(datapoints)


    path = os.path.dirname(os.path.realpath(__file__)) + '/webnlg/'
    if not os.path.exists(path):
        os.makedirs(path)

    os.system("rm " + path + '/*')

    for idx, datapoints in enumerate(dataset_points):

        part = datasets[idx]

        if part == 'dev':
            part = 'val'

        nodes = []
        surfaces = []
        surfaces_2 = []
        surfaces_3 = []

        surfaces_eval = []
        surfaces_2_eval = []
        surfaces_3_eval = []
        for datapoint in datapoints:
            node = datapoint[0]
            sur = datapoint[1]
            nodes.append(' '.join(node))
            if part != 'train':
                surfaces.append(sur[0][0])
                surfaces_eval.append(sur[0][1])
                if len(sur) > 1:
                    surfaces_2.append(sur[1][0])
                    surfaces_2_eval.append(sur[1][1])
                else:
                    surfaces_2.append('')
                    surfaces_2_eval.append('')
                if len(sur) > 2:
                    surfaces_3.append(sur[2][0])
                    surfaces_3_eval.append(sur[2][1])
                else:
                    surfaces_3.append('')
                    surfaces_3_eval.append('')
            else:
                surfaces.append(sur[0])
                surfaces_eval.append(sur[1])

        with open(path + '/' + part + '.source', 'w', encoding='utf8') as f:
            f.write('\n'.join(nodes))
            f.write('\n')
        with open(path + '/' + part + '.target', 'w', encoding='utf8') as f:
            f.write('\n'.join(surfaces))
            f.write('\n')
        if part != 'train':
            with open(path + '/' + part + '.target2', 'w', encoding='utf8') as f:
                f.write('\n'.join(surfaces_2))
                f.write('\n')
            with open(path + '/' + part + '.target3', 'w', encoding='utf8') as f:
                f.write('\n'.join(surfaces_3))
                f.write('\n')

        with open(path + '/' + part + '.target_eval', 'w', encoding='utf8') as f:
            f.write('\n'.join(surfaces_eval))
            f.write('\n')
        if part != 'train':
            with open(path + '/' + part + '.target2_eval', 'w', encoding='utf8') as f:
                f.write('\n'.join(surfaces_2_eval))
                f.write('\n')
            with open(path + '/' + part + '.target3_eval', 'w', encoding='utf8') as f:
                f.write('\n'.join(surfaces_3_eval))
                f.write('\n')

    print('Ent_Not_found %d, Ent_Found %d' % (Ent_Not_found, Ent_Found))


if __name__ == '__main__':
    main()

