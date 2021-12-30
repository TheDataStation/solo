import random
import numpy as np
from table2txt.graph_strategy.strategy import Strategy 
from webnlg.data.template_data import TemplateTag

class TemplateGraph(Strategy):
    def __init__(self):
        super(TemplateGraph, self).__init__()

    def get_topic_entity(self, table):
        topic_entity = table['documentTitle'].strip()
        return topic_entity
    
    def get_rel_name(self, col_name):
        if col_name == '':
            rel_name = ','
        else:
            rel_name = col_name
        return rel_name

    def get_col_entities(self, table):
        col_entity_lst = []
        column_data = table['columns']
        row_data = table['rows']
        for col_idx, col_info in enumerate(column_data):
            col_name = col_info['text'].strip()
            rel_name = self.get_rel_name(col_name)
            ent_info_lst = []
            for row_idx, row_info in enumerate(row_data):
                ent_text = row_info['cells'][col_idx]['text'].strip()
                ent_tokens = ent_text.split()
                ent_size = len(ent_tokens)
                ent_info = {'text':ent_text, 'size':ent_size, 'row':row_idx}
                ent_info_lst.append(ent_info)
            
            col_entity_info = {
                'col_name':col_name,
                'rel_name':rel_name,
                'entities':ent_info_lst
            }
            col_entity_lst.append(col_entity_info) 
        return col_entity_lst         

    def sample_topic_entity_templates(self, table, col_entity_lst, num_samples=1):
        out_graph_lst = []
        topic_entity = self.get_topic_entity(table)
        if topic_entity == '':
            return []

        column_data = table['columns']
        row_data = table['rows']
        for col_idx, col_info in enumerate(col_entity_lst):
            rel_name = col_info['rel_name'] 
            obj_info_lst = col_info['entities']
            sample_space = self.get_sample_space(obj_info_lst) 
            M = min(len(sample_space), num_samples)
            sample_objs = random.sample(sample_space, M)
            for obj_info in sample_objs:
                obj_entity = obj_info['text']
                if obj_entity == '':
                    continue 
                graph = TemplateTag.get_annotated_triple(topic_entity, rel_name, obj_entity) 
                graph_info = {
                    'table_id':table['tableId'],
                    'row':obj_info['row'],
                    'sub_col':None,
                    'obj_col':col_idx,
                    'graph':graph
                }
                out_graph_lst.append(graph_info)
        return out_graph_lst

    def get_sample_space(self, data):
        item_size_lst = [a['size'] for a in data]  
        q1_pct = np.percentile(item_size_lst, 25)
        q3_pct = np.percentile(item_size_lst, 75)
        sample_space = [a for a in data if (a['size'] > q1_pct) and (a['size'] < q3_pct)] 
        if len(sample_space) == 0:
            sample_space = data
        return sample_space
    
    def get_sample_triples(self, col_entity_lst, sub_col_idx, obj_col_idx, num_samples):
        sub_col_data = col_entity_lst[sub_col_idx]
        sub_col_name = sub_col_data['col_name'] 
        sub_entities = sub_col_data['entities']
        
        obj_col_data = col_entity_lst[obj_col_idx]
        rel_name = obj_col_data['rel_name']
        obj_entities = obj_col_data['entities']
        assert(len(sub_entities) == len(obj_entities))

        triple_info_lst = []
        for idx, sub_ent_item in enumerate(sub_entities):
            obj_ent_item = obj_entities[idx]
            assert(sub_ent_item['row'] == obj_ent_item['row'])
            
            sub_text = sub_ent_item['text']
            obj_text = obj_ent_item['text']
            if sub_text == '' or obj_text == '':
                continue
            triple_info = {
                'sub':(sub_col_name + ' ' + sub_text),
                'rel':rel_name,
                'obj':obj_text,
                'row':sub_ent_item['row'],
                'sub_col':sub_col_idx,
                'obj_col':obj_col_idx,
                'size':(sub_ent_item['size'] + obj_ent_item['size'])
            }
            triple_info_lst.append(triple_info)
        if len(triple_info_lst) == 0:
            return []
        sample_space = self.get_sample_space(triple_info_lst)
        M = min(len(sample_space), num_samples)
        sample_triples = random.sample(sample_space, M)
        return sample_triples 

    def sample_row_templates(self, table, col_entity_lst, num_samples=1):
        out_graph_lst = []
        N = len(col_entity_lst)
        for sub_col_idx in range(N-1):
            for obj_col_idx in range(sub_col_idx+1, N):
                sample_triples = self.get_sample_triples(col_entity_lst, sub_col_idx, obj_col_idx, num_samples)
                for triple_info in sample_triples:
                    graph = TemplateTag.get_annotated_triple(triple_info['sub'],
                                                             triple_info['rel'],
                                                             triple_info['obj'])
                    graph_info = {
                        'table_id':table['tableId'],
                        'row':triple_info['row'],
                        'sub_col':triple_info['sub_col'],
                        'obj_col':triple_info['obj_col'],
                        'graph':graph
                    }
                    out_graph_lst.append(graph_info)
        
        return out_graph_lst

    def generate(self, table, num_samples=1):
        col_entity_lst = self.get_col_entities(table)        
        graph_lst_1 = self.sample_topic_entity_templates(table, col_entity_lst, num_samples=1)
        graph_lst_2 = self.sample_row_templates(table, col_entity_lst, num_samples=1)
        return graph_lst_1 + graph_lst_2



