import random
import numpy as np
from tqdm import tqdm
from table2txt.graph_strategy.strategy import Strategy 
from table2txt.graph_strategy.rel_tags import RelationTag

class RelationGraph(Strategy):
    def __init__(self):
        super(RelationGraph, self).__init__()

    def get_topic_entity(self, table):
        topic_entity = table['documentTitle'].strip()
        return topic_entity
    
    def update_cells(self, table):
        col_data = table['columns']
        row_data = table['rows']
       
        text_size_dict = {} 
        for col_idx, col_info in enumerate(col_data):
            col_token_size = 0
            col_text = col_info['text'].strip()
            if col_text != '':
                if len(col_text.split()) > 1:
                    col_text = '" %s "' % col_text
            
            col_info['text'] = col_text

        for row_info in row_data:
            cell_data = row_info['cells'] 
            for cell_info in cell_data:
                row_cell_text = cell_info['text'].strip()
                row_cell_size = 0
                if row_cell_text != '':
                    if len(row_cell_text.split()) > 1:
                        row_cell_text = '" %s "' % row_cell_text
                
                cell_info['text'] = row_cell_text

    def get_topic_entity_rels(self, table, topic_entity, col_data, row_info, row_idx):
        topic_graph_lst = []
        for col_idx, col_info in enumerate(col_data):
            rel_name = col_info['text']
            obj = row_info['cells'][col_idx]['text']
            triple = RelationTag.get_annotated_text(topic_entity, None, None, rel_name, obj) 
            triple_info = {
                'table_id':table['tableId'],
                'row':row_idx,
                'sub_col':None,
                'obj_col':col_idx,
                'graph':triple,
            }
            topic_graph_lst.append(triple_info)
        return topic_graph_lst

    def gen_row_rels(self, table):
        topic_entity = self.get_topic_entity(table)
        topic_entity_size = 0
        if topic_entity != '':
            if len(topic_entity.split()) > 1:
                topic_entity = '" %s "' % topic_entity

        col_data = table['columns']
        N = len(col_data)
        row_data = table['rows']
        for row_idx, row_info in enumerate(row_data):
            row_triple_lst = []
            if topic_entity != '':
                topic_triple_lst = self.get_topic_entity_rels(table, topic_entity, col_data, row_info, row_idx)
                row_triple_lst.extend(topic_triple_lst)

            for sub_col_idx in range(N-1):
                sub_name = col_data[sub_col_idx]['text']
                sub = row_info['cells'][sub_col_idx]['text']
                for obj_col_idx in range(sub_col_idx+1, N):
                    rel_name = col_data[obj_col_idx]['text']
                    obj = row_info['cells'][obj_col_idx]['text']
                    triple = RelationTag.get_annotated_text(topic_entity, sub_name, sub, rel_name, obj) 
                    triple_info = {
                        'table_id':table['tableId'],
                        'row':row_idx,
                        'sub_col':sub_col_idx,
                        'obj_col':obj_col_idx,
                        'graph':triple,
                    }
                    row_triple_lst.append(triple_info) 
            
            for row_triple in row_triple_lst: 
                yield row_triple 
         
    def generate(self, table):
        self.update_cells(table)
        for triple_info in self.gen_row_rels(table):
                yield triple_info

