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
        
        for col_idx, col_info in enumerate(col_data):
            col_info['text'] = col_info['text'].strip()
            col_info['sub_size'] = (len(self.tokenizer.tokenize(col_info['text'])) if col_info['text'] != '' else 0)

        for row_info in tqdm(row_data):
            cell_data = row_info['cells'] 
            for cell_info in cell_data:
                cell_info['text'] = cell_info['text'].strip()
                cell_info['sub_size'] = (len(self.tokenizer.tokenize(cell_info['text'])) if cell_info['text'] != '' else 0)

    def get_topic_entity_rels(self, topic_entity, col_data, row_info, row_idx):
        topic_graph_lst = []
        for col_idx, col_info in enumerate(col_data):
            rel_name = col_info['text']
            obj = row_info['cells'][col_idx]['text'] 
            graph = RelationTag.get_annotated_text(topic_entity, None, None, rel_name, obj) 
            graph_info = {
                'table_id':table['tableId'],
                'row':row_idx,
                'sub_col':None,
                'obj_col':col_idx,
                'graph':graph
            }
            topic_graph_lst.append(graph_info)
        return topic_graph_lst

    def gen_row_rels(self, table):
        topic_entity = self.get_topic_entity(table)
        topic_entity_size = 0
        if topic_entity != '':
            topic_entity_size = len(self.tokenizer.tokenize(topic_entity))

        sep_sub_size = RelationTag.get_sep_sub_size()
        col_data = table['columns']
        N = len(col_data)
        row_data = table['rows']
        for row_idx, row_info in tqdm(enumerate(row_data), total=len(row_data)):
            row_triple_lst = []
            if topic_entity != '':
                topic_triple_lst = self.get_topic_entity_rels(topic_entity, col_data, row_info, row_idx)
                row_triple_lst.extend(topic_triple_lst)

            for sub_col_idx in range(N-1):
                sub_name = col_data[sub_col_idx]['text']
                #sub_name_size = col_data[sub_col_idx]['sub_size']
                
                sub = row_info['cells'][sub_col_idx]['text']
                #sub_text_size = row_info['cells'][sub_col_idx]['sub_size']

                for obj_col_idx in range(sub_col_idx+1, N):
                    rel_name = col_data[obj_col_idx]['text']
                    #rel_name_size = col_data[obj_col_idx]['sub_size']

                    obj = row_info['cells'][obj_col_idx]['text']
                    #obj_text_size = row_info['cells'][obj_col_idx]['sub_size']
                    
                    triple = RelationTag.get_annotated_text(topic_entity, sub_name, sub, rel_name, obj) 
                    
                    triple_size = 20 # len(self.tokenizer.tokenize(triple))
                    triple_info = {
                        'table_id':table['tableId'],
                        'row':row_idx,
                        'sub_col':sub_col_idx,
                        'obj_col':obj_col_idx,
                        'graph':triple,
                        'size':triple_size
                    }
                    row_triple_lst.append(triple_info) 
          
            merged_triple_lst = self.merge_row_rels(row_triple_lst)  
            for merged_triple_info in merged_triple_lst:
                yield merged_triple_info

    def merge_row_rels(self, row_triple_lst):
        merged_triple_lst = []
        buffer_triples = []
        merged_size = 0
        
        for triple_info in row_triple_lst:
            cur_size = triple_info['size']
            if (merged_size + cur_size >= self.max_text_size):
                if len(buffer_triples) > 0:
                    merged_triple_info = self.get_merged_triple(buffer_triples)
                    merged_triple_lst.append(merged_triple_info)
                    buffer_triples = []
                    merged_size = 0

            buffer_triples.append(triple_info)
            merged_size += cur_size
        
        if len(buffer_triples) > 0:
            merged_triple_info = self.get_merged_triple(buffer_triples)
            merged_triple_lst.append(merged_triple_info)

        return merged_triple_lst 

    def get_merged_triple(self, buffer_triples):
        merged_triple_info = {
            'table_id':buffer_triples[0]['table_id'],
            'row':buffer_triples[0]['row'],
            'sub_col_lst':[],
            'obj_col_lst':[],
        }
        sub_col_lst = []
        obj_col_lst = []
        merged_text = ''
        merged_size = 0
        for triple_info in buffer_triples:
            sub_col_lst.append(triple_info['sub_col'])
            obj_col_lst.append(triple_info['obj_col'])
            merged_text += triple_info['graph'] + ' '
            merged_size += triple_info['size']

        merged_triple_info['sub_col_lst'] = sub_col_lst
        merged_triple_info['obj_col_lst'] = obj_col_lst
        merged_triple_info['graph'] = merged_text
        merged_triple_info['size'] = merged_size
        return merged_triple_info
         
    def generate(self, table):
        self.update_cells(table)
        for triple_info in self.gen_row_rels(table):
                yield triple_info

