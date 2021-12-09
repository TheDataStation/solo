import random
import numpy as np
from table2txt.graph_strategy.strategy import Strategy 

class TemplateGraph(Strategy):
    def __init__(self):
        super(TemplateGraph, self).__init__()

    def get_topic_entity(self, table):
        topic_entity = table['documentTitle']
        return topic_entity

    def sample_tempate_rows(self, table, num_samples=1):
        row_data = table['rows']
        row_size_lst = []
        for row_item in row_data:
            cell_lst = row_item['cells']
            row_size = 0
            for cell in cell_lst:
                cell_tokens = cell['text'].split()
                row_size += len(cell_tokens)
            row_size_lst.append(row_size)
        q1_pct = np.percentile(row_size_lst, 25)
        q3_pct = np.percentile(row_size_lst, 75)

        sample_row_pool = []
        for row, row_size in enumerate(row_size_lst):
            if (row_size > q1_pct) and (row_size < q3_pct):
                sample_row_pool.append(row)
        if len(sample_row_pool) == 0:
            sample_row_pool = [row for row, _ in enumerate(row_size_lst)]

        M = min(num_samples, len(sample_row_pool))
        template_rows = random.sample(sample_row_pool, M)
        return template_rows 

    def generate(self, table, num_samples=1):
        topic_entity = self.get_topic_entity(table)
        columns = table['columns']
        col_name_lst = []
        for col_info in columns:
            col_name = col_info['text'].strip()
            col_name_lst.append(col_name)

        row_data = table['rows']
        sample_row_idxes = self.sample_tempate_rows(table, num_samples=num_samples)
        table_graph_lst = []
        for row_idx in sample_row_idxes:
            row_item = row_data[row_idx]
            cell_lst = row_item['cells']
            assert(len(cell_lst) == len(col_name_lst))
            row_info = []
            for col_idx, cell in enumerate(cell_lst):
                col_name = col_name_lst[col_idx] 
                cell_text = cell['text'].strip()
                cell_info = {
                    'name':col_name,
                    'value':cell_text
                }
                row_info.append(cell_info)
            graph_lst = self.gen_graph_with_topic_entity(row_info, topic_entity)
            graph_info_lst = [{'graph':a, 'row':row_idx} for a in graph_lst]
            table_graph_lst.extend(graph_info_lst)
        return (table, table_graph_lst)

    def get_annotated_triple(self, e_s, sub_ent_idx, rel, e_o, obj_ent_idx):
        tuple_text = '[T] [E%d] %s [/E%d] [R] %s [E%d] %s [/E%d] ' % (sub_ent_idx, e_s, sub_ent_idx,
                                                                      rel,
                                                                      obj_ent_idx, e_o, obj_ent_idx)
        return tuple_text 
        
    def gen_graph_with_topic_entity(self, row_info, topic_entity):
        N = len(row_info)
        tuple_dict = {}
        tuple_info_lst = []

        e_s = topic_entity
        sub_ent_idx = 0
        for idx_2 in range(0, N):
            rel = row_info[idx_2]['name']
            if rel == '':
                continue
            e_o = row_info[idx_2]['value']
            if e_o == '':
                continue
            
            obj_ent_idx = idx_2 + 1
            tuple_text = self.get_annotated_triple(e_s, sub_ent_idx, rel, e_o, obj_ent_idx)
            tuple_code = tuple_text.lower()
            if tuple_code not in tuple_dict:
                tuple_dict[tuple_code] = 1
                tuple_info = {
                    'text':tuple_text
                }
                tuple_info_lst.append(tuple_info)
        
        table_tuple_lst = self.get_table_tuples(row_info, tuple_dict)
        all_tuple_lst = tuple_info_lst + table_tuple_lst

        graph_lst = self.tuple2graph(all_tuple_lst)
        return graph_lst

    def get_table_tuples(self, row_info, tuple_dict):
        N = len(row_info)
        tuple_info_lst = []
        for idx_1 in range(0, N):
            sub_ent_idx = idx_1 + 1
            e_s_class = row_info[idx_1]['name']
            e_s = row_info[idx_1]['value']
            if e_s == '':
                continue
            for idx_2 in range(idx_1+1, N):
                obj_ent_idx = idx_2 + 1
                rel = row_info[idx_2]['name']
                if rel == '':
                    rel = ','
                e_o = row_info[idx_2]['value']
                if e_o == '':
                    continue
                tuple_text = self.get_annotated_triple(e_s_class + ' ' + e_s, sub_ent_idx, rel, e_o, obj_ent_idx) 
                tuple_code = tuple_text.lower()
                if tuple_code not in tuple_dict:
                    tuple_dict[tuple_code] = 1
                    token_lst = tuple_text.split()
                    tuple_info = {
                        'text':tuple_text
                    }
                    tuple_info_lst.append(tuple_info)

        return tuple_info_lst

    def tuple2graph(self, tuple_info_lst):
        graph_lst = [info['text'] for info in tuple_info_lst]
        return graph_lst


