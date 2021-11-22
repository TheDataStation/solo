MAX_ENTITY_SIZE = 50
MAX_COL_SIZE = 20
MAX_TUPLE_SIZE = 123
MAX_GRAPH_SIZE = 150

import random
from table2txt.graph_strategy.strategy import Strategy 

class TemplateGraph(Strategy):
    def __init__(self):
        super(TemplateGraph, self).__init__()

    def get_topic_entity(self, table):
        topic_entity = table['documentTitle']
        return topic_entity

    def generate(self, table, num_samples=1):
        topic_entity = self.get_topic_entity(table)
        columns = table['columns']
        col_name_lst = []
        for col_info in columns:
            col_name = col_info['text']
            col_tokens = col_info['text'].split()
            if len(col_tokens) > MAX_COL_SIZE:
                col_name = ' '.join(col_tokens[:MAX_COL_SIZE])
            col_name_lst.append(col_name)

        row_data = table['rows']
        row_idx_lst = [a for a in range(len(row_data))]
        
        actual_sample_num = min(num_samples, len(row_data))
        sample_row_idxes = random.sample(row_idx_lst, actual_sample_num)
        table_graph_lst = []
        for _, row_idx in enumerate(sample_row_idxes):
            row_item = row_data[row_idx]
            cell_lst = row_item['cells']
            row_info = []
            for col_idx, cell in enumerate(cell_lst):
                col_name = col_name_lst[col_idx] 
                cell_text = cell['text']
                cell_token_lst = cell_text.split()
                if len(cell_token_lst) > MAX_ENTITY_SIZE:
                    cell_text = ' '.join(cell_token_lst[:MAX_ENTITY_SIZE])
                cell_info = {
                    'name':col_name,
                    'value':cell_text
                }
                row_info.append(cell_info)
            graph_lst = self.gen_graph_with_topic_entity(row_info, table['documentUrl'], topic_entity)
            graph_info_lst = [{'graph':a, 'row':row_idx} for a in graph_lst]
            table_graph_lst.extend(graph_info_lst)
        return (table, table_graph_lst)

    def get_annotated_triple(self, e_s, sub_ent_idx, rel, e_o, obj_ent_idx):
        tuple_text = '[T] [E%d] %s [/E%d] [R] %s [E%d] %s [/E%d] ' % (sub_ent_idx, e_s, sub_ent_idx,
                                                                      rel,
                                                                      obj_ent_idx, e_o, obj_ent_idx)
        return tuple_text 
        
    def gen_graph_with_topic_entity(self, row_info, url, topic_entity):
        N = len(row_info)
        tuple_dict = {}
        tuple_info_lst = []

        e_s = topic_entity
        sub_ent_idx = 0
        for idx_2 in range(0, N):
            rel = row_info[idx_2]['name'].strip()
            if rel == '':
                continue
            e_o = row_info[idx_2]['value'].strip()
            if e_o == '':
                continue
            
            obj_ent_idx = idx_2 + 1
            tuple_text = self.get_annotated_triple(e_s, sub_ent_idx, rel, e_o, obj_ent_idx)
            tuple_code = tuple_text.lower()
            if tuple_code not in tuple_dict:
                tuple_dict[tuple_code] = 1
                
                token_lst = tuple_text.split()
                tuple_info = {
                    'text':tuple_text,
                    'token_len':len(token_lst),
                    'url':url
                }
                tuple_info_lst.append(tuple_info)
        
        table_tuple_lst = self.get_table_tuples(row_info, url)
        all_tuple_lst = tuple_info_lst + table_tuple_lst

        graph_lst = self.tuple2graph(all_tuple_lst)
        return graph_lst

    def get_table_tuples(self, row_info, url):
        N = len(row_info)
        tuple_dict = {}
        tuple_info_lst = []
        for idx_1 in range(0, N):
            sub_ent_idx = idx_1 + 1
            e_s_class = row_info[idx_1]['name'].strip()
            e_s = row_info[idx_1]['value'].strip()
            if e_s == '':
                continue
            for idx_2 in range(idx_1+1, N):
                obj_ent_idx = idx_2 + 1
                rel = row_info[idx_2]['name'].strip()
                if rel == '':
                    rel = ','
                e_o = row_info[idx_2]['value'].strip()
                if e_o == '':
                    continue
                tuple_text = self.get_annotated_triple(e_s_class + ' ' + e_s, sub_ent_idx, rel, e_o, obj_ent_idx) 
                tuple_code = tuple_text.lower()
                if tuple_code not in tuple_dict:
                    tuple_dict[tuple_code] = 1
                    
                    token_lst = tuple_text.split()
                    tuple_info = {
                        'text':tuple_text,
                        'token_len':len(token_lst),
                        'url':url
                    }
                    tuple_info_lst.append(tuple_info)

        return tuple_info_lst

    def tuple2graph(self, tuple_info_lst):
        graph_lst = []
        buffer_graphs = []
        buffer_size = 0
        N = len(tuple_info_lst)
        stride = 2
        pos_1 = 0
        while (pos_1 < N):
            pos_2 = pos_1
            while (pos_2 < N):
                tuple_info = tuple_info_lst[pos_2]
                tuple_size = tuple_info['token_len']
                if buffer_size + tuple_size <= MAX_GRAPH_SIZE:
                    buffer_graphs.append(tuple_info['text'])
                    buffer_size += tuple_size
                    pos_2 += 1
                else:
                    break

            graph = ''.join(buffer_graphs)
            graph_lst.append(graph)
            buffer_graphs = []
            buffer_size = 0
            if pos_2 >= N:
                break

            next_pos = min(pos_1 + stride, pos_2 - 1)
            if next_pos > pos_1:
                pos_1 = next_pos
            else:
                pos_1 += 1

        return graph_lst


