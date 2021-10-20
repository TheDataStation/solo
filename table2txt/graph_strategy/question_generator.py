from table2txt.graph_strategy.complete_graph import MAX_COL_SIZE, MAX_ENTITY_SIZE

class QG:
    
    def __init__(self, rdm):
        self.rdm = rdm

    def generate(self, table):
        M = 6
        qa_lst = []
        columns = table['columns']
        col_name_lst = []
        for col_info in columns:
            col_name = col_info['text']
            col_tokens = col_info['text'].split()
            if len(col_tokens) > MAX_COL_SIZE:
                col_name = ' '.join(col_tokens[:MAX_COL_SIZE])
            col_name_lst.append(col_name) 
       
        row_data = table['rows']
        for row_idx, row_item in enumerate(row_data):
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
         
            questions = self.gen_row_questions(row_info, table['documentTitle'])
            N = len(questions)
            num_samples = min(M, N)
            question_idx_lst = [a for a in range(N)]
            sample_idxes = self.rdm.sample(question_idx_lst, num_samples)
            sample_questions = [questions[a] for a in sample_idxes]
            
            qa_lst.append(sample_questions)
        return qa_lst
         
    def gen_row_questions(self, row_info, table_title):
        N = len(row_info)
        M = 3
        row_question_lst = []
        for idx_1 in range(0, M):
            e_s_class = row_info[idx_1]['name'].strip()
            e_s = row_info[idx_1]['value'].strip()
            if e_s == '':
                continue
            for idx_2 in range(idx_1+1, N):
                rel = row_info[idx_2]['name'].strip()
                if rel == '':
                    continue
                e_o = row_info[idx_2]['value'].strip()
                if e_o == '':
                    continue
                
                question_1 = {
                    'question':'What is the %s of %s %s ?' % (rel, e_s_class, e_s),
                    'answer':e_o 
                }
                question_2 = {
                    'question':'What is the %s of %s %s ?' % (e_s_class, rel, e_o),
                    'answer':e_o 
                }

                row_question_lst.append(question_1)
                row_question_lst.append(question_2)

        return row_question_lst
