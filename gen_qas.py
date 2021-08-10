import json
import os
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    q_info_dict = {}
    with open(args.input) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['table']['tableId']
            q_data = item['questions']
            for q_item in q_data:
                qid = q_item['id']
                question = q_item['originalText']
                
                if qid not in q_info_dict:
                    q_info_dict[qid] = {
                        'qid': qid,
                        'question': question,
                        'table_id_lst':[],
                        'answers':[]
                    }
                table_id_lst = q_info_dict[qid]['table_id_lst']
                table_id_lst.append(table_id)
                
                answer_lst = q_item['answer']['answerTexts']
                alt_answer_lst = q_item.get('alternativeAnswers', [])
                for alt_answer_info in alt_answer_lst:
                    answer_lst.extend(alt_answer_info['answerTexts'])
                answer_lst = list(set(answer_lst))
                q_info_dict[qid]['answers'].extend(answer_lst)

    with open(args.output, 'w') as f_o:
        for qid in q_info_dict:
            answer_text_lst = q_info_dict[qid]['answers']
            q_info_dict[qid]['answers'] = list(set(answer_text_lst))
            f_o.write(json.dumps(q_info_dict[qid]) + '\n')
            
if __name__ == '__main__':
    main()
