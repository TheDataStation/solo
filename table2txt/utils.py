import json
import re


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

def read_template(template_text):
    span_info_lst = []
  
    start_lst = re.findall(r'\[E\d+\]', template_text)
    end_lst = re.findall(r'\[/E\d+\]', template_text)
     
