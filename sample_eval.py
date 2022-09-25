import random
import json

def get_data():
    data = []
    with open('./fusion_retrieved_tagged.jsonl') as f:
        for line in f:
            item = json.loads(line)
            if good_item(item): 
                data.append(line)
    return data

def good_item(item):
    table_id_lst = item['table_id_lst']
    ctx_lst = item['ctxs']
    label_lst = [int(a['tag']['table_id'] in table_id_lst) for a in ctx_lst]
    
    if max(label_lst) < 1 or min(label_lst) > 0:
        return False
    return True

def sample_200():
    data = []
    data_file = './fusion_retrieved_tagged_fg.jsonl'
    with open(data_file) as f:
        for line in f:
            data.append(line)

    data_200 = random.sample(data, 200)
    out_file = './200.jsonl'
    with open(out_file, 'w') as f_o:
        for o_line in data_200:
            f_o.write(o_line)

def gen_1000():
    data = get_data()
    print(len(data))
    data_fg = random.sample(data, 1000)
    out_file = './fusion_retrieved_tagged_fg.jsonl'
    with open(out_file, 'w') as f_o:
        for line in data_fg:
            f_o.write(line)


def main():
    gen_1000()
    sample_200()

if __name__ == '__main__':
    main()


