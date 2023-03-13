import csv
import json
from tqdm import tqdm

def main():
    f_o = open('../data/wiki.jsonl', 'w')
    with open('/home/cc/code/catalog/dpr/downloads/data/wikipedia_split/psgs_w100.tsv') as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader):
            if row[0] == 'id':
                continue
            sample_id = row[0]
            passage = row[1].strip('"')
            
            out_item = {
                'p_id':int(sample_id),
                'passage':passage
            }
            f_o.write(json.dumps(out_item) + '\n')
    f_o.close()

if __name__ == '__main__':
    main()
