import json
from tqdm import tqdm
import os

dataset='fetaqa'

def main():
    out_file = './dataset/%s/triple_template_graph/graph_tokens.json' % dataset
    if os.path.exists(out_file):
        print('[%s] already exists' % out_file)
        return

    f_o = open(out_file, 'w')
    delimiters = '   .   '
    with open('./dataset/%s/triple_template_graph/graph_passages.json' % dataset) as f:
        for line in tqdm(f):
            item = json.loads(line)
            passage = item['passage']
            text_parts = passage.split(delimiters)
            assert(len(text_parts) == 3)
            graph_parts = text_parts[:2]
            out_text = delimiters.join(graph_parts)
            item['passage'] = out_text
            f_o.write(json.dumps(item) + '\n')
    f_o.close() 

if __name__ == '__main__':
    main()
