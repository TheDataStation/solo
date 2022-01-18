import json
import csv
from tqdm import tqdm

def read_data(data_file):
    data = []
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            data.append(item)
    return data

def main():
    data_file = './dataset/nq_tables/triple_template_graph/graph_passages.json'
    out_dir = './dataset/nq_tables/triple_template_graph/passage_encode'
    
    data = read_data(data_file)
    num_parts = 3
    N = len(data)
    bsz = (N // num_parts) + (1 if (N % num_parts) > 0 else 0)
  
    passage_id = 1
    part_no = 1
    for idx in range(0, N, bsz):
        part_data = data[idx:(idx+bsz)]
        out_file = out_dir + '/passage_part_%d.tsv' % part_no
        part_no += 1
        with open(out_file, 'w') as f_o:
            csv_writer = csv.writer(f_o, delimiter='\t')
            csv_writer.writerow(['id', 'text', 'title'])
            for item in tqdm(part_data):
                passage = item['passage']
                title = ''
                csv_writer.writerow([passage_id, passage, title])
                passage_id += 1


if __name__ == '__main__':
    main()
