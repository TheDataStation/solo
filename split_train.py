import json
import random

def read_data():
    data = []
    data_file = './fusion_retrieved_tagged.jsonl'
    with open(data_file) as f:
        for line in f:
            data.append(line)
    random.shuffle(data)
    return data

def main():
    data = read_data()
    bsz = 1000
    N = len(data)
    part_no = 0
    for offset in range(0, N, bsz):
        part_no += 1
        pos = offset + bsz
        data_part = data[offset:pos]
        out_file = './data_parts/part_%d.jsonl' % part_no
        with open(out_file, 'w') as f_o:
            for line in data_part:
                f_o.write(line)
    print('ok')
if __name__ == '__main__':
    main()
