if [ "$#" -ne 2 ]; then
    echo "Usage: ./gen_passages_to_import.sh <dataset> <experiment>"
    exit
fi
dataset=$1
exptr=$2
python ./gen_passages.py \
--dataset ${dataset} \
--experiment ${exptr}
data_dir=./dataset/${dataset}/${exptr}
passage_parts_dir=${data_dir}/passage_parts
rm -rf ${passage_parts_dir}
mkdir ${passage_parts_dir}
out_file_name=graph_passages.json
passage_file=${data_dir}/${out_file_name}
total_lines=`wc -l < ${passage_file}`
part_num=30
part_size=$(( ( ${total_lines} + ${part_num} - 1 ) / ${part_num} ))
split -l ${part_size} ${passage_file} ${passage_parts_dir}/${out_file_name}_part_
