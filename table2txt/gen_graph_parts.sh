if [ "$#" -ne 3 ]; then
    echo "Usage: ./gen_graph_parts.sh <dataset> <experiment> <part num>"
    exit
fi
dataset=$1
experiment=$2
part_num=$3
graph_dir=./dataset/${dataset}/${experiment}
graph_parts_dir=${graph_dir}/graph_parts
rm -rf ${graph_parts_dir}
mkdir ${graph_parts_dir}
row_table_file=${graph_dir}/graph_row_table.txt
total_lines=`wc -l < ${row_table_file}`
chunk_size=$(( ( ${total_lines} + ${part_num} - 1 ) / ${part_num} ))
split -l ${chunk_size} ${graph_dir}/test_unseen.source ${graph_parts_dir}/test_unseen.source_part_
split -l ${chunk_size} ${graph_dir}/test_unseen.target ${graph_parts_dir}/test_unseen.target_part_
split -l ${chunk_size} ${graph_dir}/graph_row_table.txt ${graph_parts_dir}/graph_row_table.txt_part_

