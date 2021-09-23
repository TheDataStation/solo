if [ "$#" -ne 3 ]; then
    echo "Usage: ./gen_graph_parts.sh <dataset> <experiment> <chunk size>"
    exit
fi
dataset=$1
experiment=$2
chunk_size=$3
graph_dir=./dataset/${dataset}/${experiment}
graph_parts_dir=${graph_dir}/graph_parts
rm -rf ${graph_parts_dir}
mkdir ${graph_parts_dir}
split -l ${chunk_size} ${graph_dir}/test_unseen.source ${graph_parts_dir}/test_unseen.source_part_
split -l ${chunk_size} ${graph_dir}/test_unseen.target ${graph_parts_dir}/test_unseen.target_part_
split -l ${chunk_size} ${graph_dir}/graph_row_table.txt ${graph_parts_dir}/graph_row_table.txt_part_

