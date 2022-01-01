if [ "$#" -ne 2 ]; then
    echo "Usage: ./import_graph_txt.sh <dataset> <experiment>"
    exit
fi
dataset=$1
exptr=$2
index_name=${dataset}_${exptr}
data_dir="./dataset/${dataset}/${exptr}/passage_parts"
python ~/code/fabric_qa/src/data_process/indexer/data_indexer.py \
--text_type table \
--index_name ${index_name} \
--input_dir ${data_dir} \
--file_pattern '*.json_part_*'

