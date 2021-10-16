if [ "$#" -ne 2 ]; then
    echo "Usage: ./gen_graph.sh <dataset> <experiment>"
    exit
fi
dataset=$1
exptr=$2
python ./table2graph.py \
--dataset ${dataset} \
--table_filter table_id_small.jsonl \
--experiment ${exptr}
