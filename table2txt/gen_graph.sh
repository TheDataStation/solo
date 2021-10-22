if [ "$#" -ne 3 ]; then
    echo "Usage: ./gen_graph.sh <dataset> <experiment> <strategy>"
    exit
fi
dataset=$1
exptr=$2
stg=$3
python ./table2graph.py \
--dataset ${dataset} \
--table_file tables_500 \
--experiment ${exptr} \
--strategy ${stg}
