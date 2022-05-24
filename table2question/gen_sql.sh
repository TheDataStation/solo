if [ "$#" -ne 3 ]; then
    echo "Usage: ./gen_sql.sh <dataset> <experiment> <number of train questions>"
    exit
fi
dataset=$1
table_file=tables.jsonl
exptr=$2
num_train=$3
python ./table2sql.py \
--dataset ${dataset} \
--table_file ${table_file} \
--experiment ${exptr} \
--num_train_queries ${num_train} 
