if [ "$#" -ne 2 ]; then
    echo "Usage: ./gen_sql.sh <dataset> <experiment>"
    exit
fi
dataset=$1
table_file=tables.jsonl
exptr=$2
python ./table2sql.py \
--dataset ${dataset} \
--table_file ${table_file} \
--experiment ${exptr} \
