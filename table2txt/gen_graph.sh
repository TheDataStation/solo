if [ "$#" -ne 2 ]; then
    echo "Usage: ./gen_graph.sh <dataset> <experiment>"
    exit
fi
dataset=$1
table_file=tables.jsonl
exptr=$3
stg=TemplateGraph
python ./table2graph.py \
--dataset ${dataset} \
--table_file ${table_file} \
--experiment ${exptr} \
--strategy ${stg}
