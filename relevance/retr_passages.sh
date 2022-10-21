if [ "$#" -ne 5 ]; then
    echo "Usage: ./retr_passages_syt.sh <dataset> <table_repre> <mode> <passage> <synthetic>"
    exit
fi
dataset=$1
expr=$2
mode=$3
file_name=$4.jsonl
synthetic=$5
if [ ${synthetic} = "1" ]
then
    sql_expr=sql_data
    sql_data_dir=~/code/open_table_discovery/table2question/dataset/${dataset}/${sql_expr}
    query_file=${sql_data_dir}/${mode}/fusion_query.jsonl
    out_file=${sql_data_dir}/${mode}/${expr}/fusion_retrieved.jsonl
else
    query_file=~/code/data/${dataset}/query/${mode}/fusion_query.jsonl
    out_file=~/code/data/${dataset}/query/${mode}/${expr}/fusion_retrieved.jsonl
fi     
python ./passage_ondisk_retrieval.py \
    --model_path ~/code/models/tqa_retriever \
    --index_file ./data/on_disk_index_${dataset}_${expr}/populated.index \
    --passage_file ./data/on_disk_index_${dataset}_${expr}/${file_name} \
    --data ${query_file} \
    --output_path ${out_file} \
    --n-docs 100 \
    --min_tables 5 \
    --max_retr 1000
