if [ "$#" -ne 1 ]; then
    echo "Usage: ./import_table_row_text.sh <dataset>"
    exit
fi
dataset=$1
expr=table_row_text

python ~/code/fabric_qa/src/data_process/indexer/data_indexer.py \
--text_type table \
--index_name ${dataset}_${expr} \
--input_dir ./dataset/${dataset}/${expr} \
--file_pattern 'passages.jsonl'

