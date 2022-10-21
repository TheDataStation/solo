if [ "$#" -ne 3 ]; then
    echo "Usage: ./encode_passages.sh <dataset> <experiment> <passage_file_name>"
    exit
fi

dataset=$1
exptr=$2
file_name=$3
exptr_dir=/home/cc/code/open_table_discovery/table2txt/dataset/${dataset}/${exptr}
full_file_name=${exptr_dir}/${file_name}
out_file_name=${exptr_dir}/${file_name}_embeddings
python generate_passage_embeddings.py \
        --model_path ~/code/models//tqa_retriever \
        --passages ${full_file_name} \
        --output_path ${out_file_name} \
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 3000 \
