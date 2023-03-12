full_file_name=./data/wiki.jsonl
out_file_name=./data/wiki_embeddings
python generate_passage_embeddings.py \
        --model_path ~/code/table_discovery_project/models/tqa_retriever \
        --passages ${full_file_name} \
        --output_path ${out_file_name} \
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 100 \
