full_file_name=~/code/table_discovery_project/open_table_discovery/table2txt/dataset/nq_tables/rel_graph/passages.jsonl
out_file_name=./data/wiki_embeddings_teacher_float16/embeddings
python generate_passage_embeddings.py \
    --is_student 0 \
    --model_path ~/code/table_discovery_project/models/tqa_retriever \
    --passages ${full_file_name} \
    --output_path ${out_file_name} \
    --shard_id 0 \
    --num_shards 1 \
    --per_gpu_batch_size 3600 \
    --output_batch_size 500000 \
