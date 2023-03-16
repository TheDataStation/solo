full_file_name=./data/wiki.jsonl
out_file_name=./data/wiki_embeddings_student_float16/embeddings
python generate_passage_embeddings.py \
    --is_student 1 \
    --model_path ~/code/table_discovery_project/models/student_tqa_retriever_epoch_16 \
    --passages ${full_file_name} \
    --output_path ${out_file_name} \
    --shard_id 0 \
    --num_shards 1 \
    --per_gpu_batch_size 1000 \
    --output_batch_size 1000000 \
