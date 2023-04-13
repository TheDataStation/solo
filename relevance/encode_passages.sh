full_file_name=../../open_table_discovery/table2txt/dataset/nq_tables/rel_graph/passages.jsonl
out_file_name=./data/wiki_emb_student_nq_tables/embeddings
python generate_passage_embeddings.py \
    --is_student 1 \
    --model_path ~/code/table_discovery_project/models/student_tqa_retriever_step_29500 \
    --passages ${full_file_name} \
    --output_path ${out_file_name} \
    --shard_id 0 \
    --num_shards 1 \
    --per_gpu_batch_size 3600 \
    --output_batch_size 500000 \
