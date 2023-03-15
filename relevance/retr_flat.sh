python passage_retrieval.py \
 --data ./data/TQA/test.json \
 --passages ./data/wiki.jsonl \
 --passages_embeddings ./data/wiki_embeddings_float16_00 \
 --model_path ~/code/table_discovery_project/models/tqa_retriever \
 --output_path output/retr_teacher_float_16.json \

