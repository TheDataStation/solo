python passage_retrieval.py \
 --is_student 1 \
 --data ./data/TQA/test.json \
 --passages ./data/wiki.jsonl \
 --passages_embeddings "./data/wiki_embeddings_student_float16/embeddings*" \
 --model_path ~/code/table_discovery_project/models/student_tqa_retriever_epoch_16 \
 --output_path output/retr_student_float_16.json \

