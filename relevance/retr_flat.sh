if [ "$#" -ne 1 ]; then
    echo "Usage: ./retr_flat.sh <tag>"
    exit
fi
tag=$1
python passage_retrieval.py \
 --is_student 1 \
 --data ~/code/table_discovery_project/open_domain_data/TQA/${tag}.json \
 --passages ./data/wiki.jsonl \
 --passages_embeddings "./data/wiki_embeddings_student_float16/embeddings_part_*" \
 --model_path ~/code/table_discovery_project/models/student_tqa_retriever_epoch_16 \
 --output_path ./data/TQA/${tag}_retr_student_float_16.json \

