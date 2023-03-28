if [ "$#" -ne 1 ]; then
    echo "Usage: ./retr_flat.sh <tag>"
    exit
fi
tag=$1
python passage_retrieval.py \
 --is_student 1 \
 --data ./data/TQA/${tag}.json \
 --passages ./data/wiki.jsonl \
 --passages_embeddings "./data/wiki_emb_student_19500/embeddings_part_*" \
 --model_path ~/code/table_discovery_project/models/student_tqa_retrieval_step_19500 \
 --output_path ./data/TQA/${tag}_retr_student_19500.json \

