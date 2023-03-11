if [ "$#" -ne 1 ]; then
    echo "Usage: ./train_student.sh <out_dir>"
    exit
fi

python train_student_retriever.py \
--lr 1e-4 \
--optim adamw \
--scheduler linear \
--teacher_model_path /home/cc/code/table_discovery_project/models/tqa_retriever \
--train_data ./data/trivia_qa/train.jsonl \
--eval_data ./data/trivia_qa/dev.jsonl \
--n_context 100 \
--total_steps 20000 \
--scheduler_steps 30000 \
