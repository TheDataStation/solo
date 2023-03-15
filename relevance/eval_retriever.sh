if [ "$#" -ne 2 ]; then
    echo "Usage: ./eval_retriever.sh <is_student> <model path>"
    exit
fi
is_student=$1
model_path=$2
python train_student_retriever.py \
--is_student ${is_student} \
--model_path ${model_path} \
--eval_data ./data/trivia_qa/dev.jsonl \
--n_context 100 \
--per_gpu_batch_size 32 \
--num_eval_neg_ctxs 15 \
