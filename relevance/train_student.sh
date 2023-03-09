python train_student_retriever.py \
        --lr 1e-4 \
        --optim adamw \
        --scheduler linear \
        --train_data train_data.json \
        --eval_data eval_data.json \
        --n_context 100 \
        --total_steps 20000 \
        --scheduler_steps 30000 \

