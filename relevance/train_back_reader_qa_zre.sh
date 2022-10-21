python train_reader.py \
        --train_data ./open_domain_data/QA_ZRE/train_1_per.jsonl \
        --eval_data ./open_domain_data/QA_ZRE/dev_100.jsonl \
        --model_size base \
        --per_gpu_batch_size 1 \
        --n_context 50 \
        --name qa_zre_backward_reader_data_1_percent \
        --checkpoint_dir output \
        --total_steps 20000 \
        --eval_freq 1000 \
        --save_freq 1000 \
        --backward 1 \

