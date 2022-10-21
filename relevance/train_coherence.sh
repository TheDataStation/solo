python train_coherence.py \
        --train_data ./open_domain_data/QA_ZRE/train_1_per.jsonl \
        --eval_data ./open_domain_data/QA_ZRE/dev_100.jsonl \
        --f_reader_model_path ./pretrained_models/qa_zre_f_reader_1_per_p_10/checkpoint/best_dev \
        --b_reader_model_path ./pretrained_models/qa_zre_b_reader_1_per/checkpoint/best_dev \
        --per_gpu_batch_size 1 \
        --n_context 10 \
        --name coherence_train_data_1_percent_p_10 \
        --checkpoint_dir output \
        --total_steps 30000 \
        --eval_freq 500 \
        --save_freq 500 \

