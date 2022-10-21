python test_reader.py \
        --model_path ./pretrained_models/qa_zre_forward_reader_data_1_percent/checkpoint/best_dev \
        --eval_data ./open_domain_data/QA_ZRE/dev_100.jsonl \
        --per_gpu_batch_size 1 \
        --n_context 50 \
        --name qa_zre_forward_dev_100 \
        --checkpoint_dir output \
        --write_crossattention_scores

