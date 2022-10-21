python test_reader.py \
        --model_path ./output/qa_zre_backward_reader/checkpoint/best_dev \
        --eval_data ./open_domain_data/QA_ZRE/dev.jsonl \
        --per_gpu_batch_size 1 \
        --n_context 50 \
        --name qa_zre_back_dev \
        --checkpoint_dir output \
        --backward 1 \
