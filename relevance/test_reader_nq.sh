python test_reader.py \
        --model_path ./pretrained_models/nq_reader_large \
        --eval_data ./open_domain_data/NQ/test.json \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name nq_test \
        --checkpoint_dir output \

