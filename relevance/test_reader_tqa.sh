python test_reader.py \
        --model_path ./pretrained_models/tqa_reader_base \
        --eval_data ~/data/qa_zre_data/data/open_qa/dev/qa_zre_fushion_input.jsonl \
        --per_gpu_batch_size 1 \
        --n_context 1000 \
        --name qa_zre_dev_100 \
        --checkpoint_dir output \

