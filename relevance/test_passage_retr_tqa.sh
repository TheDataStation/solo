python -m pdb ./test_passage_retr.py \
    --model_path ./pretrained_models/tqa_reader_base \
    --eval_data /home/cc/data/qa_zre_data/data/open_qa/dev/qa_zre_fushion_input.jsonl \
    --per_gpu_batch_size 1 \
    --n_context 500 \
    --name qa_zre_passage_dev \
    --checkpoint_dir output \

