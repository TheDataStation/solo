dataset=nq_tables
expr=tqa_reader
mode=dev
python ./test_table_retr.py \
    --model_path ./pretrained_models/tqa_reader_base \
    --eval_data ./data/retrieved_dev.json \
    --per_gpu_batch_size 1 \
    --n_context 100 \
    --name ${dataset}_${expr}_${mode} \
    --checkpoint_dir output \

