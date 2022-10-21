python ./finetune_table_retr.py \
    --model_path ./pretrained_models/tqa_reader_base \
    --fusion_retr_model ./output/train_bm25_nq_tables_table_token_slide/epoc_4_step_24000_model.pt \
    --eval_data /home/cc/code/open_table_discovery/table2txt/dataset/nq_tables/bm25_table_token_slide/test/bm25_fusion_retrieve_test.jsonl \
    --n_context 100 \
    --per_gpu_batch_size 1 \
    --cuda 0 \
    --name bm25_test_nq_tables_table_token_slide_step_24000 \
    --checkpoint_dir output \

