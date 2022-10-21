step=$1
python ./finetune_table_retr.py \
    --model_path ./pretrained_models/tqa_reader_base \
    --fusion_retr_model ./output/train_fetaqa_rel_graph_2/epoc_0_step_${step}_model.pt \
    --eval_data ~/code/open_table_discovery/table2txt/dataset/fetaqa/rel_graph_2/fusion_retrieved_dev.jsonl \
    --n_context 100 \
    --per_gpu_batch_size 1 \
    --cuda 0 \
    --name pred_dev_step_${step}_top_30 \
    --checkpoint_dir output \

