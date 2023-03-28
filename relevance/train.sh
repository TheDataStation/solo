if [ "$#" -ne 2 ]; then
    echo "Usage: ./finetune_syt_retr_.sh <dataset> <train itr>"
    exit
fi
dataset=$1
train_itr=$2
chk_name=train_syt_${dataset}_${train_itr}
python ./finetune_table_retr.py \
    --do_train \
    --model_path ~/code/table_discovery_project/models/tqa_reader_base \
    --train_data ~/code/table_discovery_project/open_table_discovery/table2question/dataset/nq_tables/sql_data/train_0/rel_graph/fusion_retrieved_tagged.jsonl \
    --eval_data ~/code/table_discovery_project/open_table_discovery/table2question/dataset/nq_tables/sql_data/dev/rel_graph/fusion_retrieved_tagged.jsonl \
    --per_gpu_batch_size 4 \
    --cuda 0 \
    --name ${chk_name} \
    --checkpoint_dir output \
    --max_epoch 20 \
    --patience_steps 2 \
    --bnn 1\
