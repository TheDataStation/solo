if [ "$#" -ne 2 ]; then
    echo "Usage: ./finetune_table_retr.py <dataset> <experiment>"
    exit
fi
dataset=$1
exprt=$2
exprt_dir=/home/cc/code/open_table_discovery/table2txt/dataset/${dataset}/${exprt}
chk_name=train_${dataset}_${exprt} 
python ./finetune_table_retr.py \
    --do_train \
    --model_path ./pretrained_models/tqa_reader_base \
    --train_data ${exprt_dir}/fusion_retrieved_train.jsonl \
    --eval_data ${exprt_dir}/fusion_retrieved_dev.jsonl \
    --n_context 100 \
    --per_gpu_batch_size 1 \
    --cuda 0 \
    --name ${chk_name} \
    --checkpoint_dir output \
    --checkpoint_steps 1000 \

