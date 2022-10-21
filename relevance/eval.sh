if [ "$#" -ne 3 ]; then
    echo "Usage: ./eval.sh <dataset> <epoch> <step>"
    exit
fi
dataset=$1
epoch=$2
step=$3
data_file=~/code/data/${dataset}/query/test/rel_graph/fusion_retrieved_tagged.jsonl
python ./finetune_table_retr.py \
    --model_path ./pretrained_models/tqa_reader_base \
    --fusion_retr_model ./output/train_syt_${dataset}_rel_graph_auto_sql_updated_tagged/epoc_${epoch}_step_${step}_model.pt \
    --eval_data ${data_file} \
    --n_context 100 \
    --per_gpu_batch_size 1 \
    --cuda 0 \
    --name test_${dataset}_${epoch}_step_${step} \
    --checkpoint_dir output \
    --bnn 0 \
