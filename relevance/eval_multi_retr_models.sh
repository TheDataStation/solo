if [ "$#" -ne 3 ]; then
    echo "Usage: ./eval_multi_retr_models.sh <dataset> <part_no> <multi_model_dir>"
    exit
fi
dataset=$1
part_no=$2
multi_model_dir=$3

python ./finetune_table_retr.py \
    --multi_model_eval 1 \
    --multi_model_dir ${multi_model_dir} \
    --model_path ~/code/models/tqa_reader_base \
    --eval_data ~/code/open_table_discovery/table2question/dataset/${dataset}/sql_data/train_0/rel_graph/data_parts/${part_no}.jsonl \
    --n_context 100 \
    --per_gpu_eval_batch_size 4 \
    --cuda 0 \
    --name ${dataset}_multi_models_${part_no} \
    --checkpoint_dir output \
    --text_maxlength 300 \
    --bnn 1
