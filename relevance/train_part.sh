if [ "$#" -ne 5 ]; then
    echo "Usage: ./finetune_syt_retr_.sh <dataset> <part_no> <bnn> <prior> <repre>"
    exit
fi
dataset=$1
sql_expr=sql_data
train_itr=train_0
part_no=$2
bnn=$3
prior=$4
exprt=$5
exprt_dir=/home/cc/code/open_table_discovery/table2question/dataset/${dataset}/${sql_expr}
chk_name=${dataset}_${part_no}_bnn_${bnn}
if [ "${prior}" != "none" ]; then
    chk_name=${chk_name}_prior
fi
train_file=${exprt_dir}/${train_itr}/${exprt}/data_parts/${part_no}.jsonl
eval_file_name=fusion_retrieved_tagged.jsonl
if [ "${exprt}" = "graph_text" ]; then
   eval_file_name=fusion_retrieved_tagged_merged.jsonl 
fi
eval_file=${exprt_dir}/dev/${exprt}/${eval_file_name}

python ./finetune_table_retr.py \
    --do_train \
    --model_path ~/code/models/tqa_reader_base \
    --train_data ${train_file} \
    --eval_data ${eval_file} \
    --n_context 100 \
    --per_gpu_batch_size 4 \
    --cuda 0 \
    --name ${chk_name} \
    --checkpoint_dir output \
    --max_epoch 20 \
    --question_maxlength 50 \
    --text_maxlength 300 \
    --bnn ${bnn} \
    --prior_model ${prior} \
    --per_gpu_eval_batch_size 1
