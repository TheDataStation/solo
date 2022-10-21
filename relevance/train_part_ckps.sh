if [ "$#" -ne 3 ]; then
    echo "Usage: ./finetune_syt_retr_.sh <dataset> <part_no> <prior>"
    exit
fi
dataset=$1
exprt=rel_graph
sql_expr=sql_data
train_itr=train_0
part_no=$2
bnn=1
prior=$3
exprt_dir=/home/cc/code/open_table_discovery/table2question/dataset/${dataset}/${sql_expr}
chk_name=${dataset}_ckp_${part_no}_bnn_${bnn}
if [ "${prior}" != "none" ]; then
    chk_name=${chk_name}_prior
fi
train_file=${exprt_dir}/${train_itr}/${exprt}/data_parts/${part_no}.jsonl

python ./finetune_table_retr.py \
    --do_train \
    --model_path ~/code/models/tqa_reader_base \
    --train_data ${train_file} \
    --n_context 100 \
    --per_gpu_batch_size 4 \
    --cuda 0 \
    --name ${chk_name} \
    --checkpoint_dir output \
    --ckp_steps 5 \
    --max_epoch 1 \
    --question_maxlength 50 \
    --text_maxlength 300 \
    --bnn ${bnn} \
    --prior_model ${prior}
