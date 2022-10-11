if [ "$#" -ne 4 ]; then
    echo "Usage: ./test.sh <dataset> <bnn> <model_dir> <table_repre>"
    exit
fi
export CUDA_VISIBLE_DEVICES=0
work_dir="$(dirname "$PWD")"
dataset=$1
bnn=$2
train_model_dir=~/code/fusion_in_decoder/output/$3
table_repre=$4
export PYTHONPATH=${work_dir}/fusion_in_decoder:${work_dir}/open_table_discovery:${work_dir}/plms_graph2text
source ../pyenv/fusion_decoder/bin/activate
python tester.py \
    --work_dir ${work_dir} \
    --dataset ${dataset} \
    --bnn ${bnn} \
    --train_model_dir ${train_model_dir} \
    --table_repre ${table_repre} \
