if [ "$#" -ne 1 ]; then
    echo "Usage: ./test.sh <dataset>"
    exit
fi
export CUDA_VISIBLE_DEVICES=0
work_dir="$(dirname "$PWD")"
dataset=$1
bnn=1
table_repre=rel_graph
export PYTHONPATH=${work_dir}/fusion_in_decoder:${work_dir}/open_table_discovery:${work_dir}/plms_graph2text
source ../pyenv/fusion_decoder/bin/activate
python tester.py \
    --work_dir ${work_dir} \
    --dataset ${dataset} \
    --bnn ${bnn} \
    --table_repre ${table_repre} \
