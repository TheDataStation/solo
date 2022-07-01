if [ "$#" -ne 1 ]; then
    echo "Usage: ./train.sh <dataset>"
    exit
fi
CUDA_VISIBLE_DEVICES=0
work_dir="$(dirname "$PWD")"
dataset=$1
export PYTHONPATH=${work_dir}/fusion_in_decoder:${work_dir}/open_table_discovery:${work_dir}/plms_graph2text
source ../pyenv/fusion_decoder/bin/activate
if [ -d "data" ]; then
    unlink data
fi
ln -s ../data/${dataset}/index data
python trainer.py \
--work_dir ${work_dir} \
--dataset ${dataset} \
