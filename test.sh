if [ "$#" -ne 2 ]; then
    echo "Usage: ./test.sh <dataset> <tag>"
    exit
fi
export CUDA_VISIBLE_DEVICES=0
work_dir="$(dirname "$PWD")"
dataset=$1
tag=$2
export PYTHONPATH=${work_dir}/fusion_in_decoder:${work_dir}/open_table_discovery:${work_dir}/plms_graph2text
source ../pyenv/fusion_decoder/bin/activate
python tester.py \
    --work_dir ${work_dir} \
    --dataset ${dataset} \
    --tag ${tag}

