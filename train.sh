if [ "$#" -ne 1 ]; then
    echo "Usage: ./train.sh <dataset>"
    exit
fi
export CUDA_VISIBLE_DEVICES=0
work_dir="$(dirname "$PWD")"
dataset=$1
export PYTHONPATH=${work_dir}/open_table_discovery:${work_dir}/open_table_discovery/relevance:${work_dir}/open_table_discovery/sql2question
source ../pyenv/s2ld/bin/activate
python trainer.py \
--work_dir ${work_dir} \
--dataset ${dataset} \
