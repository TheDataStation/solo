if [ "$#" -ne 1 ]; then
  echo "Usage: ./index.sh <dataset>"
  exit
fi
export CUDA_VISIBLE_DEVICES=0
work_dir="$(dirname "$PWD")"
dataset=$1
export PYTHONPATH=${work_dir}/open_table_discovery:${work_dir}/open_table_discovery/relevance:${work_dir}/open_table_discovery/sql2question

python index_tables.py \
--work_dir ${work_dir} \
--dataset ${dataset} \
