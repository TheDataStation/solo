if [ "$#" -ne 1 ]; then
  echo "Usage: ./index_emb.sh <dataset>"
  exit
fi
work_dir="$(dirname "$PWD")"
dataset=$1
export PYTHONPATH=${work_dir}/open_table_discovery:${work_dir}/open_table_discovery/relevance:${work_dir}/open_table_discovery/sql2question

python index_tables.py \
--work_dir ${work_dir} \
--dataset ${dataset} \
--pipe_step emb_to_index \
