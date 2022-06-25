if [ "$#" -ne 1 ]; then
  echo "Usage: ./index.sh <dataset>"
  exit
fi
work_dir="$(dirname "$PWD")"
dataset=$1
export PYTHONPATH=${work_dir}/fusion_in_decoder:${work_dir}/open_table_discovery:${work_dir}/plms_graph2text
source ../pyenv/fusion_decoder/bin/activate

python index_tables.py \
--work_dir ${work_dir} \
--dataset ${dataset} \
