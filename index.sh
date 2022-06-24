if [ "$#" -ne 1 ]; then
  echo "Usage: ./index.sh <dataset>"
  exit
fi
dataset=$1
work_dir="$(dirname "$PWD")"
echo ${work_dir}
source ../pyenv/fusion_decoder/bin/activate
python index_tables.py \
--work_dir ${work_dir} \
--dataset ${dataset} \
