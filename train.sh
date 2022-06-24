if [ "$#" -ne 1 ]; then
    echo "Usage: ./train.sh <dataset>"
    exit
fi
work_dir="$(dirname "$PWD")"
dataset=$1
source ../pyenv/fusion_decoder/bin/activate
python trainer.py \
--work_dir ${work_dir} \
--datset ${dataset} \
