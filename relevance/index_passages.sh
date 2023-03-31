if [ "$#" -ne 3 ]; then
    echo "Usage: ./index_passages.sh <dataset> <experiment> <emb_file>"
    exit
fi

dataset=$1
exptr=$2
emb_file=$3

python ./src/ondisk_index.py \
    --work_dir "/home/cc/code/table_discovery_project" \
    --dataset ${dataset} \
    --experiment ${exptr} \
    --emb_file ${emb_file} \
