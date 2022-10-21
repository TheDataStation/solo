if [ "$#" -ne 3 ]; then
    echo "Usage: ./index_passages.sh <dataset> <experiment> <emb_file>"
    exit
fi

dataset=$1
exptr=$2
emb_file=$3

python ./src/ondisk_index.py \
    --dataset ${dataset} \
    --experiment ${exptr} \
    --emb_file ${emb_file} \
