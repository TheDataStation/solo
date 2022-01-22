if [ "$#" -ne 2 ]; then
    echo "Usage: ./gen_query.sh <dataset> <experiment>"
    exit
fi
dataset=$1
expr=$2
python ./gen_fusion_query.py \
--dataset ${dataset} \
--expr ${expr} \
