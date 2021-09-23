if [ "$#" -ne 2 ]; then
    echo "Usage: ./gen_passages_to_import.sh <dataset> <experiment>"
    exit
fi
dataset=$1
exptr=$2
python ./gen_passages.py \
--dataset ${dataset} \
--experiment ${exptr}  
