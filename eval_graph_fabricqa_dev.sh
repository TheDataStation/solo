if [ "$#" -ne 2 ]; then
    echo "Usage: ./eval_graph_fabricqa_dev.sh <dataset> <experiment>"
    exit
fi
dataset=$1
exptr=$2
query_dir=~/data/${dataset}/interactions
index_name=${dataset}_${exptr}
out_dir=./dataset/${dataset}/${exptr}/dev
python eval_graph_fabricqa.py \
--query_dir ${query_dir} \
--index_name ${index_name} \
--mode dev \
--out_dir ${out_dir} 
