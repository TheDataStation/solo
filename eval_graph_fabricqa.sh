if [ "$#" -ne 3 ]; then
    echo "Usage: ./eval_graph_fabricqa.sh <mode> <dataset> <experiment>"
    exit
fi
mode=$1
dataset=$2
exptr=$3
query_dir=~/data/${dataset}/interactions
index_name=${dataset}_${exptr}
out_dir=./dataset/${dataset}/${exptr}/${mode}
python eval_graph_fabricqa.py \
--query_dir ${query_dir} \
--index_name ${index_name} \
--mode ${mode} \
--out_dir ${out_dir} 
