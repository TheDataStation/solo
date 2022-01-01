if [ "$#" -ne 3 ]; then
    echo "Usage: ./eval_graph_bm25_dev.sh <debug> <dataset> <experiment>"
    exit
fi
debug=$1
if [ $debug = "1" ]; then
    opt="-m pdb"
else
    opt=""
fi
dataset=$2
exptr=$3
index_name=${dataset}_${exptr}
out_dir=./dataset/nq_tables/bm25_${dataset}_${exptr}/dev
python ${opt} eval_bm25.py \
--index_name ${index_name} \
--mode dev \
--out_dir ${out_dir}
