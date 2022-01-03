if [ "$#" -ne 3 ]; then
  echo "Usage: ./eval_graph_fabricqa_test.sh <mode> <dataset> <experiment>"
  exit
fi
mode=$1
dataset=$2
expr=$3
python eval_bm25.py \
--index_name ${dataset}_${expr} \
--mode ${mode} \
--out_dir ./dataset/${dataset}/bm25_${dataset}_${expr}/${mode}
