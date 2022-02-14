if [ "$#" -ne 3 ]; then
  echo "Usage: ./eval_bm25_syt.sh <mode> <dataset> <experiment>"
  exit
fi
mode=$1
dataset=$2
expr=$3
python eval_bm25.py \
--index_name ${dataset}_${expr} \
--mode ${mode} \
--dataset ${dataset} \
--expr ${expr} \
--synthetic 1 \
--out_dir ./dataset/${dataset}/bm25_syt_${expr}/${mode}
