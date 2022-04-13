if [ "$#" -ne 3 ]; then
  echo "Usage: ./eval_bm25.sh <mode> <dataset> <expr>"
  exit
fi
mode=$1
dataset=$2
expr=$3
python eval_bm25.py \
--index_name ${dataset}_${expr} \
--mode ${mode} \
--dataset ${dataset} \
--expr ${expr}
