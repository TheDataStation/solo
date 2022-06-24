if [ "$#" -ne 3 ]; then
  echo "Usage: ./eval_bm25_syt.sh <dataset> <expr> <sql_expr>"
  exit
fi
dataset=$1
expr=$2
sql_expr=$3
python eval_bm25.py \
--index_name ${dataset}_${expr} \
--dataset ${dataset} \
--expr ${expr} \
--sql_expr ${sql_expr} \
--synthetic 1 \
