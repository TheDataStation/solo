if [ "$#" -ne 4 ]; then
  echo "Usage: ./gen_train_dev_retr.sh <dataset> <expr> <sql_expr> <strategy>"
  exit
fi

dataset=$1
expr=$2
sql_expr=$3
strategy=$4

python ./gen_train_dev_retr.py \
--dataset ${dataset} \
--expr ${expr} \
--sql_expr ${sql_expr} \
--strategy ${strategy} \
--top_n_train 200 \
--top_n_dev 200 \
--min_tables 10 \

