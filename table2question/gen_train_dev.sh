if [ "$#" -ne 1 ]; then
  echo "Usage: ./gen_train_dev.py.sh <dataset>"
  exit
fi
dataset=$1
python ./gen_train_dev.py --dataset ${dataset} --sql_expr auto_sql
