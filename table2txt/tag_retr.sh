if [ "$#" -ne 5 ]; then
    echo "Usage: ./tag_retr_data.py <dataset> <table_expr> <strategy> <mode> <synthetic>"
    exit
fi
dataset=$1
table_expr=$2
strategy=$3
mode=$4
synthetic=$5
python ./tag_retr_data.py \
--dataset ${dataset} \
--table_expr ${table_expr} \
--strategy ${strategy} \
--mode ${mode} \
--synthetic ${synthetic} \
