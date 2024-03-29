if [ "$#" -ne 6 ]; then
    echo "Usage: ./tag_retr_data.py <dataset> <table_expr> <strategy> <mode> <top_n> <synthetic>"
    exit
fi
dataset=$1
table_expr=$2
strategy=$3
mode=$4
top_n=$5
synthetic=$6
python ./tag_retr_data.py \
--dataset ${dataset} \
--table_expr ${table_expr} \
--strategy ${strategy} \
--mode ${mode} \
--top_n ${top_n} \
--min_tables 5 \
--synthetic ${synthetic} \

