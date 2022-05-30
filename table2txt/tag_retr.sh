if [ "$#" -ne 5 ]; then
    echo "Usage: ./tag_retr_data.py <dataset> <expr> <strategy> <mode> <top_n>"
    exit
fi
dataset=$1
expr=$2
strategy=$3
mode=$4
top_n=$5
python ./tag_retr_data.py \
--dataset ${dataset} \
--expr ${expr} \
--strategy ${strategy} \
--mode ${mode} \
--top_n ${top_n} \
--min_tables 10 \
