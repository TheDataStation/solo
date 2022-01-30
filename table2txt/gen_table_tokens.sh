if [ "$#" -ne 3 ]; then
    echo "Usage: ./gen_table_tokens.sh <dataset> <table_file> <strategy>"
    exit
fi

dataset=$1
table_file=$2
exptr='table_token_'$3
stg=$3

python ./table2tokens.py \
    --dataset ${dataset} \
    --table_file ${table_file} \
    --experiment ${exptr} \
    --strategy ${stg}
