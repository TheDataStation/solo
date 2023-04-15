if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_server.sh <dataset>"
    exit
fi
dataset=$1
work_dir="$(dirname "$PWD")"
export PYTHONPATH=${work_dir}/open_table_discovery:${work_dir}/open_table_discovery/relevance:${work_dir}/open_table_discovery/sql2question
python ./demo/app.py --work_dir ${work_dir} --dataset ${dataset}

