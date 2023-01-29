if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_demo.sh <dataset>"
    exit
fi
dataset=$1
work_dir="$(dirname "$PWD")"
#ln -s ${work_dir}/index data
#ln -s ${work_dir}/open_table_discovery/system.config
export PYTHONPATH=${work_dir}/open_table_discovery:${work_dir}/open_table_discovery/relevance:${work_dir}/open_table_discovery/sql2question
source ../pyenv/s2ld/bin/activate
python ./demo/app.py --work_dir ${work_dir} --dataset ${dataset}

