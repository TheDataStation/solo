if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_demo.sh <port>"
    exit
fi
port=$1
export CUDA_VISIBLE_DEVICES=0
work_dir="$(dirname "$PWD")"
export PYTHONPATH=${work_dir}/open_table_discovery:${work_dir}/open_table_discovery/relevance:${work_dir}/open_table_discovery/sql2question
source ../pyenv/s2ld/bin/activate
python -m ipykernel install --user --name fusion_decoder
jupyter notebook --no-browser --port=${port}
