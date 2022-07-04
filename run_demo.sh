if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_demo.sh <port>"
    exit
fi
port=$1
export CUDA_VISIBLE_DEVICES=0
work_dir="$(dirname "$PWD")"
export PYTHONPATH=${work_dir}/fusion_in_decoder:${work_dir}/open_table_discovery:${work_dir}/plms_graph2text
source ../pyenv/fusion_decoder/bin/activate
python -m ipykernel install --user --name fusion_decoder
jupyter notebook --no-browser --port=${port}
