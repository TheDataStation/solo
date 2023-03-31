if [ "$#" -ne 3 ]; then
    echo "Usage: ./eval.sh <dataset> <epoch> <step>"
    exit
fi
dataset=$1
epoch=$2
step=$3
data_file=~/code/table_discovery_project/data/nq_tables/query/test/rel_graph/fusion_retrieved_tagged.jsonl
python ./finetune_table_retr.py \
    --model_path  ~/code/table_discovery_project/models/tqa_reader_base \
    --fusion_retr_model ~/code/table_discovery_project/models/nq_tables/sql_0_epoc_1_step_814_model_3_29_2023.pt \
    --eval_data ${data_file} \
    --n_context 1000 \
    --per_gpu_batch_size 1 \
    --cuda 0 \
    --name test_${dataset}_${epoch}_step_${step} \
    --checkpoint_dir output \
    --bnn 1 \
