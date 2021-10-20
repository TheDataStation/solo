if [ "$#" -ne 2 ]; then
    echo "Usage: ./main.sh <debug> <example>"
    exit
fi
debug=$1
example=$2
if [ $debug = "1" ]; then
    opt="-m pdb"
else
    opt=""
fi
export OMP_NUM_THREADS=10
export CUDA_VISIBLE_DEVICES=0
python -W ignore ${opt} main.py \
--data_tag ${example} \
--input_tables /home/cc/data/nq_tables/tables/${example}.jsonl \
--dataset_in_dir=/home/cc/code/plms_graph2text/webnlg/data/webnlg/nq_tables/${example} \
--data_dir=default \
--task graph2text \
--model_name_or_path=t5-base \
--eval_batch_size=8 \
--gpus 1 \
--dataset_out_dir=/home/cc/code/plms_graph2text/webnlg/outputs/test_model/nq_tables/${example} \
--output_dir=default \
--checkpoint=/home/cc/code/plms_graph2text/models/webnlg-t5-base.ckpt \
--max_source_length=384 \
--max_target_length=384 \
--val_max_target_length=384 \
--test_max_target_length=384 \
--eval_max_gen_length=384 \
--do_predict \
--eval_beams 3 \
--show_progress_bar 1
