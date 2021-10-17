export OMP_NUM_THREADS=10
export CUDA_VISIBLE_DEVICES=0
python -m pdb ./main.py \
--input_tables /home/cc/data/nq_tables/tables/table_example.jsonl \
--dataset_in_dir=/home/cc/code/plms_graph2text/webnlg/data/webnlg/nq_tables \
--data_dir=default \
--task graph2text \
--model_name_or_path=t5-base \
--eval_batch_size=8 \
--gpus 1 \
--dataset_out_dir=/home/cc/code/plms_graph2text/webnlg/outputs/test_model/nq_tables \
--output_dir=default \
--checkpoint=/home/cc/code/plms_graph2text/models/webnlg-t5-base.ckpt \
--max_source_length=384 \
--max_target_length=384 \
--val_max_target_length=384 \
--test_max_target_length=384 \
--eval_max_gen_length=384 \
--do_predict \
--eval_beams 3 \
