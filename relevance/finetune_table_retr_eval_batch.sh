for (( step = 3000; step <= 24000; step+=3000))
do
    ./finetune_table_retr_eval.sh ${step}
done
