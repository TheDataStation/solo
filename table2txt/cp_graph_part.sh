if [ "$#" -ne 3 ]; then
    echo "Usage: ./gen_graph_parts.sh <dataset> <experiment> <graph part>"
    exit
fi
dataset=$1
expt_name=$2
part_name=$3
expt_dir="/home/cc/code/plms_graph2text/webnlg/data/${dataset}/${expt_name}"
mkdir -p ${expt_dir}
rm -rf ${expt_dir}/${part_name}
cp -r ~/code/plms_graph2text/webnlg/data/template ${expt_dir}/${part_name}
graph_parts_dir=./dataset/${dataset}/${expt_name}/graph_parts
cp ${graph_parts_dir}/test_unseen.source_${part_name} ${expt_dir}/${part_name}/test_unseen.source
cp ${graph_parts_dir}/test_unseen.target_${part_name} ${expt_dir}/${part_name}/test_unseen.target
echo "${expt_dir}/${part_name}"
