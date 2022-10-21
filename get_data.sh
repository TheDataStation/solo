if [ "$#" -ne 2 ]; then
  echo "Usage: ./get_data.sh <dataset> <server>"
  exit
fi
dataset=$1
server=$2
work_dir="$(dirname "$PWD")"
data_dir=${work_dir}/data
mkdir -p ${data_dir}
cd ${data_dir}
rm -f ${dataset}.tar
wget http://${server}/open_table_discovery/${dataset}.tar
rm -rf ${dataset}
tar -xf ${dataset}.tar
rm ${dataset}.tar
cd ${work_dir}
mkdir -p index
cd index
index_dir=on_disk_index_${dataset}_rel_graph
rm -rf ${index_dir}
mkdir ${index_dir}
cd ${index_dir}
rm -f merged_index.ivfdata
wget http://${server}/open_table_discovery/${index_dir}/merged_index.ivfdata
rm -f passages.jsonl
wget http://${server}/open_table_discovery/${index_dir}/passages.jsonl
rm -f populated.index
wget http://${server}/open_table_discovery/${index_dir}/populated.index


