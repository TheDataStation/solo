if [ "$#" -ne 1 ]; then
  echo "Usage: ./get_data.sh <dataset>"
  exit
fi
dataset=$1
work_dir="$(dirname "$PWD")"
data_dir=${work_dir}/data
mkdir -p ${data_dir}
cd ${data_dir}
rm -f ${dataset}.tar
wget https://storage.googleapis.com/open_data_123/${dataset}_data/${dataset}.tar
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
wget https://storage.googleapis.com/open_data_123/${dataset}_data/${index_dir}/merged_index.ivfdata
rm -f passages.jsonl
wget https://storage.googleapis.com/open_data_123/${dataset}_data/${index_dir}/passages.jsonl
rm -f populated.index
wget https://storage.googleapis.com/open_data_123/${dataset}_data/${index_dir}/populated.index
index_store_dir=$PWD
index_create_dir=/home/cc/code/table_discovery_project/index/${index_dir}
mkdir -p ${index_create_dir}
cd ${index_create_dir}
ln -s ${index_store_dir}/merged_index.ivfdata