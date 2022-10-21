if [ "$#" -ne 1 ]; then
  echo "Usage: ./get_data.sh <dataset>"
  exit
fi
dataset=$1
if [ "${dataset}" != "fetaqa" ]; then
    echo "only fetaqa supported right now, please use ./get_data.sh fetaqa"
    exit
fi
server=$2
work_dir="$(dirname "$PWD")"
data_dir=${work_dir}/data
mkdir -p ${data_dir}
cd ${data_dir}
rm -f ${dataset}.tar
wget https://storage.googleapis.com/open_data_123/fetaqa_data/fetaqa.tar
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
wget https://storage.googleapis.com/open_data_123/${dataset}_data/index/merged_index.ivfdata
rm -f passages.jsonl
wget https://storage.googleapis.com/open_data_123/${dataset}_data/index/passages.jsonl
rm -f populated.index
wget https://storage.googleapis.com/open_data_123/${dataset}_data/index/populated.index 

