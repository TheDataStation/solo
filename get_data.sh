if [ "$#" -ne 1 ]; then
  echo "Usage: ./get_data.sh <dataset>"
  exit
fi
dataset=$1
work_dir="$(dirname "$PWD")"
data_dir=${work_dir}/data
mkdir ${data_dir}
cd ${data_dir}
wget http://plover.cs.uchicago.edu/open_table_discovery/${dataset}.tar
rm -rf ${dataset}
tar -xf ${dataset}.tar
rm ${dataset}.tar
cd ${dataset}
mkdir -p index
cd index
index_dir=on_disk_index_${dataset}_rel_graph
rm -rf ${index_dir}
mkdir ${index_dir}
cd ${index_dir}
wget http://plover.cs.uchicago.edu/open_table_discovery/${index_dir}/merged_index.ivfdata
wget http://plover.cs.uchicago.edu/open_table_discovery/${index_dir}/passages.jsonl
wget http://plover.cs.uchicago.edu/open_table_discovery/${index_dir}/populated.index


