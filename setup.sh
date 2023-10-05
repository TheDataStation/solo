cur_dir=${PWD}
work_dir="$(dirname "$PWD")"
source prep_env.sh
cd ${cur_dir}/sql2question
bash prep_env.sh
cd ${work_dir}
mkdir -p index
cd ${cur_dir}
ln -s ${work_dir}/index data
cd ${work_dir}
if [ ! -d "open_table_discovery" ]; then
    ln -s solo open_table_discovery
fi
