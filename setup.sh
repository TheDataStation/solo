sudo apt-get update
sudo apt-get -y install libssl-dev libffi-dev zlib1g-dev libsqlite3-dev gcc g++ make sqlite3 liblzma-dev libbz2-dev
mkdir -p downloads/python_install
cur_dir=${PWD}
work_dir="$(dirname "$PWD")"
cd downloads/python_install
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
tar -xvzf Python-3.7.9.tgz
cd Python-3.7.9
sudo ./configure --enable-loadable-sqlite-extensions
sudo make install
cd ${cur_dir}
sudo rm -rf downloads

bash prep_env.sh
cd sql2question
bash prep_env.sh
cd ${work_dir}
mkdir -p index
cd ${cur_dir}
ln -s ${work_dir}/index data
sudo apt-get -y install jupyter-notebook
