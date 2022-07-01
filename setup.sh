sudo apt-get update
sudo apt-get -y install libssl-dev libffi-dev zlib1g-dev libsqlite3-dev gcc g++ make sqlite3 liblzma-dev libbz2-dev
mkdir -p downloads/python_install
cur_dir=${PWD}
cd downloads/python_install
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
tar -xvzf Python-3.7.9.tgz
cd Python-3.7.9
sudo ./configure --enable-loadable-sqlite-extensions
sudo make install
cd ${cur_dir}
sudo rm -rf downloads
cd ..
git clone git@github.com:mingqiwangcn/fusion_in_decoder.git
cd fusion_in_decoder
bash prep_env.sh
cd ..
git clone git@github.com:mingqiwangcn/plms_graph2text.git
cd plms_graph2text
bash prep_env.sh

