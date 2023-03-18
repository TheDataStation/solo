conda create -y --name s2ld python=3.7.9
conda activate s2ld
conda install -y -c pytorch faiss-gpu
pip install --upgrade pip
pip install -r ./requirements.txt
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
sudo apt update
sudo apt install -y wget gcc-8 unzip libssl1.0.0 software-properties-common
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt-get install -y --only-upgrade libstdc++6

cd ./relevance/transformers-3.0.2
pip install -e .
cd ../../

