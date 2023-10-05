eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda create -y --name s2ld python=3.7.9
conda activate s2ld
conda install -y -c pytorch faiss-gpu
pip install --upgrade pip
pip install -r ./requirements.txt
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

cd ./relevance/transformers-3.0.2
pip install -e .
cd ../../

