eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda create -y --name s2ld python=3.7.9
conda activate s2ld
conda install -y -c pytorch faiss-gpu
conda install -y --file ./requirements.txt
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

cd ./relevance/transformers-3.0.2
pip install -e .
cd ../../

