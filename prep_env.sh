python3.7 -m venv ../pyenv/s2ld
source ../pyenv/s2ld/bin/activate
pip install -r ./requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
cd ./relevance/transformers-3.0.2
pip install -e .
