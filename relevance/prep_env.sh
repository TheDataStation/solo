python3.7 -m venv ../pyenv/fusion_decoder
source ../pyenv/fusion_decoder/bin/activate
pip install -r ./requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
cd ./transformers-3.0.2
pip install -e .
