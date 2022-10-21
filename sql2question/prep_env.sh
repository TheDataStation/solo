python3.7 -m venv ../pyenv/plms_graph2text
source ../pyenv/plms_graph2text/bin/activate
pip install -r ./requirements.txt
python -m spacy download en_core_web_sm
cd transformers
pip install -e .
