python3.7 -m venv ../../pyenv/sql2question
source ../../pyenv/sql2question/bin/activate
pip install -r ./requirements.txt
python -m spacy download en_core_web_sm
cd transformers
pip install -e .
