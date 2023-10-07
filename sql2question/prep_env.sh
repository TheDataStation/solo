eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate s2ld
python3.7 -m venv ../../pyenv/sql2question
conda deactivate
source ../../pyenv/sql2question/bin/activate
pip install -r ./requirements.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m spacy download en_core_web_sm
cd transformers
pip install -e .
