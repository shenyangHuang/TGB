module load python/3.9
python -m venv $HOME/tgbenv
source $HOME/tgbenv/bin/activate
pip3 install -r requirements.txt
pip3 install -e .


