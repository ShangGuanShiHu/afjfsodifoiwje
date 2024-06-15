# git clone https://github.com/ShangGuanShiHu/afjfsodifoiwje.git

bash run.sh

python -m venv .venv

source .venv/bin/activate

pip install -r demo/requirements.txt

cp demo/.env.example demo/.env

echo "environment finishing. plase turn to demo and execute python main.py"