#!/bin/bash

# create bin dir
mkdir bin

# install ollama
wget "https://modelscope.cn/api/v1/models/issaccv/OllamaDeploy/repo?Revision=master&FilePath=ollama-linux-amd64" -O bin/ollama

# 设置执行权限
chmod u+x bin/ollama

# run ollama in the background
bin/ollama serve &

# download model 
wget "https://www.modelscope.cn/api/v1/models/qwen/Qwen2-7B-Instruct-GGUF/repo?Revision=master&FilePath=qwen2-7b-instruct-q8_0.gguf" -O qwen-2-7b-instruct.gguf

# create qwen with modelfile
bin/ollama create qwen -f modelfile

# download the embedding model
git clone https://www.modelscope.cn/AI-ModelScope/bge-small-zh-v1.5.git demo/BAAI/bge-base-zh-v1.5

# download the reranker model
git clone https://www.modelscope.cn/quietnight/bge-reranker-large.git demo/BAAI/bge-reranker-large

# download dataset
git clone https://www.modelscope.cn/datasets/issaccv/aiops2024-challenge-dataset.git demo/dataset

mv demo/dataset/question.jsonl demo/question.jsonl

unzip demo/dataset/data.zip -d demo/

# install punkt.zip
mkdir -p /root/nltk_data/tokenizers
unzip -o punkt.zip -d /root/nltk_data/tokenizers/


# print info 
echo "The Ollama server is running in the background. Please close current terminal and you can now play with demo!"

# python -m venv .venv

# pip install -r demo/requirements.txt

# cp demo/.env.example demo/.env

# echo "environment finishing. please source .venv/bin/activate , cd demo and  nohup python main.py &"
