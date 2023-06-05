#!/bin/bash

set -ex

WORKING_DIRECTORY=/tmp/apply-awq-to-llama
LLAMA_DIRECTORY=/tmp/apply-awq-to-llama/llama-model/

#rm -rf "$WORKING_DIRECTORY"
mkdir -p "$WORKING_DIRECTORY"
mkdir -p "$LLAMA_DIRECTORY"

#rm -rf "$LLAMA_DIRECTORY/llama-7b"

# Download llama 7b
( cd "$LLAMA_DIRECTORY"
if [ ! -d "llama-7b" ];
then
    # forget git lfs it sucks
    # git lfs install
    # git clone git@hf.co:huggyllama/llama-7b.git

    mkdir llama-7b
    cd llama-7b
    wget https://huggingface.co/huggyllama/llama-7b/blob/main/config.json
    wget https://huggingface.co/huggyllama/llama-7b/blob/main/generation_config.json
    wget https://huggingface.co/huggyllama/llama-7b/blob/main/model.safetensors.index.json
    wget https://huggingface.co/huggyllama/llama-7b/blob/main/special_tokens_map.json
    wget https://huggingface.co/huggyllama/llama-7b/blob/main/tokenizer.json
    wget https://huggingface.co/huggyllama/llama-7b/blob/main/tokenizer.model
    wget https://huggingface.co/huggyllama/llama-7b/blob/main/tokenizer_config.json
    wget https://huggingface.co/huggyllama/llama-7b/blob/main/model-00001-of-00002.safetensors
    wget https://huggingface.co/huggyllama/llama-7b/blob/main/model-00002-of-00002.safetensors
else
    echo already got llama-7b
fi )

# Download the AWQ code
( cd "$WORKING_DIRECTORY"
if [ ! -d "llm-awq" ];
then
    git clone --depth 1 https://github.com/mit-han-lab/llm-awq.git
    git checkout 3a6dfc39ed20d793f7c26624c4b9f9599960dd3b
else
    echo already got llm-awq
fi )

# Setup deps
cd "$WORKING_DIRECTORY/llm-awq"

# Don't bother to create a conda env for this
#conda create -n awq python=3.10 -y
#conda activate awq
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
( cd awq/kernels
python setup.py install --user
)

# Perform the conversion

mkdir -p "$WORKING_DIRECTORY/awq_cache"
mkdir -p "$WORKING_DIRECTORY/quant_cache"

# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path "$LLAMA_DIRECTORY/llama-7b/" \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq "$WORKING_DIRECTORY/awq_cache/$MODEL-w4-g128.pt"

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path "$LLAMA_DIRECTORY/llama-7b/" \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq "$WORKING_DIRECTORY/awq_cache/$MODEL-w4-g128.pt" \
    --q_backend fake

# generate real quantized weights (w4)
python -m awq.entry --model_path "$LLAMA_DIRECTORY/llama-7b/" \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w4-g128.pt \
    --q_backend real --dump_quant "$WORKING_DIRECTORY/quant_cache/$MODEL-w4-g128-awq.pt"

# load and evaluate the real quantized model (smaller gpu memory usage)
python -m awq.entry --model_path "$LLAMA_DIRECTORY/llama-7b/" \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_quant "$WORKING_DIRECTORY/quant_cache/$MODEL-w4-g128-awq.pt"
