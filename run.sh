#!/bin/bash

# llama-2-7b-hf
python dcis.py --model meta-llama/Llama-2-7b-hf \
    --model-name llama \
    --tokenized pg19-validation-tokenized \
    --dataset-min-tokens 131072 \
    --samples 1 \
    --factor 16 \
    --init-type yarn \
    --original-max-position-embeddings 4096 \
    --itercnt 10

# mistral-7b-v0.1
python dcis.py --model mistralai/Mistral-7B-v0.1 \
    --model-name mistral \
    --tokenized pg19-validation-tokenized-mistral \
    --dataset-min-tokens 131072 \
    --samples 1 \
    --factor 8 \
    --init-type yarn \
    --original-max-position-embeddings 8192 \
    --beta-fast 128 \
    --beta-slow 2 \
    --itercnt 10

# eval w/o finetuning, --model-name llama-dcis w/ finetuning
python eval/eval_perplexity.py \
    --tokenized proofpile-test-tokenized \
    --dataset-min-tokens 131072 \
    --samples 10 \
    --min-tokens 4096 \
    --max-tokens 65536 \
    --tokens-step 4096  \
    --model meta-llama/Llama-2-7b-hf \
    --model-name llama \
    --factor 16 \
    --factors ./factors/llama/16.0_yarn_0.pt \
    --output-file testdata/proofpile.csv

python eval/eval_passkey.py \
    --min-tokens 4096 \
    --max-tokens 65536 \
    --tokens-step 4096 \
    --iterations 10  \
    --model meta-llama/Llama-2-7b-hf \
    --model-name llama \
    --factor 16 \
    --factors ./factors/llama/16.0_yarn_0.pt \
    --output-file testdata/passkey.csv

# finetune
accelerate launch --num_processes 4 finetune.py \
    --model meta-llama/Llama-2-7b-hf \
    --output-dir ./model/llama2-7b-64k-dcis-t16k \
    --max-train-steps 400 \
    --batch-size 2 \
    --gradient-accumulate-every 8 \
    --scaling-factor 16 \
    --scaling-factors ./factors/llama/16.0_yarn_0.pt \
    --deepspeed \
    --dataset pg_books-tokenized-bos-eos-chunked-65536-truncated-16384
