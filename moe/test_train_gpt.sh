#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

TRAINING_STEPS=100000

# Model hyperparameters.
MODEL_ARGS="\
--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--seq-length 1024 \
--max-position-embeddings 1024"

# Training hyperparameters.
TRAINING_ARGS="\
--micro-batch-size 4 \
--global-batch-size 64 \
--train-iters ${TRAINING_STEPS} \
--lr-decay-iters ${TRAINING_STEPS} \
--lr 0.00015 \
--min-lr 0.00001 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--init-method-std 0.01"

# Dataset
DATA_ARGS="\
--data-path ../dataset/wikitext103_text_document \
--vocab-file ../dataset/gpt2-vocab.json \
--merge-file ../dataset/gpt2-merges.txt \
--make-vocab-size-divisible-by 1024 \
--data-impl mmap \
--split 949,50,1"

COMPUTE_ARGS="\
--fp16 \
--DDP-impl local"

OUTPUT_ARGS="\
--log-interval 100 \
--save-interval 2000 \
--save ./checkpoints \
--eval-interval 1000 \
--eval-iters 100"

DS_ARGS="\
--deepspeed \
--zero-stage 0 \
--deepspeed_config ds_config.json 
"

deepspeed ../pretrain_gpt.py \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $DATA_ARGS \
    $COMPUTE_ARGS \
    $OUTPUT_ARGS \
    --exit-interval 500 | tee output.log