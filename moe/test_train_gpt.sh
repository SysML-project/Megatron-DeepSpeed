#!/bin/bash

# Check if a configuration file path was provided as an argument
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 path_to_ds_config [path_to_host_config]"
    exit 1
fi

DS_CONFIG=$1

if [ $# -eq 2 ]; then
    source $2
    HOST_IP_ADDRESS=$(hostname -I | awk '{print $1}')
    HOST_ARGS="--hostfile=${HOST_FILE} --ssh_port=2222 --master_addr ${HOST_IP_ADDRESS}"
else
    HOST_ARGS=""
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1

TRAINING_STEPS=100000

# Create a new directory for this run
job_name=gpt125M
unique_id=${job_name}_$(date +%Y%m%d%H%M%S)_$$
log_path=output/test/${unique_id}

mkdir -p $log_path

new_ds_config=${log_path}/ds_config.json

# Modify the ds config to point to this
jq --arg logDir "$log_path" \
    --arg jobName "$job_name" \
    '.tensorboard.output_path = $logDir |
     .tensorboard.job_name = $jobName |
     .csv_monitor.output_path = $logDir |
     .csv_monitor.job_name = $jobName' \
    "$DS_CONFIG" > $new_ds_config

echo "JOB ID ${unique_id}"
echo "LOGGING METRICS TO ${log_path}"
echo "USING DS CONFIG ${new_ds_config}"

# Copy the config file to all remote notes
if [[ -n $HOST_FILE ]]; then
    while IFS= read -r line; do
        SERVER_IP=$(echo $line | awk '{print $1}')
        ssh -p 2222 deepspeed@${SERVER_IP} "mkdir -p /workspace/Megatron-DeepSpeed/moe/${log_path}" < /dev/null
        if [ $? -ne 0 ]; then
            echo "Failed to create directory on $SERVER_IP."
            exit
        fi
        scp -P 2222 ${new_ds_config} deepspeed@${SERVER_IP}:/workspace/Megatron-DeepSpeed/moe/${log_path} < /dev/null
        if [ $? -ne 0 ]; then
            echo "Failed to copy config to $SERVER_IP."
            exit
        fi
    done < ${HOST_FILE}
fi
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
--data-path ../datasets/wikitext/wikitext103_text_document \
--vocab-file ../datasets/gpt2-vocab.json \
--merge-file ../datasets/gpt2-merges.txt \
--make-vocab-size-divisible-by 1024 \
--data-impl mmap \
--split 949,50,1"

COMPUTE_ARGS="\
--fp16 \
--DDP-impl local"

OUTPUT_ARGS="\
--log-interval 1 \
--save-interval 2000 \
--save ./checkpoints \
--eval-interval 1000 \
--eval-iters 100"

DS_ARGS="\
--deepspeed \
--zero-stage 0 \
--deepspeed_config /workspace/Megatron-DeepSpeed/moe/${new_ds_config}
"
deepspeed $HOST_ARGS ../pretrain_gpt.py \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $DATA_ARGS \
    $COMPUTE_ARGS \
    $OUTPUT_ARGS \
    $DS_ARGS \
    --exit-interval 500 2>&1 | tee ${log_path}/output.log
