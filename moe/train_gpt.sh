#!/bin/bash
DIR=`pwd`
###############################################################################
### System configs
export PDSH_SSH_ARGS_APPEND="-p 2222"
# Check if a configuration file path was provided as an argument
if [ "$#" -eq 1 ]; then
    echo "Using hostfile at $1"
    source $1
    HOST_IP_ADDRESS=$(hostname -I | awk '{print $1}')
    HOST_ARGS="--hostfile=${HOST_FILE} --ssh_port=2222 --master_addr ${HOST_IP_ADDRESS}"
    NUM_NODE=$(grep -cve '^\s*$' ${HOST_FILE})
else
    HOST_ARGS=""
    NUM_NODE=1
fi

# Set to 1 to log expert selection.
# WARNING: This creates very large CSVs and may slow down
# training
log_expert_selection=0

###############################################################################
### Model configs
## GPT-3 models use 2K sequence length/context window
SEQ_LEN=2048
SEQ_LEN=512

### The "GPT-3 XXX" below are configs from GPT-3 paper
### https://arxiv.org/abs/2005.14165, choose based on
### your desired model size or build your own configs

# **NOTE**: See the later comments regarding LR

## GPT-3 Small 125M
MODEL_SIZE=0.125
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12
GLOBAL_BATCH_SIZE=256

# LR=6.0e-4
# MIN_LR=6.0e-5

## GPT-3 Medium 350M
# MODEL_SIZE=0.35
# NUM_LAYERS=24
# HIDDEN_SIZE=1024
# NUM_ATTN_HEADS=16
# GLOBAL_BATCH_SIZE=256

# LR=3.0e-4
# MIN_LR=3.0e-5

## GPT-3 Large 760M
# MODEL_SIZE=0.76
# NUM_LAYERS=24
# HIDDEN_SIZE=1536
# NUM_ATTN_HEADS=16
# GLOBAL_BATCH_SIZE=256

# LR=2.5e-4
# MIN_LR=2.5e-5

## GPT-3 XL 1.3B
# MODEL_SIZE=1.3
# NUM_LAYERS=24
# HIDDEN_SIZE=2048
# NUM_ATTN_HEADS=16
# GLOBAL_BATCH_SIZE=512

# LR=2.0e-4
# MIN_LR=2.0e-5

## GPT-3 2.7B
# MODEL_SIZE=2.7
# NUM_LAYERS=32
# HIDDEN_SIZE=2560
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=512
# LR=1.6e-4
# MIN_LR=1.6e-5

## GPT-3 6.7B
# MODEL_SIZE=6.7
# NUM_LAYERS=32
# HIDDEN_SIZE=4096
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=1024
# LR=1.2e-4
# MIN_LR=1.2e-5

## GPT-3 13B
# MODEL_SIZE=13
# NUM_LAYERS=40
# HIDDEN_SIZE=5120
# NUM_ATTN_HEADS=40
# GLOBAL_BATCH_SIZE=1024
# LR=1.0e-4
# MIN_LR=1.0e-5

## GPT-3 175B
# MODEL_SIZE=175
# NUM_LAYERS=96
# HIDDEN_SIZE=12288
# NUM_ATTN_HEADS=96
# GLOBAL_BATCH_SIZE=1536
# LR=0.6e-4
# MIN_LR=0.6e-5

###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens
## For MoE model, we found sometimes training a bit more to 330B tokens helps
TRAIN_TOKENS=300000000
# TRAIN_TOKENS=330000000000

## TRAIN_ITERS is another termination condition and also affect the number of
## data samples to be indexed. Since we want to reach the TRAIN_TOKENS
## above, and techniques like curriculum learning has less token in some steps,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by TRAIN_ITERS.
TRAIN_ITERS=$(( ${TRAIN_TOKENS} * 3 / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
EXIT_DURATION=30000000
###############################################################################
### LR configs
## LR warmup and decay duration, this token-based config is preferable since
## no need to readjust when the batch size/seqlen is changed.
## Original GPT-3 paper uses 375M warmup tokens and 260B decay tokens.
## For MoE model, we found that setting the decay token to 300B helps.
WARMUP_TOKENS=375000000
# LR_DECAY_TOKENS=260000000000
LR_DECAY_TOKENS=300000000000
###############################################################################
### Parallelism configs
## Micro batch size per GPU
## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS
BATCH_SIZE=4

## Model parallelism, 1 is no MP
MP_SIZE=1

## Pipeline parallelism
## Currently we don't support PP for MoE. To disable PP, set PP_SIZE
## to 1 and use the "--no-pipeline-parallel" arg.
PP_SIZE=1
NUM_GPUS_PERNODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_GPUS=$(( ${NUM_NODE} * ${NUM_GPUS_PERNODE} ))

echo $NUM_GPUS GPUS
echo $NUM_NODE NODES


###############################################################################
### SYMI MoE configs ##########################################################
###############################################################################

## Baselines
#          | ADAPTIVE_MOE | BIND_OPTIMIZER | ZERO_STAGE
# ---------+--------------+----------------+------------
# SYMI     |     true     |      false     |     1
# FlexZeRO |     true     |      true      |     1
# FlexMoE  |     true     |      true      |     0
# DS-ZeRO  |     false    |      true      |     1
# DS-GPU   |     false    |      true      |     0

## Enable adaptive expert replication
# ADAPTIVE_MOE="false"
ADAPTIVE_MOE="true"

## Bind the optimizer placement with to the experts
BIND_OPTIMIZER="false"
# BIND_OPTIMIZER="true"

## ZeRO optimizer stage
ZERO_STAGE=1

## Construct an MoE layer every EXPERT_INTERVAL layers
EXPERT_INTERVAL=1

## EXPERTS is the number of expert instances (1 means dense model without MoE).
EXPERTS=4
if [[ $EXPERTS -lt $NUM_GPUS ]]; then
    echo "ERROR: EXPERTS should be larger than NUM_GPUS"
    exit
fi
## EXPERT_CLASSES is the number of expert classes that expert instances group into (for adaptive baselines).
EXPERT_CLASSES=3

## EP_PARALLEL_SIZE is the number of expert classes for the non-adaptive baselines.
## EXPERTS / EP_PARALLEL_SIZE is the number of expert slots per GPU for all baselines.
# EP_PARALLEL_SIZE=4
# if [[ $ADAPTIVE_MOE == "true" ]]; then
#     EP_PARALLEL_SIZE=$NUM_GPUS
# fi
### FIXME: why doesn't megastron/deepspeed support tuning EDP groups?
### This used to work. Megatron should have some assert somewhere
EP_PARALLEL_SIZE=$NUM_GPUS

## Coefficient for MoE loss (load balancing loss)
## Megatron: 0.01 works well for 1.3B MoE-128 model
MLC=0.01

## Capacity inputs have minor effect to adaptive baselines
## To completely disable capacity limit, set MOE_DROP_TOKEN to false.
## Larger capacity factor or disabling capacity limit could improve training
## convergence, but will also reduce training throughput.
MOE_TRAIN_CAP_FACTOR=1.0
MOE_MIN_CAP=4
MOE_DROP_TOKEN="true"
# MOE_DROP_TOKEN="false"

###############################################################################
###############################################################################


## Original GPT-3 model always set min LR at 10% of max LR. For MoE model, we
## found that lower LR and min LR (than the base dense model) helps.
## For 1.3B MoE-128 model we used LR=1.2e-4 and MIN_LR=1.0e-6.
## For 350M MoE-128 model we used LR=2.0e-4 and MIN_LR=2.0e-6, but they are not
## heavily tuned.
LR=4.5e-4
MIN_LR=4.5e-06

## Below configs adjust the MoE expert token capacity limit during eval
## eval. To completely disable capacity limit, set MOE_DROP_TOKEN to false.
## Larger capacity factor or disabling capacity limit could improve training
## convergence, but will also reduce training throughput.
MOE_EVAL_CAP_FACTOR=1.0

###############################################################################
### Curriculum learning (CL) configs
## Enable/disable CL
CL_ENABLED="false"
## Consult the tutorial https://www.deepspeed.ai/tutorials/curriculum-learning/
## for tuning the following configs
CL_START_SEQLEN=80
CL_AVG_SEQLEN=$(( (${CL_START_SEQLEN} + ${SEQ_LEN}) / 2 ))
CL_TOKENS=60
CL_TOKENS=$((${CL_TOKENS} * 1000000000))
CL_STEP=$(( ${CL_TOKENS} / (${GLOBAL_BATCH_SIZE} * ${CL_AVG_SEQLEN}) ))
###############################################################################
### Misc configs
LOG_INTERVAL=1
EVAL_ITERS=10
EVAL_INTERVAL=100
SAVE_INTERVAL=10000

## Standard deviation for weight initialization
## We used 0.014 for 350M/1.3B dense/MoE models, and used 0.01 for 6.7B
## dense model. Usually larger model needs lower std.
INIT_STD=0.014
# INIT_STD=0.01

## Activation checkpointing saves GPU memory, but reduces training speed
# ACTIVATION_CHECKPOINT="true"
ACTIVATION_CHECKPOINT="false"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
NAME="gpt-${MODEL_SIZE}B-lr-${LR}-minlr-${MIN_LR}-bs-${GLOBAL_BATCH_SIZE}-gpus-${NUM_GPUS}-mp-${MP_SIZE}-pp-${PP_SIZE}-zero-${ZERO_STAGE}"
if [[ $EXPERTS -gt 1 ]]; then
    NAME="${NAME}-expi-${EXPERTS}-expc-${EXPERT_CLASSES}-ada-${ADAPTIVE_MOE}-bindopt-${BIND_OPTIMIZER}-mlc-${MLC}-cap-${MOE_TRAIN_CAP_FACTOR}-drop-${MOE_DROP_TOKEN}"
fi
if [ "${CL_ENABLED}" = "true" ]; then
    NAME="${NAME}-cl-${CL_START_SEQLEN}-${CL_STEP}"
fi
if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
    NAME="${NAME}-activationcheckpointing"
fi

OUTPUT_BASEPATH=$DIR/output
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
mkdir -p "${OUTPUT_BASEPATH}/configs/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${host}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
## Note that for MoE model with billion-scale base model, the checkpoint can be
## as large as TB-scale which normal NFS cannot handle efficiently.
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

VOCAB_PATH=../dataset/gpt2-vocab.json
MERGE_PATH=../dataset/gpt2-merges.txt
DATA_PATH=../dataset/wikitext103_text_document

###############################################################################
data_options=" \
         --vocab-file ${VOCAB_PATH} \
         --merge-file ${MERGE_PATH} \
         --data-path ${DATA_PATH} \
         --data-impl mmap"

megatron_options=" \
        --override-opt_param-scheduler \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --expert-interval ${EXPERT_INTERVAL} \
        --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
        --num-experts ${EXPERTS} \
        --moe-loss-coeff ${MLC} \
        --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
        --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
        --moe-min-capacity ${MOE_MIN_CAP} \
        --init-method-std ${INIT_STD} \
        --lr-decay-tokens ${LR_DECAY_TOKENS} \
        --lr-warmup-tokens ${WARMUP_TOKENS} \
        --micro-batch-size ${BATCH_SIZE} \
        --exit-duration-in-mins ${EXIT_DURATION} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-tokens ${TRAIN_TOKENS} \
        --train-iters ${TRAIN_ITERS} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 98,2,0 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 0 \
        --fp16 \
        --save ${CHECKPOINT_PATH} \
        --tensorboard-queue-size 1 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensorboard-dir ${TENSORBOARD_DIR}"

if [ "${ZERO_STAGE}" -gt "0" ]; then
megatron_options="${megatron_options} \
        --cpu-optimizer"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
megatron_options="${megatron_options} \
        --checkpoint-activations"
fi

if [[ $EXPERTS -gt 1 ]]; then
megatron_options="${megatron_options} \
        --create-moe-param-group"
fi

if [ "${MOE_DROP_TOKEN}" = "false" ]; then
megatron_options="${megatron_options} \
        --disable-moe-token-dropping"
fi

if [ "${ADAPTIVE_MOE}" = "true" ]; then
megatron_options="${megatron_options} \
        --adaptive-expert-replication \
        --num-expert-classes ${EXPERT_CLASSES}"
fi

if [ "${BIND_OPTIMIZER}" = "true" ]; then
megatron_options="${megatron_options} \
        --bind-optimizer"
fi

if [ $log_expert_selection -ne 0 ]; then
megatron_options="${megatron_options} \
        --log-moe-expert-selection \
        --log-moe-expert-selection-dir ${TENSORBOARD_DIR}"
fi

template_json="configs/ds_config_gpt_TEMPLATE.json"
config_json="output/configs/ds_config_gpt_${NAME}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/${ZERO_STAGE}/" \
    | sed "s/ZERO_ALLGATHER_PARTITIONS/true/" \
    | sed "s/ZERO_REDUCESCATTER/true/" \
    | sed "s/ZERO_ALLGATHER_BUCKET_SIZE/50000000/" \
    | sed "s/ZERO_REDUCE_BUCKET_SIZE/50000000/" \
    | sed "s/ZERO_OVERLAP_COMM/true/" \
    | sed "s/ZERO_CONTIGUOUS_GRADIENTS/true/" \
    | sed "s/ZERO_CPU_OFFLOAD/true/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
    | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
    | sed "s|TENSORBOARD_OUTPUT|\"${TENSORBOARD_DIR}\"|" \
    | sed "s|CSV_OUTPUT|\"${TENSORBOARD_DIR}\"|" \
    | sed "s|TENSORBOARD_JOB_NAME|\"${NAME}\"|" \
    | sed "s|CSV_JOB_NAME|\"${NAME}\"|" \
        > ${config_json}

deepspeed_options=" \
		    --deepspeed \
		    --deepspeed_config ${config_json} \
		    --pipeline-model-parallel-size ${PP_SIZE}"

# Currently MoE is not compatible with pipeline parallel
if [[ $EXPERTS -gt 1 ]]; then
deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing"
fi

# Copy the config file to all remote notes
if [[ -n $HOST_FILE ]]; then
    while IFS= read -r line; do
        SERVER_IP=$(echo $line | awk '{print $1}')
        ssh -p 2222 deepspeed@${SERVER_IP} "mkdir -p ${OUTPUT_BASEPATH}/configs/" < /dev/null
        if [ $? -ne 0 ]; then
            echo "Failed to create directory on $SERVER_IP."
            exit
        fi
        scp -P 2222 $config_json deepspeed@${SERVER_IP}:${OUTPUT_BASEPATH}/configs/ < /dev/null
        if [ $? -ne 0 ]; then
            echo "Failed to copy config to $SERVER_IP."
            exit
        fi
    done < ${HOST_FILE}
fi

run_cmd="deepspeed ${HOST_ARGS} ${DIR}/../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} 2>&1 | tee ${OUTPUT_BASEPATH}/log/${NAME}_${host}_${current_time}.log"
echo ${run_cmd}
eval ${run_cmd}
set +x