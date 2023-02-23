#!/usr/bin/env bash

MODEL=$1
CONFIG=$2

GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29502}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

TIME=$(date "+%Y%m%d-%H%M%S")

JOB_NAME=${CONFIG}[${TIME}]
CONFIG_FILE="configs/${MODEL}/${CONFIG}.py" 
WORK_DIR="work_dirs/${JOB_NAME}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --config ${CONFIG_FILE} \
    --work-dir=${WORK_DIR} \
    --seed 0 \
    --launcher pytorch ${@:3}\
    --no-validate

# bash ./tools/dist_train.sh loft_foahfm_ssl loft_foahfm_r50_fpn_2x_bonai_ssl