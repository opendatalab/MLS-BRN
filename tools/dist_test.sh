#!/usr/bin/env bash

CONFIG=$1
TIME=$2

GPUS=1

JOB_DIR="work_dirs/${CONFIG}[${TIME}]"
CONFIG_FILE="${JOB_DIR}/${CONFIG}.py" 
CHECKPOINT="${JOB_DIR}/latest.pth"
PKL_FILE="${JOB_DIR}/result.pkl" 
CITY="bonai"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29705}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/bonai/bonai_test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --out ${PKL_FILE} \
    --city ${CITY} \
    --launcher pytorch \
    ${@:3}

python $(dirname "$0")/bonai/bonai_evaluation.py \
    ${PKL_FILE} \
    ${JOB_DIR} \
    --city ${CITY}

# ./tools/dist_test.sh loft_foahfm_r50_fpn_2x_bonai_ssl <timestamp>