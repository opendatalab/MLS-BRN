#!/usr/bin/env bash

set -x

# GPUS=1
GPUS=2
# GPUS=4
CPUS_PER_TASK=5
GPUS_PER_NODE=$GPUS
SRUN_ARGS=""
# SRUN_ARGS="--debug"
PARTITION="PARTITION"


MODEL=$1
CONFIG=$2
PY_ARGS=${@:3:$#-3}

TIME=$(date "+%Y%m%d-%H%M%S")

JOB_NAME=${CONFIG}[${TIME}]
JOB_DIR="work_dirs/${JOB_NAME}"
CONFIG_FILE="configs/${MODEL}/${CONFIG}.py"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py --config=${CONFIG_FILE} --work-dir=${JOB_DIR} --launcher="slurm" ${PY_ARGS} --no-validate

# ==================== The command to call this shell script ====================
# ./tools/slurm_train.sh loft_foahfm_ssl loft_foahfm_r50_fpn_2x_bonai_ssl