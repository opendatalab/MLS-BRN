#!/usr/bin/env bash

set -x

GPUS=1
CPUS_PER_TASK=5
GPUS_PER_NODE=$GPUS
SRUN_ARGS=""
# SRUN_ARGS="--debug"
PARTITION="PARTITION"


CONFIG=$1
TIME=$2
PY_ARGS=${@:3:$#-3}

EVAL_TYPE="segm"
EPOCH_NAME="latest"

JOB_NAME="${CONFIG}[${TIME}]"
TEST_JOB_NAME="${JOB_NAME}_test"
EVALUATE_JOB_NAME="${JOB_NAME}_evaluate"
JOB_DIR="work_dirs/${JOB_NAME}"
CONFIG_FILE="${JOB_DIR}/${CONFIG}.py"
CHECKPOINT="${JOB_DIR}/${EPOCH_NAME}.pth"
PKL_FILE="${JOB_DIR}/result.pkl"
CITY="bonai"
TEST_PY_ARGS="$PY_ARGS --eval $EVAL_TYPE --city $CITY --out $PKL_FILE"
EVAL_PY_ARGS="${PKL_FILE} ${JOB_DIR} --city $CITY"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${TEST_JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/bonai/bonai_test.py ${CONFIG_FILE} ${CHECKPOINT} --launcher="slurm" ${TEST_PY_ARGS}

srun -p ${PARTITION} \
    --job-name=${EVALUATE_JOB_NAME} \
    --ntasks=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/bonai/bonai_evaluation.py ${EVAL_PY_ARGS}
# ==================== The command to call this shell script ====================
# ./tools/slurm_test.sh loft_foahfm_r50_fpn_2x_bonai_ssl <timestamp>
