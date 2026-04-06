#!/bin/bash
# Auto-merge and evaluate when checkpoint appears
# Usage: ./scripts/auto_merge_eval.sh <checkpoint_dir> <base_model> <gpu_id> <eval_category>

CKPT_DIR=$1
BASE_MODEL=$2
GPU_ID=$3
EVAL_CAT=$4
MERGED_DIR="${CKPT_DIR}-merged"

cd /data/project/private/minstar/workspace/BIOAgents

# Wait for checkpoint to appear
echo "Waiting for ${CKPT_DIR}..."
while [ ! -f "${CKPT_DIR}/adapter_model.safetensors" ]; do
    sleep 30
done
echo "Checkpoint found at $(date)!"

# Merge
echo "Merging..."
.venv/bin/python scripts/merge_lora.py \
    --base-model "${BASE_MODEL}" \
    --adapter "${CKPT_DIR}" \
    --output "${MERGED_DIR}"

# Evaluate
echo "Evaluating ${EVAL_CAT}..."
CUDA_VISIBLE_DEVICES=${GPU_ID} .venv/bin/python scripts/run_full_benchmark_suite.py \
    --model_path "${MERGED_DIR}" \
    --category "${EVAL_CAT}" \
    --gpus "${GPU_ID}" \
    --output-dir "results/algorithm_comparison/$(basename $(dirname ${CKPT_DIR}))_$(basename ${CKPT_DIR})_${EVAL_CAT}"

echo "Done at $(date)!"
