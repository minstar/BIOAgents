#!/bin/bash
# Evaluate 4-modality checkpoints across all benchmark categories
# Usage: CUDA_VISIBLE_DEVICES=6 bash scripts/eval_4modality_checkpoints.sh

set -e
cd /data/project/private/minstar/workspace/BIOAgents

CKPTS=("checkpoint-50-merged" "checkpoint-100-merged")
CATEGORIES=("textqa" "vqa" "medlfqa" "ehr")

for ckpt in "${CKPTS[@]}"; do
    ckpt_path="checkpoints/full_4modality_lingshu7b/${ckpt}"
    for cat in "${CATEGORIES[@]}"; do
        echo "============================================"
        echo "  Evaluating ${ckpt} — ${cat}"
        echo "  $(date)"
        echo "============================================"
        .venv/bin/python scripts/run_full_benchmark_suite.py \
            --model_path "${ckpt_path}" \
            --category "${cat}" \
            --output-dir "results/4modality_${ckpt}"
        echo "  DONE: ${ckpt} — ${cat} at $(date)"
        echo ""
    done
done

echo "ALL EVALUATIONS COMPLETE at $(date)"
