#!/bin/bash
# Full evaluation pipeline with tool-use format alignment
# TextQA: tool-use format (submit_answer via apply_chat_template)
# VQA: existing evaluator (open-ended, format less critical)
# MedLFQA: existing evaluator (long-form, format less critical)
# EHR: existing evaluator (already uses AgentRunner with tools)

set -e
cd /data/project/private/minstar/workspace/BIOAgents

CKPT_BASE="/mnt/aiplatform/csi-volumes/pvc-e668fe31-e015-4e4e-a3f4-35f18e2ad53f-bd5321b06ddb2b68ae682cc934af2027aeea25db/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_vlm_grpo_multiturn_v6"
PYTHON="/data/project/private/minstar/miniconda3/envs/verl/bin/python"
GPUS="0"

STEPS=(80 90 100)

echo "$(date): Starting full tool-use format evaluation pipeline"
echo "============================================================"

# ── Phase 1: TextQA with tool-use format (highest priority) ──
for step in "${STEPS[@]}"; do
    MODEL_PATH="$CKPT_BASE/global_step_${step}/merged_hf"
    OUT_DIR="results/benchmarks_tooluse/v6_step${step}"
    echo ""
    echo "============================================================"
    echo "$(date): TextQA (tool-use) — step ${step}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPUS $PYTHON scripts/eval_textqa_tooluse_format.py \
        --model_path "$MODEL_PATH" \
        --batch-size 1 --gpus "$GPUS" \
        --output-dir "$OUT_DIR" 2>&1
    echo "$(date): Done TextQA (tool-use) — step ${step}"
done

# ── Phase 2: VQA (vision, uses existing evaluator) ──
for step in "${STEPS[@]}"; do
    MODEL_PATH="$CKPT_BASE/global_step_${step}/merged_hf"
    OUT_DIR="results/benchmarks_tooluse/v6_step${step}"
    echo ""
    echo "============================================================"
    echo "$(date): VQA — step ${step}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPUS $PYTHON scripts/run_full_benchmark_suite.py \
        --model_path "$MODEL_PATH" \
        --category vqa \
        --gpus "$GPUS" \
        --output-dir "$OUT_DIR" 2>&1
    echo "$(date): Done VQA — step ${step}"
done

# ── Phase 3: MedLFQA (long-form, uses existing evaluator) ──
for step in "${STEPS[@]}"; do
    MODEL_PATH="$CKPT_BASE/global_step_${step}/merged_hf"
    OUT_DIR="results/benchmarks_tooluse/v6_step${step}"
    echo ""
    echo "============================================================"
    echo "$(date): MedLFQA — step ${step}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPUS $PYTHON scripts/run_full_benchmark_suite.py \
        --model_path "$MODEL_PATH" \
        --category medlfqa \
        --gpus "$GPUS" \
        --output-dir "$OUT_DIR" 2>&1
    echo "$(date): Done MedLFQA — step ${step}"
done

# ── Phase 4: EHR (already tool-use via AgentRunner) ──
for step in "${STEPS[@]}"; do
    MODEL_PATH="$CKPT_BASE/global_step_${step}/merged_hf"
    OUT_DIR="results/benchmarks_tooluse/v6_step${step}"
    echo ""
    echo "============================================================"
    echo "$(date): EHR — step ${step}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPUS $PYTHON scripts/run_full_benchmark_suite.py \
        --model_path "$MODEL_PATH" \
        --category ehr \
        --gpus "$GPUS" \
        --output-dir "$OUT_DIR" 2>&1
    echo "$(date): Done EHR — step ${step}"
done

echo ""
echo "============================================================"
echo "$(date): ALL EVALUATIONS COMPLETE for v6 step 80/90/100"
echo "============================================================"
