#!/bin/bash
# Merge LoRA checkpoint and run TextQA evaluation for algorithm comparison
# Usage: ./scripts/merge_and_eval_ablation.sh <checkpoint_dir> <run_name> <gpu_id>

set -e
cd /data/project/private/minstar/workspace/BIOAgents

CKPT_DIR="$1"      # e.g., checkpoints/grpo_baseline_lingshu7b/checkpoint-50
RUN_NAME="$2"      # e.g., grpo_baseline
GPU_ID="$3"        # e.g., 2

if [ -z "$CKPT_DIR" ] || [ -z "$RUN_NAME" ] || [ -z "$GPU_ID" ]; then
    echo "Usage: $0 <checkpoint_dir> <run_name> <gpu_id>"
    exit 1
fi

MERGED_DIR="${CKPT_DIR}-merged"
RESULTS_DIR="results/algorithm_comparison/${RUN_NAME}"

echo "[$(date)] Starting merge and eval for $RUN_NAME (GPU $GPU_ID)"

# Step 1: Merge LoRA
if [ ! -d "$MERGED_DIR" ]; then
    echo "[$(date)] Merging LoRA weights..."
    CUDA_VISIBLE_DEVICES=$GPU_ID .venv/bin/python scripts/merge_lora.py \
        --base-model checkpoints/sft_warmup_lingshu7b_v2_merged/merged \
        --lora-path "$CKPT_DIR" \
        --output-dir "$MERGED_DIR"
    
    # Copy preprocessor config for VL model
    cp checkpoints/sft_warmup_lingshu7b_v2_merged/merged/preprocessor_config.json "$MERGED_DIR/" 2>/dev/null || true
    echo "[$(date)] Merge complete: $MERGED_DIR"
else
    echo "[$(date)] Already merged: $MERGED_DIR"
fi

# Step 2: Run TextQA evaluation
echo "[$(date)] Running TextQA evaluation..."
mkdir -p "$RESULTS_DIR"
CUDA_VISIBLE_DEVICES=$GPU_ID .venv/bin/python scripts/run_full_benchmark_suite.py \
    --category textqa \
    --model_path "$MERGED_DIR" \
    --gpus 0 \
    --output-dir "$RESULTS_DIR" \
    > "logs/${RUN_NAME}_eval_$(date +%Y%m%d_%H%M%S).log" 2>&1

echo "[$(date)] TextQA evaluation complete for $RUN_NAME"
echo "[$(date)] Results in $RESULTS_DIR"
