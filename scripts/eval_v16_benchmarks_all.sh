#!/bin/bash
# ============================================================
# v16 Step 60 — Full Benchmark Evaluation (Training-Aligned Format)
# Uses the same tool-use prompt format as training across all benchmarks
# ============================================================

MODEL_PATH="/data/project/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_self_distill_v16_ema_cosine/global_step_60/actor/merged_hf"
OUTPUT_DIR="results/benchmarks_tooluse/v16_step60"
PYTHON="/data/project/private/minstar/workspace/BIOAgents/.venv/bin/python"
PROJECT_ROOT="/data/project/private/minstar/workspace/BIOAgents"
LOG_DIR="${PROJECT_ROOT}/logs/v16_step60_benchmarks"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

echo "========================================================"
echo "  v16 Step 60 — Full Benchmark Evaluation"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  $(date)"
echo "========================================================"

# ── 1. TextQA on GPUs 0-3 (all 3 benchmarks at once) ──
echo "[1/3] TextQA (MedQA + MedMCQA + MMLU) on GPUs 0-3..."
CUDA_VISIBLE_DEVICES=0,1,2,3 $PYTHON scripts/eval_textqa_tooluse_format.py \
    --model_path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --no-think \
    > "$LOG_DIR/textqa.log" 2>&1 &
PID_TEXTQA=$!

# ── 2. VQA on GPUs 4-5 (VQA-RAD, SLAKE, PathVQA — sequential, batch=1 due to Qwen3.5 issue) ──
echo "[2/3] VQA (VQA-RAD + SLAKE + PathVQA) on GPUs 4,5..."
CUDA_VISIBLE_DEVICES=4,5 $PYTHON scripts/eval_vqa_tooluse_format.py \
    --model_path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --gpus "0,1" \
    --batch-size 1 \
    --no-think \
    > "$LOG_DIR/vqa.log" 2>&1 &
PID_VQA=$!

# ── 3. MedLFQA on GPUs 6-7 ──
echo "[3/3] MedLFQA (5 datasets) on GPUs 6,7..."
CUDA_VISIBLE_DEVICES=6,7 $PYTHON scripts/eval_medlfqa_tooluse_format.py \
    --model_path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --gpus "0,1" \
    > "$LOG_DIR/medlfqa.log" 2>&1 &
PID_MEDLFQA=$!

echo ""
echo "All 3 benchmark groups launched:"
echo "  TextQA  PID=$PID_TEXTQA  (GPUs 0-3)"
echo "  VQA     PID=$PID_VQA     (GPUs 4-5)"
echo "  MedLFQA PID=$PID_MEDLFQA (GPUs 6-7)"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/textqa.log"
echo "  tail -f $LOG_DIR/vqa.log"
echo "  tail -f $LOG_DIR/medlfqa.log"
echo ""

wait
echo ""
echo "========================================================"
echo "  All benchmarks completed! $(date)"
echo "========================================================"
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null
