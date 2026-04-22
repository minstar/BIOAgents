#!/bin/bash
# ============================================================
# v16 Step 60 — Multi-Turn Benchmark Evaluation
# Uses AgentRunner with full tool-use loop matching training format
# 70% MedQA accuracy validated on 10-sample pilot (vs 0% single-turn)
# ~93s/sample avg, 7.7 turns avg
# ============================================================

MODEL_PATH="/data/project/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_self_distill_v16_ema_cosine/global_step_60/actor/merged_hf"
OUTPUT_DIR="results/benchmarks_multiturn/v16_step60"
PYTHON="/data/project/private/minstar/workspace/BIOAgents/.venv/bin/python"
PROJECT_ROOT="/data/project/private/minstar/workspace/BIOAgents"
LOG_DIR="${PROJECT_ROOT}/logs/v16_step60_multiturn"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

echo "========================================================"
echo "  v16 Step 60 — Multi-Turn Benchmark Evaluation"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  $(date)"
echo "========================================================"

# ── 1. MedQA (1273 samples) — GPU 0 ──
# Est: 1273 × 93s = ~33h
echo "[1/6] MedQA (1273 samples) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/eval_benchmark_multiturn.py \
    --model_path "$MODEL_PATH" \
    --benchmarks medqa \
    --max-turns 10 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/medqa.log" 2>&1 &
PID1=$!

# ── 2. MedMCQA shard 1 (first 2000) — GPU 1 ──
echo "[2/6] MedMCQA shard1 (0-2000) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/eval_benchmark_multiturn.py \
    --model_path "$MODEL_PATH" \
    --benchmarks medmcqa \
    --max-samples 2000 \
    --max-turns 10 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/medmcqa_s1.log" 2>&1 &
PID2=$!

# ── 3. MedMCQA shard 2 (2000-4183) — GPU 2 ──
echo "[3/6] MedMCQA shard2 (2000+) on GPU 2..."
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/eval_benchmark_multiturn.py \
    --model_path "$MODEL_PATH" \
    --benchmarks medmcqa \
    --resume-from 2000 \
    --max-turns 10 \
    --output-dir "${OUTPUT_DIR}" \
    > "$LOG_DIR/medmcqa_s2.log" 2>&1 &
PID3=$!

# ── 4. MMLU (1089 samples) — GPU 3 ──
echo "[4/6] MMLU (1089 samples) on GPU 3..."
CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/eval_benchmark_multiturn.py \
    --model_path "$MODEL_PATH" \
    --benchmarks mmlu \
    --max-turns 10 \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/mmlu.log" 2>&1 &
PID4=$!

# GPUs 4-7 are free for other work (VQA, v16 training, etc.)

echo ""
echo "All 4 TextQA jobs launched:"
echo "  MedQA       PID=$PID1   (GPU 0, 1273 samples, ~33h)"
echo "  MedMCQA-s1  PID=$PID2   (GPU 1, 2000 samples, ~52h)"
echo "  MedMCQA-s2  PID=$PID3   (GPU 2, 2183 samples, ~56h)"
echo "  MMLU        PID=$PID4   (GPU 3, 1089 samples, ~28h)"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/medqa.log"
echo "  tail -f $LOG_DIR/medmcqa_s1.log"
echo "  tail -f $LOG_DIR/medmcqa_s2.log"
echo "  tail -f $LOG_DIR/mmlu.log"
echo ""

wait
echo ""
echo "========================================================"
echo "  All benchmarks completed! $(date)"
echo "========================================================"
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null
