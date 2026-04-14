#!/bin/bash
# Run all benchmarks (TextQA + VQA + MedLFQA) via SGLang server.
#
# Usage:
#   ./scripts/eval_all_sglang.sh \
#       --model /path/to/merged_hf \
#       --name v6_step80 \
#       --gpu 6 \
#       --output-dir results/benchmarks_tooluse/v6_step80
#
# This script:
#   1. Launches SGLang server on the specified GPU
#   2. Waits for it to be healthy
#   3. Runs VQA eval (vqa_rad, slake, pathvqa)
#   4. Runs MedLFQA eval (5 subtasks)
#   5. Kills the server

set -e

PYTHON=/data/project/private/minstar/miniconda3/envs/verl/bin/python
PORT=30080
CONCURRENCY=16

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$NAME" ] || [ -z "$GPU" ]; then
    echo "Usage: $0 --model PATH --name NAME --gpu GPU_ID [--output-dir DIR] [--port PORT]"
    exit 1
fi

OUTPUT_DIR=${OUTPUT_DIR:-"results/benchmarks_tooluse/${NAME}"}
BIOAGENTS=$(cd "$(dirname "$0")/.." && pwd)

echo "=============================================="
echo "  Full Eval via SGLang"
echo "  Model: $NAME"
echo "  GPU: $GPU"
echo "  Port: $PORT"
echo "  Output: $OUTPUT_DIR"
echo "=============================================="

# 1. Launch SGLang server
echo "[1/4] Launching SGLang server on GPU $GPU..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m sglang.launch_server \
    --model-path "$MODEL" \
    --port $PORT \
    --attention-backend triton \
    --tp 1 \
    --trust-remote-code \
    --dtype bfloat16 \
    --mem-fraction-static 0.85 \
    > "$BIOAGENTS/logs/sglang_${NAME}.log" 2>&1 &
SGLANG_PID=$!
echo "  SGLang PID: $SGLANG_PID"

# Wait for server to be healthy
echo "  Waiting for server to be ready..."
for i in $(seq 1 120); do
    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "  ERROR: SGLang process died. Check logs/sglang_${NAME}.log"
        exit 1
    fi
    if curl -s http://localhost:$PORT/health 2>/dev/null | grep -qi "ok\|true\|healthy"; then
        echo "  Server ready! (${i}*5s)"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "  ERROR: Server did not start within 10 minutes"
        kill $SGLANG_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done

SERVER_URL="http://localhost:$PORT"

# 2. Run VQA eval
echo ""
echo "[2/4] Running VQA eval..."
cd "$BIOAGENTS"
$PYTHON scripts/eval_vqa_sglang.py \
    --server-url $SERVER_URL \
    --model-name "$NAME" \
    --output-dir "$OUTPUT_DIR" \
    --concurrency $CONCURRENCY \
    2>&1 | tee "logs/eval_vqa_sglang_${NAME}.log"

# 3. Run MedLFQA eval
echo ""
echo "[3/4] Running MedLFQA eval..."
$PYTHON scripts/eval_medlfqa_sglang.py \
    --server-url $SERVER_URL \
    --model-name "$NAME" \
    --output-dir "$OUTPUT_DIR" \
    --concurrency 8 \
    2>&1 | tee "logs/eval_medlfqa_sglang_${NAME}.log"

# 4. Kill server
echo ""
echo "[4/4] Stopping SGLang server..."
kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null || true
echo "  Done!"

echo ""
echo "=============================================="
echo "  All evals complete for $NAME"
echo "  Results: $OUTPUT_DIR"
echo "=============================================="
