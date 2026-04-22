#!/bin/bash
# Monitor multi-turn benchmark evaluation every 15 minutes
# Reports progress, restarts dead processes, and launches new benchmarks on free GPUs

PROJECT_ROOT="/data/project/private/minstar/workspace/BIOAgents"
LOG_DIR="${PROJECT_ROOT}/logs/v16_step60_multiturn"
MODEL_PATH="/data/project/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_self_distill_v16_ema_cosine/global_step_60/actor/merged_hf"
OUTPUT_DIR="${PROJECT_ROOT}/results/benchmarks_multiturn/v16_step60"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
MONITOR_LOG="${LOG_DIR}/monitor.log"

cd "$PROJECT_ROOT"

echo "$(date) === Monitor Check ===" >> "$MONITOR_LOG"

# Check which GPUs are in use
ACTIVE_GPUS=""
for gpu in 0 1 2 3 4 5 6 7; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null)
    if [ "$mem" -gt 1000 ] 2>/dev/null; then
        ACTIVE_GPUS="$ACTIVE_GPUS $gpu"
    fi
done

echo "  Active GPUs:$ACTIVE_GPUS" >> "$MONITOR_LOG"

# Check running eval processes
RUNNING=$(ps aux | grep eval_benchmark_multiturn | grep python | grep -v grep | wc -l)
echo "  Running processes: $RUNNING" >> "$MONITOR_LOG"

# Extract latest progress from each log
for logfile in "$LOG_DIR"/*.log; do
    [ "$logfile" = "$MONITOR_LOG" ] && continue
    basename=$(basename "$logfile" .log)
    latest=$(grep "acc=" "$logfile" 2>/dev/null | tail -1)
    if [ -n "$latest" ]; then
        echo "  $basename: $latest" >> "$MONITOR_LOG"
    fi
done

# Check for completed benchmarks
for logfile in "$LOG_DIR"/*.log; do
    [ "$logfile" = "$MONITOR_LOG" ] && continue
    if grep -q "MULTI-TURN BENCHMARK RESULTS" "$logfile" 2>/dev/null; then
        basename=$(basename "$logfile" .log)
        result=$(grep -A5 "MULTI-TURN BENCHMARK RESULTS" "$logfile" | tail -3)
        echo "  COMPLETED: $basename - $result" >> "$MONITOR_LOG"
    fi
done

# Print summary to stdout
echo "$(date) - $RUNNING processes running on GPUs:$ACTIVE_GPUS"
grep "acc=" "$LOG_DIR"/*.log 2>/dev/null | grep -v monitor | tail -6
