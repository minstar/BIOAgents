#!/bin/bash
# Auto-queue monitor: checks every 15 min, launches new benchmarks on free GPUs
# Usage: nohup bash scripts/eval_queue_monitor.sh > logs/eval_queue_monitor.log 2>&1 &

PROJECT_ROOT="/data/project/private/minstar/workspace/BIOAgents"
MODEL_PATH="/data/project/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_self_distill_v16_ema_cosine/global_step_60/actor/merged_hf"
OUTPUT_DIR="${PROJECT_ROOT}/results/benchmarks_multiturn/v16_step60"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
LOG_DIR="${PROJECT_ROOT}/logs/v16_step60_multiturn"
QUEUE_FILE="${PROJECT_ROOT}/logs/eval_queue.txt"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ── Benchmark queue (ordered by priority / size) ──
# Format: benchmark_name:type:max_turns:max_samples
# type: text (medical_qa domain) or vqa (visual_diagnosis domain)
# Already running: medqa, mmlu, medmcqa(x4), vqa_rad, pathvqa
# Queue for new benchmarks:
QUEUE=(
    # MMLU subtypes (small, fast)
    "mmlu_anatomy:text:10:0"
    "mmlu_clinical:text:10:0"
    "mmlu_professional:text:10:0"
    "mmlu_genetics:text:10:0"
    "mmlu_biology:text:10:0"
    "mmlu_college_med:text:10:0"
    # MedLFQA (medium)
    "live_qa:text:10:0"
    "kqa_golden:text:10:0"
    "medication_qa:text:10:0"
    "kqa_silver:text:10:0"
    "healthsearch_qa:text:10:1000"
    # VQA (need VLM)
    "slake:vqa:5:0"
    "pmc_vqa:vqa:5:1000"
    "vqa_med_2021:vqa:5:0"
    "quilt_vqa:vqa:5:0"
)

# Initialize queue file if not exists
if [ ! -f "$QUEUE_FILE" ]; then
    for item in "${QUEUE[@]}"; do
        echo "$item" >> "$QUEUE_FILE"
    done
    echo "$(date) Initialized queue with ${#QUEUE[@]} benchmarks"
fi

get_free_gpus() {
    free_gpus=""
    for gpu in 0 1 2 3 4 5 6 7; do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu 2>/dev/null)
        if [ "$mem" -lt 1000 ] 2>/dev/null; then
            free_gpus="$free_gpus $gpu"
        fi
    done
    echo "$free_gpus"
}

launch_benchmark() {
    local gpu=$1
    local bench=$2
    local btype=$3
    local max_turns=$4
    local max_samples=$5

    local domain="medical_qa"
    [ "$btype" = "vqa" ] && domain="visual_diagnosis"
    [ "$btype" = "ehr" ] && domain="ehr_management"

    local sample_arg=""
    [ "$max_samples" -gt 0 ] 2>/dev/null && sample_arg="--max-samples $max_samples"

    echo "$(date) Launching $bench on GPU $gpu (domain=$domain, turns=$max_turns, samples=$max_samples)"

    CUDA_VISIBLE_DEVICES=$gpu nohup $PYTHON scripts/eval_benchmark_multiturn.py \
        --model_path "$MODEL_PATH" \
        --benchmarks $bench \
        --domain $domain \
        --max-turns $max_turns \
        $sample_arg \
        --output-dir "$OUTPUT_DIR" \
        > "${LOG_DIR}/${bench}.log" 2>&1 &

    echo "  PID: $!"
}

# ── Main loop ──
while true; do
    echo ""
    echo "$(date) === Queue Monitor Check ==="

    # Get free GPUs
    FREE=$(get_free_gpus)
    echo "  Free GPUs:$FREE"

    # Count remaining queue items
    remaining=$(wc -l < "$QUEUE_FILE" 2>/dev/null || echo 0)
    echo "  Queue remaining: $remaining"

    if [ "$remaining" -eq 0 ]; then
        echo "  Queue empty! All benchmarks launched."
        # Check if any processes still running
        running=$(ps aux | grep eval_benchmark_multiturn | grep python | grep -v grep | wc -l)
        if [ "$running" -eq 0 ]; then
            echo "  All processes done. Exiting monitor."
            break
        fi
        echo "  $running processes still running. Waiting..."
    else
        # Launch benchmarks on free GPUs
        for gpu in $FREE; do
            if [ "$remaining" -le 0 ]; then
                break
            fi

            # Pop first item from queue
            item=$(head -1 "$QUEUE_FILE")
            if [ -z "$item" ]; then
                break
            fi

            # Parse item
            IFS=':' read -r bench btype max_turns max_samples <<< "$item"

            # Check if already running or completed
            if ps aux | grep "benchmarks $bench" | grep -v grep > /dev/null 2>&1; then
                echo "  $bench already running, skip"
                sed -i '1d' "$QUEUE_FILE"
                remaining=$((remaining - 1))
                continue
            fi

            if ls "${OUTPUT_DIR}/${bench}_multiturn_"*.json > /dev/null 2>&1; then
                echo "  $bench already completed, skip"
                sed -i '1d' "$QUEUE_FILE"
                remaining=$((remaining - 1))
                continue
            fi

            # Launch
            launch_benchmark "$gpu" "$bench" "$btype" "$max_turns" "$max_samples"
            sed -i '1d' "$QUEUE_FILE"
            remaining=$((remaining - 1))

            # Wait a bit for GPU memory to be allocated
            sleep 30
        done
    fi

    # Print current status
    echo "  --- Running processes ---"
    ps aux | grep eval_benchmark_multiturn | grep python | grep -v grep | awk '{print "  PID=" $2 " CMD=" $NF}'

    # Print latest progress
    echo "  --- Latest progress ---"
    for logfile in "$LOG_DIR"/*.log; do
        bn=$(basename "$logfile" .log)
        [ "$bn" = "monitor" ] || [ "$bn" = "monitor_loop" ] || [ "$bn" = "eval_queue_monitor" ] && continue
        latest=$(grep "acc=" "$logfile" 2>/dev/null | tail -1)
        if [ -n "$latest" ]; then
            echo "  $bn: $(echo $latest | grep -oP '\[.*')"
        fi
    done

    # Check for completed benchmarks
    echo "  --- Completed ---"
    for logfile in "$LOG_DIR"/*.log; do
        bn=$(basename "$logfile" .log)
        if grep -q "MULTI-TURN BENCHMARK RESULTS\|accuracy=" "$logfile" 2>/dev/null; then
            result=$(grep "accuracy=" "$logfile" 2>/dev/null | tail -1)
            [ -n "$result" ] && echo "  DONE: $bn - $result"
        fi
    done

    echo "  Sleeping 15 minutes..."
    sleep 900
done

echo "$(date) === Queue Monitor Finished ==="
