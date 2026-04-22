#!/bin/bash
# Evaluate Qwen3.5-9B base model on all benchmarks (same multi-turn protocol as v16)
# Usage: nohup bash scripts/eval_base_model_queue.sh > logs/eval_base_queue_monitor.log 2>&1 &

PROJECT_ROOT="/data/project/private/minstar/workspace/BIOAgents"
MODEL_PATH="${PROJECT_ROOT}/checkpoints/models/Qwen3.5-9B"
OUTPUT_DIR="${PROJECT_ROOT}/results/benchmarks_multiturn/base_qwen35_9b"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
LOG_DIR="${PROJECT_ROOT}/logs/base_qwen35_9b_multiturn"
QUEUE_FILE="${PROJECT_ROOT}/logs/eval_base_queue.txt"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Initialize queue file if not exists
if [ ! -f "$QUEUE_FILE" ]; then
    cat > "$QUEUE_FILE" << 'EOF'
medqa:text:10:0
mmlu:text:10:0
medmcqa:text:10:1000
mmlu_anatomy:text:10:0
mmlu_clinical:text:10:0
mmlu_professional:text:10:0
mmlu_genetics:text:10:0
mmlu_biology:text:10:0
mmlu_college_med:text:10:0
vqa_rad:vqa:5:0
pathvqa:vqa:5:1000
slake:vqa:5:0
pmc_vqa:vqa:5:1000
vqa_med_2021:vqa:5:0
quilt_vqa:vqa:5:0
mimic_iii:ehr:10:0
eicu:ehr:10:0
live_qa:text:10:0
kqa_golden:text:10:0
medication_qa:text:10:0
kqa_silver:text:10:0
healthsearch_qa:text:10:1000
EOF
    echo "$(date) Initialized base model queue with 22 benchmarks"
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

    echo "$(date) Launching BASE $bench on GPU $gpu (domain=$domain, turns=$max_turns, samples=$max_samples)"

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
    echo "$(date) === Base Model Queue Monitor Check ==="

    FREE=$(get_free_gpus)
    echo "  Free GPUs:$FREE"

    remaining=$(wc -l < "$QUEUE_FILE" 2>/dev/null || echo 0)
    echo "  Queue remaining: $remaining"

    if [ "$remaining" -eq 0 ]; then
        echo "  Queue empty! All benchmarks launched."
        running=$(ps aux | grep eval_benchmark_multiturn | grep "$OUTPUT_DIR" | grep -v grep | wc -l)
        if [ "$running" -eq 0 ]; then
            echo "  All processes done. Exiting monitor."
            break
        fi
        echo "  $running processes still running. Waiting..."
    else
        for gpu in $FREE; do
            if [ "$remaining" -le 0 ]; then break; fi

            item=$(head -1 "$QUEUE_FILE")
            if [ -z "$item" ]; then break; fi

            IFS=':' read -r bench btype max_turns max_samples <<< "$item"

            if ls "${OUTPUT_DIR}/${bench}_multiturn_"*.json > /dev/null 2>&1; then
                echo "  $bench already completed, skip"
                sed -i '1d' "$QUEUE_FILE"
                remaining=$((remaining - 1))
                continue
            fi

            launch_benchmark "$gpu" "$bench" "$btype" "$max_turns" "$max_samples"
            sed -i '1d' "$QUEUE_FILE"
            remaining=$((remaining - 1))

            sleep 30
        done
    fi

    echo "  --- Running processes ---"
    ps aux | grep eval_benchmark_multiturn | grep "$OUTPUT_DIR" | grep -v grep | awk '{print "  PID=" $2 " CMD=" $NF}'

    echo "  --- Latest progress ---"
    for logfile in "$LOG_DIR"/*.log; do
        bn=$(basename "$logfile" .log)
        latest=$(grep "acc=" "$logfile" 2>/dev/null | tail -1)
        if [ -n "$latest" ]; then
            echo "  $bn: $(echo $latest | grep -oP '\[.*')"
        fi
    done

    echo "  Sleeping 15 minutes..."
    sleep 900
done

echo "$(date) === Base Model Queue Monitor Finished ==="
