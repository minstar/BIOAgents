#!/bin/bash
# Agentic loop evaluation of v16 step 60 across all 8 training domains
# Each domain runs on its own GPU in parallel

MODEL_PATH="/data/project/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_self_distill_v16_ema_cosine/global_step_60/actor/merged_hf"
MODEL_NAME="v16_step60"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_ROOT}/logs/agentic_eval_v16_step60"
RESULT_DIR="${PROJECT_ROOT}/results/agentic_eval/v16_step60"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

cd "$PROJECT_ROOT"

# Use project venv
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

echo "========================================"
echo "  v16 Step 60 Agentic Loop Evaluation"
echo "  Model: $MODEL_PATH"
echo "  $(date)"
echo "========================================"

# Domain -> GPU mapping (8 domains, 8 GPUs)
# medical_qa has 1000 tasks, use 200 for reasonable time
# Others use all available tasks

run_domain() {
    local gpu=$1
    local domain=$2
    local num_tasks=$3
    local logfile="${LOG_DIR}/${domain}.log"

    echo "[GPU $gpu] Starting $domain ($num_tasks tasks) -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON scripts/eval_agentic_sft_vs_rl.py \
        --custom-models "${MODEL_NAME}=${MODEL_PATH}" \
        --domain "$domain" \
        --num-tasks "$num_tasks" \
        --max-turns 10 \
        --max-new-tokens 1024 \
        > "$logfile" 2>&1 &
}

# Launch all 8 domains in parallel
run_domain 0 medical_qa 200
run_domain 1 clinical_diagnosis 5
run_domain 2 visual_diagnosis 8
run_domain 3 drug_interaction 5
run_domain 4 obstetrics 20
run_domain 5 triage_emergency 20
run_domain 6 psychiatry 20
run_domain 7 ehr_management 15

echo ""
echo "All 8 domains launched. Waiting for completion..."
echo "Monitor: tail -f ${LOG_DIR}/*.log"
echo ""

wait
echo ""
echo "========================================"
echo "  All domains completed! $(date)"
echo "========================================"

# Collect results
echo ""
echo "Results:"
for f in results/agentic_eval/*.json; do
    if [ -f "$f" ]; then
        echo "  $f"
    fi
done
