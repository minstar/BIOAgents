#!/bin/bash
# ============================================================
# GPU Experiment Scheduler — keeps GPUs busy with review-aligned experiments
# ============================================================
# Priority order (from paper review):
#   P1: Base+AgentRunner evals (isolate RL vs Tool/KB effect) — 1 GPU each
#   P2: Additional benchmark evals (fill Table 1 gaps) — 1 GPU each
#   P3: 3-seed training (statistical significance) — ALL 8 GPUs
# ============================================================

set -uo pipefail

BIOAGENTS_DIR="/data/project/private/minstar/workspace/BIOAgents"
PYTHON="/data/project/private/minstar/miniconda3/envs/verl/bin/python"
LOG_DIR="${BIOAGENTS_DIR}/logs"
LOCK_FILE="/tmp/gpu_scheduler.lock"
SCHEDULER_LOG="${LOG_DIR}/gpu_scheduler.log"

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "${SCHEDULER_LOG}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Prevent concurrent runs
if [ -f "${LOCK_FILE}" ]; then
    OTHER_PID=$(cat "${LOCK_FILE}")
    if kill -0 "${OTHER_PID}" 2>/dev/null; then
        log "Another scheduler running (PID=${OTHER_PID}), exiting."
        exit 0
    fi
fi
echo $$ > "${LOCK_FILE}"
trap "rm -f ${LOCK_FILE}" EXIT

# ── Get idle GPUs ──
get_idle_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | \
    awk -F',' '$2 < 1000 {print $1}' | tr -d ' '
}

# ── Check if a specific eval is already running or completed ──
is_eval_running() {
    local model_dir="$1"
    local benchmark="$2"
    ps aux | grep "eval_benchmark_multiturn" | grep -v grep | \
        grep "${model_dir}" | grep -q "${benchmark}"
}

is_eval_done() {
    local output_dir="$1"
    local benchmark="$2"
    # Check for completed result file (not partial)
    ls "${output_dir}/${benchmark}_multiturn_"*.json 2>/dev/null | head -1 | grep -q .
}

is_eval_partial() {
    local output_dir="$1"
    local benchmark="$2"
    [ -f "${output_dir}/${benchmark}_partial.json" ]
}

# ── Launch a single-GPU eval ──
launch_eval() {
    local gpu_id="$1"
    local model_path="$2"
    local benchmark="$3"
    local output_dir="$4"
    local model_label="$5"

    log "LAUNCH: GPU=${gpu_id} model=${model_label} bench=${benchmark} -> ${output_dir}"

    mkdir -p "${output_dir}/logs"

    CUDA_VISIBLE_DEVICES=${gpu_id} nohup ${PYTHON} \
        ${BIOAGENTS_DIR}/scripts/eval_benchmark_multiturn.py \
        --model_path "${model_path}" \
        --benchmarks "${benchmark}" \
        --max-turns 10 \
        --output-dir "${output_dir}" \
        --max-new-tokens 2048 \
    > "${LOG_DIR}/${model_label}_${benchmark}.log" 2>&1 &

    log "  PID=$! started on GPU ${gpu_id}"
}

# ── Launch a single-GPU text-only eval (no tools) ──
launch_textonly_eval() {
    local gpu_id="$1"
    local model_path="$2"
    local benchmark="$3"
    local output_dir="$4"
    local model_label="$5"

    log "LAUNCH TEXT-ONLY: GPU=${gpu_id} model=${model_label} bench=${benchmark} -> ${output_dir}"

    mkdir -p "${output_dir}/logs"

    CUDA_VISIBLE_DEVICES=${gpu_id} nohup ${PYTHON} \
        ${BIOAGENTS_DIR}/scripts/eval_benchmark_textonly.py \
        --model_path "${model_path}" \
        --benchmarks "${benchmark}" \
        --output-dir "${output_dir}" \
        --max-new-tokens 1024 \
    > "${LOG_DIR}/${model_label}_textonly_${benchmark}.log" 2>&1 &

    log "  PID=$! started on GPU ${gpu_id}"
}

# ── Launch 3-seed training (needs all 8 GPUs) ──
launch_3seed_training() {
    local script="$1"
    local label="$2"
    log "LAUNCH 3-SEED: ${label}"
    nohup bash "${script}" > "${LOG_DIR}/3seed_${label}.log" 2>&1 &
    log "  PID=$! started (all 8 GPUs)"
}

# ============================================================
# EXPERIMENT QUEUE — ordered by review priority
# ============================================================

BASE_MODEL="${BIOAGENTS_DIR}/checkpoints/models/Qwen3.5-9B"
GRPO_MODEL="${BIOAGENTS_DIR}/checkpoints/models/grpo_baseline_step60"
V16_MODEL="/mnt/aiplatform/csi-volumes/pvc-e668fe31-e015-4e4e-a3f4-35f18e2ad53f-bd5321b06ddb2b68ae682cc934af2027aeea25db/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_self_distill_v16_ema_cosine/global_step_60/actor/merged_hf"

BASE_AR_DIR="${BIOAGENTS_DIR}/results/benchmarks_multiturn/base_qwen35_9b_agentrunner"
BASE_TEXT_DIR="${BIOAGENTS_DIR}/results/benchmarks_textonly/base_qwen35_9b"
GRPO_DIR="${BIOAGENTS_DIR}/results/benchmarks_multiturn/grpo_baseline_step60"
V16_DIR="${BIOAGENTS_DIR}/results/benchmarks_multiturn/v16_step60"

# P0: Base text-only evals (pure model capability, no tools)
P0_BENCHMARKS=(medqa mmlu medmcqa)

# P1: Base+AgentRunner evals (RL vs Tool/KB isolation)
P1_BENCHMARKS=(medqa mmlu medmcqa vqa_rad pathvqa slake kqa_golden live_qa medication_qa mimic_iii eicu)

# P2: Fill gaps in GRPO baseline
P2_GRPO_BENCHMARKS=(vqa_rad pathvqa slake mimic_iii eicu live_qa kqa_golden)

# P2: Fill gaps in v16
P2_V16_BENCHMARKS=(mimic_iii eicu medmcqa mmlu_anatomy mmlu_clinical mmlu_professional mmlu_genetics mmlu_biology mmlu_college_med)

# ============================================================
# MAIN SCHEDULING LOGIC
# ============================================================

IDLE_GPUS=($(get_idle_gpus))
NUM_IDLE=${#IDLE_GPUS[@]}

log "=== Scheduler run: ${NUM_IDLE} idle GPUs: [${IDLE_GPUS[*]:-none}] ==="

if [ "${NUM_IDLE}" -eq 0 ]; then
    log "No idle GPUs, all busy. Nothing to do."
    exit 0
fi

# ── If ALL 8 GPUs are idle → check if 3-seed training should start ──
if [ "${NUM_IDLE}" -ge 8 ]; then
    # Check if 3-seed GRPO is needed
    GRPO_SEEDS_DONE=0
    for seed in 42 123 456; do
        if ls /mnt/aiplatform/csi-volumes/*/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_grpo_baseline_seed${seed}/global_step_*/actor/ 2>/dev/null | grep -q .; then
            GRPO_SEEDS_DONE=$((GRPO_SEEDS_DONE + 1))
        fi
    done

    if [ "${GRPO_SEEDS_DONE}" -lt 3 ]; then
        log "All 8 GPUs idle + GRPO seeds incomplete (${GRPO_SEEDS_DONE}/3) → launching 3-seed GRPO"
        launch_3seed_training "${BIOAGENTS_DIR}/scripts/verl/run_3seed_grpo_baseline.sh" "grpo_baseline"
        exit 0
    fi

    # Check if 3-seed TT-OPD is needed
    TTOPD_SEEDS_DONE=0
    for seed in 42 123 456; do
        if ls /mnt/aiplatform/csi-volumes/*/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_ttopd_v16_seed${seed}/global_step_*/actor/ 2>/dev/null | grep -q .; then
            TTOPD_SEEDS_DONE=$((TTOPD_SEEDS_DONE + 1))
        fi
    done

    if [ "${TTOPD_SEEDS_DONE}" -lt 3 ]; then
        log "All 8 GPUs idle + TT-OPD seeds incomplete (${TTOPD_SEEDS_DONE}/3) → launching 3-seed TT-OPD"
        launch_3seed_training "${BIOAGENTS_DIR}/scripts/verl/run_3seed_ttopd_v16.sh" "ttopd_v16"
        exit 0
    fi

    log "All 3-seed experiments already completed."
fi

# ── Individual GPU scheduling: fill with eval jobs ──
GPU_IDX=0

# P0: Base text-only evals (pure model capability — fast, 1 GPU each)
is_textonly_done() {
    local output_dir="$1"
    local benchmark="$2"
    ls "${output_dir}/${benchmark}_textonly_"*.json 2>/dev/null | head -1 | grep -q .
}

for bench in "${P0_BENCHMARKS[@]}"; do
    [ "${GPU_IDX}" -ge "${NUM_IDLE}" ] && break

    if is_textonly_done "${BASE_TEXT_DIR}" "${bench}"; then
        continue
    fi
    if is_eval_running "eval_benchmark_textonly" "${bench}"; then
        continue
    fi
    # Check partial progress (already started)
    if [ -f "${BASE_TEXT_DIR}/${bench}_partial.json" ]; then
        # Partial exists — already running or crashed. Let it resume.
        continue
    fi

    launch_textonly_eval "${IDLE_GPUS[${GPU_IDX}]}" "${BASE_MODEL}" "${bench}" "${BASE_TEXT_DIR}" "base_textonly"
    GPU_IDX=$((GPU_IDX + 1))
done

# P1: Base+AgentRunner evals (highest priority)
for bench in "${P1_BENCHMARKS[@]}"; do
    [ "${GPU_IDX}" -ge "${NUM_IDLE}" ] && break

    if is_eval_done "${BASE_AR_DIR}" "${bench}"; then
        continue
    fi
    if is_eval_running "Qwen3.5-9B" "${bench}"; then
        continue
    fi

    launch_eval "${IDLE_GPUS[${GPU_IDX}]}" "${BASE_MODEL}" "${bench}" "${BASE_AR_DIR}" "base_agentrunner"
    GPU_IDX=$((GPU_IDX + 1))
done

# P2: GRPO baseline gaps
for bench in "${P2_GRPO_BENCHMARKS[@]}"; do
    [ "${GPU_IDX}" -ge "${NUM_IDLE}" ] && break

    if is_eval_done "${GRPO_DIR}" "${bench}"; then
        continue
    fi
    if is_eval_running "grpo_baseline_step60" "${bench}"; then
        continue
    fi

    launch_eval "${IDLE_GPUS[${GPU_IDX}]}" "${GRPO_MODEL}" "${bench}" "${GRPO_DIR}" "grpo_baseline"
    GPU_IDX=$((GPU_IDX + 1))
done

# P2: v16 TT-OPD gaps
for bench in "${P2_V16_BENCHMARKS[@]}"; do
    [ "${GPU_IDX}" -ge "${NUM_IDLE}" ] && break

    if is_eval_done "${V16_DIR}" "${bench}"; then
        continue
    fi
    if is_eval_running "v16_step60" "${bench}"; then
        continue
    fi

    launch_eval "${IDLE_GPUS[${GPU_IDX}]}" "${V16_MODEL}" "${bench}" "${V16_DIR}" "v16_step60"
    GPU_IDX=$((GPU_IDX + 1))
done

if [ "${GPU_IDX}" -eq 0 ]; then
    log "No new jobs to launch — all queued experiments are running or complete."
else
    log "Launched ${GPU_IDX} new jobs."
fi
