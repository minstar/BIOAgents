#!/bin/bash
# ============================================================
# Healthcare AI GYM — 8-GPU Distributed Training Launcher
# ============================================================
#
# Runs 8 independent experiments in parallel, one per GPU:
#   GPU 0: Qwen3-8B-Base      → SFT multi-domain
#   GPU 1: Qwen3-4B-Instruct  → SFT multi-domain
#   GPU 2: Qwen3-4B-Thinking  → SFT multi-domain
#   GPU 3: Qwen2.5-VL-7B      → SFT multi-domain
#   GPU 4: Lingshu-7B SFT     → GRPO clinical_diagnosis
#   GPU 5: Lingshu-7B SFT     → GRPO drug_interaction
#   GPU 6: Lingshu-7B SFT     → GRPO triage_emergency (w/ safety reward)
#   GPU 7: Self-play iter3     → GRPO multi-domain
#
# Usage:
#   bash scripts/launch_8gpu_training.sh
#
# Monitor:
#   tail -f logs/8gpu_*/gpu*_*.log
#   nvidia-smi -l 5
# ============================================================

PROJECT_ROOT="/data/project/private/minstar/workspace/BIOAgents"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/8gpu_${TIMESTAMP}"

mkdir -p "$LOG_DIR"

# Save log dir path for external monitoring
echo "$LOG_DIR" > /tmp/bioagent_8gpu_logdir.txt

echo "============================================================"
echo "  Healthcare AI GYM — 8-GPU Training Launch"
echo "  Timestamp: $TIMESTAMP"
echo "  Logs: $LOG_DIR/"
echo "============================================================"

PIDS=()
NAMES=()

# ── SFT Training (GPU 0-3) ──
echo ""
echo "=== SFT Training (GPU 0-3) ==="

echo "[GPU 0] SFT Qwen3-8B-Base"
CUDA_VISIBLE_DEVICES=0 nohup python -m bioagents.training.sft_trainer \
    --config configs/8gpu/sft_qwen3_8b_gpu0.yaml \
    > "$LOG_DIR/gpu0_sft_qwen3_8b.log" 2>&1 &
PIDS+=($!); NAMES+=("gpu0_sft_qwen3_8b")

echo "[GPU 1] SFT Qwen3-4B-Instruct"
CUDA_VISIBLE_DEVICES=1 nohup python -m bioagents.training.sft_trainer \
    --config configs/8gpu/sft_qwen3_4b_instruct_gpu1.yaml \
    > "$LOG_DIR/gpu1_sft_qwen3_4b_instruct.log" 2>&1 &
PIDS+=($!); NAMES+=("gpu1_sft_qwen3_4b_instruct")

echo "[GPU 2] SFT Qwen3-4B-Thinking"
CUDA_VISIBLE_DEVICES=2 nohup python -m bioagents.training.sft_trainer \
    --config configs/8gpu/sft_qwen3_4b_thinking_gpu2.yaml \
    > "$LOG_DIR/gpu2_sft_qwen3_4b_thinking.log" 2>&1 &
PIDS+=($!); NAMES+=("gpu2_sft_qwen3_4b_thinking")

echo "[GPU 3] SFT Qwen2.5-VL-7B"
CUDA_VISIBLE_DEVICES=3 nohup python -m bioagents.training.sft_trainer \
    --config configs/8gpu/sft_qwen25vl_7b_gpu3.yaml \
    > "$LOG_DIR/gpu3_sft_qwen25vl_7b.log" 2>&1 &
PIDS+=($!); NAMES+=("gpu3_sft_qwen25vl_7b")

# ── GRPO Training (GPU 4-7) ──
echo ""
echo "=== GRPO Training (GPU 4-7) ==="

echo "[GPU 4] GRPO Lingshu → clinical_diagnosis"
CUDA_VISIBLE_DEVICES=4 nohup python -m bioagents.training.grpo_trainer \
    --config configs/8gpu/grpo_lingshu_clinical_dx_gpu4.yaml \
    > "$LOG_DIR/gpu4_grpo_lingshu_clinical_dx.log" 2>&1 &
PIDS+=($!); NAMES+=("gpu4_grpo_lingshu_clinical_dx")

echo "[GPU 5] GRPO Lingshu → drug_interaction"
CUDA_VISIBLE_DEVICES=5 nohup python -m bioagents.training.grpo_trainer \
    --config configs/8gpu/grpo_lingshu_drug_int_gpu5.yaml \
    > "$LOG_DIR/gpu5_grpo_lingshu_drug_int.log" 2>&1 &
PIDS+=($!); NAMES+=("gpu5_grpo_lingshu_drug_int")

echo "[GPU 6] GRPO Lingshu → triage (w/ safety reward)"
CUDA_VISIBLE_DEVICES=6 nohup python -m bioagents.training.grpo_trainer \
    --config configs/8gpu/grpo_lingshu_triage_gpu6.yaml \
    > "$LOG_DIR/gpu6_grpo_lingshu_triage.log" 2>&1 &
PIDS+=($!); NAMES+=("gpu6_grpo_lingshu_triage")

echo "[GPU 7] GRPO Self-play → medical_qa"
CUDA_VISIBLE_DEVICES=7 nohup python -m bioagents.training.grpo_trainer \
    --config configs/8gpu/grpo_selfplay_multidomain_gpu7.yaml \
    > "$LOG_DIR/gpu7_grpo_selfplay_multidomain.log" 2>&1 &
PIDS+=($!); NAMES+=("gpu7_grpo_selfplay_multidomain")

# ── Save PID info ──
echo ""
echo "============================================================"
echo "  All 8 jobs launched!"
echo ""
PID_FILE="$LOG_DIR/pids.txt"
for i in "${!PIDS[@]}"; do
    echo "  ${NAMES[$i]}: PID=${PIDS[$i]}"
    echo "${NAMES[$i]} ${PIDS[$i]}" >> "$PID_FILE"
done
echo ""
echo "  PID file: $PID_FILE"
echo "  Monitor: tail -f $LOG_DIR/*.log"
echo "  GPU:     nvidia-smi -l 10"
echo "============================================================"
