#!/bin/bash
# Full benchmark pipeline for v6 step 80/90/100
# Waits for current TextQA (PID 2613105) to finish, then runs VQA, MedLFQA, EHR

set -e
cd /data/project/private/minstar/workspace/BIOAgents

CKPT_BASE="/mnt/aiplatform/csi-volumes/pvc-e668fe31-e015-4e4e-a3f4-35f18e2ad53f-bd5321b06ddb2b68ae682cc934af2027aeea25db/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_vlm_grpo_multiturn_v6"
PYTHON="/data/project/private/minstar/miniconda3/envs/verl/bin/python"
GPUS="0,1,2,3,4,5,6,7"

STEPS=(80 90 100)
CATEGORIES=(vqa medlfqa ehr)

echo "$(date): Waiting for TextQA pipeline (PID 2613105) to finish..."
while kill -0 2613105 2>/dev/null; do
    sleep 300
    echo "$(date): TextQA still running..."
done
echo "$(date): TextQA pipeline finished!"

for cat in "${CATEGORIES[@]}"; do
    for step in "${STEPS[@]}"; do
        MODEL_PATH="$CKPT_BASE/global_step_${step}/merged_hf"
        OUT_DIR="results/benchmarks/v6_step${step}"
        echo ""
        echo "============================================================"
        echo "$(date): Starting ${cat} eval — step ${step}"
        echo "============================================================"
        $PYTHON scripts/run_full_benchmark_suite.py \
            --model_path "$MODEL_PATH" \
            --category "$cat" \
            --gpus "$GPUS" \
            --output-dir "$OUT_DIR" 2>&1
        echo "$(date): Finished ${cat} eval — step ${step}"
    done
done

echo ""
echo "============================================================"
echo "$(date): ALL BENCHMARKS COMPLETE for v6 step 80/90/100"
echo "============================================================"
