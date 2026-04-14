#!/bin/bash
# Wave 2: VQA + MedLFQA tool-use format evaluation on GPUs 3-7
# TextQA continues on GPUs 0-2 from previous launch
set -e
cd /data/project/private/minstar/workspace/BIOAgents

CKPT_BASE="/mnt/aiplatform/csi-volumes/pvc-e668fe31-e015-4e4e-a3f4-35f18e2ad53f-bd5321b06ddb2b68ae682cc934af2027aeea25db/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_vlm_grpo_multiturn_v6"
PYTHON="/data/project/private/minstar/miniconda3/envs/verl/bin/python"

echo "$(date): Starting VQA + MedLFQA tool-use format eval (GPUs 3-7)"
echo "============================================================"

# ── VQA tool-use on GPUs 3,4,5 (steps 80,90,100) ──

echo "$(date): [GPU 3] VQA tool-use step 80"
CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/eval_vqa_tooluse_format.py \
    --model_path "$CKPT_BASE/global_step_80/merged_hf" \
    --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step80 \
    > logs/parallel_vqa_tooluse_step80.log 2>&1 &
PID_VQA80=$!
echo "  PID: $PID_VQA80"

echo "$(date): [GPU 4] VQA tool-use step 90"
CUDA_VISIBLE_DEVICES=4 $PYTHON scripts/eval_vqa_tooluse_format.py \
    --model_path "$CKPT_BASE/global_step_90/merged_hf" \
    --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step90 \
    > logs/parallel_vqa_tooluse_step90.log 2>&1 &
PID_VQA90=$!
echo "  PID: $PID_VQA90"

echo "$(date): [GPU 5] VQA tool-use step 100"
CUDA_VISIBLE_DEVICES=5 $PYTHON scripts/eval_vqa_tooluse_format.py \
    --model_path "$CKPT_BASE/global_step_100/merged_hf" \
    --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step100 \
    > logs/parallel_vqa_tooluse_step100.log 2>&1 &
PID_VQA100=$!
echo "  PID: $PID_VQA100"

# ── MedLFQA tool-use on GPUs 6,7 (steps 80,90) ──

echo "$(date): [GPU 6] MedLFQA tool-use step 80"
CUDA_VISIBLE_DEVICES=6 $PYTHON scripts/eval_medlfqa_tooluse_format.py \
    --model_path "$CKPT_BASE/global_step_80/merged_hf" \
    --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step80 \
    > logs/parallel_medlfqa_tooluse_step80.log 2>&1 &
PID_LF80=$!
echo "  PID: $PID_LF80"

echo "$(date): [GPU 7] MedLFQA tool-use step 90"
CUDA_VISIBLE_DEVICES=7 $PYTHON scripts/eval_medlfqa_tooluse_format.py \
    --model_path "$CKPT_BASE/global_step_90/merged_hf" \
    --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step90 \
    > logs/parallel_medlfqa_tooluse_step90.log 2>&1 &
PID_LF90=$!
echo "  PID: $PID_LF90"

echo ""
echo "$(date): Wave 2A launched — 5 GPUs active (3-7)"
echo "  VQA tool-use:    GPUs 3,4,5 (steps 80,90,100)"
echo "  MedLFQA tool-use: GPUs 6,7 (steps 80,90)"
echo ""

# Wait for all Wave 2A to finish
wait $PID_VQA80 $PID_VQA90 $PID_VQA100 $PID_LF80 $PID_LF90
echo "$(date): Wave 2A complete!"

# ── Wave 2B: Remaining MedLFQA + EHR ──
echo ""
echo "$(date): Starting Wave 2B"

# GPU 3: MedLFQA step 100
echo "$(date): [GPU 3] MedLFQA tool-use step 100"
CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/eval_medlfqa_tooluse_format.py \
    --model_path "$CKPT_BASE/global_step_100/merged_hf" \
    --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step100 \
    > logs/parallel_medlfqa_tooluse_step100.log 2>&1 &
PID_LF100=$!

# GPU 4: EHR step 80
echo "$(date): [GPU 4] EHR step 80"
CUDA_VISIBLE_DEVICES=4 $PYTHON scripts/run_full_benchmark_suite.py \
    --model_path "$CKPT_BASE/global_step_80/merged_hf" \
    --category ehr --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step80 \
    > logs/parallel_ehr_step80.log 2>&1 &
PID_EHR80=$!

# GPU 5: EHR step 90
echo "$(date): [GPU 5] EHR step 90"
CUDA_VISIBLE_DEVICES=5 $PYTHON scripts/run_full_benchmark_suite.py \
    --model_path "$CKPT_BASE/global_step_90/merged_hf" \
    --category ehr --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step90 \
    > logs/parallel_ehr_step90.log 2>&1 &
PID_EHR90=$!

# GPU 6: EHR step 100
echo "$(date): [GPU 6] EHR step 100"
CUDA_VISIBLE_DEVICES=6 $PYTHON scripts/run_full_benchmark_suite.py \
    --model_path "$CKPT_BASE/global_step_100/merged_hf" \
    --category ehr --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step100 \
    > logs/parallel_ehr_step100.log 2>&1 &
PID_EHR100=$!

echo "$(date): Wave 2B launched (4 GPUs)"
wait $PID_LF100 $PID_EHR80 $PID_EHR90 $PID_EHR100
echo ""
echo "============================================================"
echo "$(date): WAVE 2 ALL COMPLETE (VQA + MedLFQA + EHR)"
echo "============================================================"
