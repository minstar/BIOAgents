#!/bin/bash
# Parallel evaluation across all 8 GPUs
# Each GPU runs one model instance with max batch size
set -e
cd /data/project/private/minstar/workspace/BIOAgents

CKPT_BASE="/mnt/aiplatform/csi-volumes/pvc-e668fe31-e015-4e4e-a3f4-35f18e2ad53f-bd5321b06ddb2b68ae682cc934af2027aeea25db/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_vlm_grpo_multiturn_v6"
PYTHON="/data/project/private/minstar/miniconda3/envs/verl/bin/python"

echo "$(date): Starting parallel evaluation on 8 GPUs"
echo "============================================================"

# ── Wave 1: TextQA (tool-use) on GPUs 0,1,2 + VQA on GPUs 3,4,5 + MedLFQA on GPUs 6,7 ──

# GPU 0: TextQA step 80
echo "$(date): [GPU 0] TextQA tool-use step 80"
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/eval_textqa_tooluse_format.py \
    --model_path "$CKPT_BASE/global_step_80/merged_hf" \
    --batch-size 4 --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step80 \
    > logs/parallel_textqa_step80.log 2>&1 &
PID_TQ80=$!
echo "  PID: $PID_TQ80"

# GPU 1: TextQA step 90
echo "$(date): [GPU 1] TextQA tool-use step 90"
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/eval_textqa_tooluse_format.py \
    --model_path "$CKPT_BASE/global_step_90/merged_hf" \
    --batch-size 4 --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step90 \
    > logs/parallel_textqa_step90.log 2>&1 &
PID_TQ90=$!
echo "  PID: $PID_TQ90"

# GPU 2: TextQA step 100
echo "$(date): [GPU 2] TextQA tool-use step 100"
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/eval_textqa_tooluse_format.py \
    --model_path "$CKPT_BASE/global_step_100/merged_hf" \
    --batch-size 4 --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step100 \
    > logs/parallel_textqa_step100.log 2>&1 &
PID_TQ100=$!
echo "  PID: $PID_TQ100"

# GPU 3: VQA step 80
echo "$(date): [GPU 3] VQA step 80"
CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/run_full_benchmark_suite.py \
    --model_path "$CKPT_BASE/global_step_80/merged_hf" \
    --category vqa --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step80 \
    > logs/parallel_vqa_step80.log 2>&1 &
PID_VQA80=$!
echo "  PID: $PID_VQA80"

# GPU 4: VQA step 90
echo "$(date): [GPU 4] VQA step 90"
CUDA_VISIBLE_DEVICES=4 $PYTHON scripts/run_full_benchmark_suite.py \
    --model_path "$CKPT_BASE/global_step_90/merged_hf" \
    --category vqa --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step90 \
    > logs/parallel_vqa_step90.log 2>&1 &
PID_VQA90=$!
echo "  PID: $PID_VQA90"

# GPU 5: VQA step 100
echo "$(date): [GPU 5] VQA step 100"
CUDA_VISIBLE_DEVICES=5 $PYTHON scripts/run_full_benchmark_suite.py \
    --model_path "$CKPT_BASE/global_step_100/merged_hf" \
    --category vqa --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step100 \
    > logs/parallel_vqa_step100.log 2>&1 &
PID_VQA100=$!
echo "  PID: $PID_VQA100"

# GPU 6: MedLFQA step 80
echo "$(date): [GPU 6] MedLFQA step 80"
CUDA_VISIBLE_DEVICES=6 $PYTHON scripts/run_full_benchmark_suite.py \
    --model_path "$CKPT_BASE/global_step_80/merged_hf" \
    --category medlfqa --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step80 \
    > logs/parallel_medlfqa_step80.log 2>&1 &
PID_LF80=$!
echo "  PID: $PID_LF80"

# GPU 7: MedLFQA step 90
echo "$(date): [GPU 7] MedLFQA step 90"
CUDA_VISIBLE_DEVICES=7 $PYTHON scripts/run_full_benchmark_suite.py \
    --model_path "$CKPT_BASE/global_step_90/merged_hf" \
    --category medlfqa --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step90 \
    > logs/parallel_medlfqa_step90.log 2>&1 &
PID_LF90=$!
echo "  PID: $PID_LF90"

echo ""
echo "$(date): Wave 1 launched — 8 GPUs active"
echo "  TextQA: GPUs 0,1,2 (steps 80,90,100)"
echo "  VQA:    GPUs 3,4,5 (steps 80,90,100)"
echo "  MedLFQA: GPUs 6,7 (steps 80,90)"
echo ""

# Wait for all Wave 1 to finish
wait $PID_TQ80 $PID_TQ90 $PID_TQ100 $PID_VQA80 $PID_VQA90 $PID_VQA100 $PID_LF80 $PID_LF90
echo "$(date): Wave 1 complete!"

# ── Wave 2: Remaining tasks ──
echo ""
echo "$(date): Starting Wave 2"

# GPU 0: MedLFQA step 100
echo "$(date): [GPU 0] MedLFQA step 100"
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/run_full_benchmark_suite.py \
    --model_path "$CKPT_BASE/global_step_100/merged_hf" \
    --category medlfqa --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step100 \
    > logs/parallel_medlfqa_step100.log 2>&1 &
PID_LF100=$!

# GPU 1: EHR step 80
echo "$(date): [GPU 1] EHR step 80"
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/run_full_benchmark_suite.py \
    --model_path "$CKPT_BASE/global_step_80/merged_hf" \
    --category ehr --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step80 \
    > logs/parallel_ehr_step80.log 2>&1 &
PID_EHR80=$!

# GPU 2: EHR step 90
echo "$(date): [GPU 2] EHR step 90"
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/run_full_benchmark_suite.py \
    --model_path "$CKPT_BASE/global_step_90/merged_hf" \
    --category ehr --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step90 \
    > logs/parallel_ehr_step90.log 2>&1 &
PID_EHR90=$!

# GPU 3: EHR step 100
echo "$(date): [GPU 3] EHR step 100"
CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/run_full_benchmark_suite.py \
    --model_path "$CKPT_BASE/global_step_100/merged_hf" \
    --category ehr --gpus 0 \
    --output-dir results/benchmarks_tooluse/v6_step100 \
    > logs/parallel_ehr_step100.log 2>&1 &
PID_EHR100=$!

echo "$(date): Wave 2 launched (4 GPUs)"
wait $PID_LF100 $PID_EHR80 $PID_EHR90 $PID_EHR100
echo ""
echo "============================================================"
echo "$(date): ALL EVALUATIONS COMPLETE"
echo "============================================================"
