#!/bin/bash
# Auto-redistribute GPUs when TextQA finishes MedQA
# Checks every 2 minutes, kills TextQA MedMCQA if running, gives GPUs to VQA

CKPT_BASE="/mnt/aiplatform/csi-volumes/pvc-e668fe31-e015-4e4e-a3f4-35f18e2ad53f-bd5321b06ddb2b68ae682cc934af2027aeea25db/private/minstar/workspace/verl/checkpoints/bioagents-verl-grpo/qwen3_5_9b_vlm_grpo_multiturn_v6"
PYTHON="/data/project/private/minstar/miniconda3/envs/verl/bin/python"
WORKDIR="/data/project/private/minstar/workspace/BIOAgents"

echo "$(date): Waiting for TextQA to finish..."

while true; do
    # Check if TextQA processes are still running
    textqa_count=$(ps aux | grep "eval_textqa_tooluse" | grep -v grep | wc -l)
    
    if [ "$textqa_count" -eq 0 ]; then
        echo "$(date): TextQA finished! Redistributing GPUs 0-2 to VQA..."
        
        # Check if VQA steps already running on these GPUs
        cd "$WORKDIR"
        
        # Launch additional VQA processes on freed GPUs
        # We can run vqa_rad subsets or other VQA benchmarks
        echo "$(date): GPUs 0-2 freed. VQA already running on 3-5."
        echo "$(date): Starting MedLFQA step 100 on GPU 0, EHR steps 80/90/100 on GPUs 1/2"
        
        # MedLFQA step 100
        CUDA_VISIBLE_DEVICES=0 nohup $PYTHON scripts/eval_medlfqa_tooluse_format.py \
            --model_path ${CKPT_BASE}/global_step_100/merged_hf \
            --gpus 0 --output-dir results/benchmarks_tooluse/v6_step100 \
            > logs/parallel_medlfqa_tooluse_step100_v2.log 2>&1 &
        echo "$(date): MedLFQA step100 PID: $!"
        
        # EHR step 80
        CUDA_VISIBLE_DEVICES=1 nohup $PYTHON scripts/run_full_benchmark_suite.py \
            --model_path ${CKPT_BASE}/global_step_80/merged_hf \
            --benchmarks ehr --gpus 0 \
            --output-dir results/benchmarks_tooluse/v6_step80 \
            > logs/parallel_ehr_step80.log 2>&1 &
        echo "$(date): EHR step80 PID: $!"
        
        # EHR step 90
        CUDA_VISIBLE_DEVICES=2 nohup $PYTHON scripts/run_full_benchmark_suite.py \
            --model_path ${CKPT_BASE}/global_step_90/merged_hf \
            --benchmarks ehr --gpus 0 \
            --output-dir results/benchmarks_tooluse/v6_step90 \
            > logs/parallel_ehr_step90.log 2>&1 &
        echo "$(date): EHR step90 PID: $!"
        
        echo "$(date): Redistribution complete!"
        exit 0
    fi
    
    echo "$(date): TextQA still running ($textqa_count processes)..."
    sleep 120
done
