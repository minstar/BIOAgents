#!/bin/bash
# Continue 4-modality eval after textqa ckpt-50 (already running separately)
set -e
cd /data/project/private/minstar/workspace/BIOAgents

echo "=== Checkpoint-50: VQA === $(date)"
.venv/bin/python scripts/run_full_benchmark_suite.py \
    --model_path checkpoints/full_4modality_lingshu7b/checkpoint-50-merged \
    --category vqa \
    --output-dir results/4modality_checkpoint-50-merged

echo "=== Checkpoint-50: MedLFQA === $(date)"
.venv/bin/python scripts/run_full_benchmark_suite.py \
    --model_path checkpoints/full_4modality_lingshu7b/checkpoint-50-merged \
    --category medlfqa \
    --output-dir results/4modality_checkpoint-50-merged

echo "=== Checkpoint-50: EHR === $(date)"
.venv/bin/python scripts/run_full_benchmark_suite.py \
    --model_path checkpoints/full_4modality_lingshu7b/checkpoint-50-merged \
    --category ehr \
    --output-dir results/4modality_checkpoint-50-merged

echo "=== Checkpoint-100: TextQA === $(date)"
.venv/bin/python scripts/run_full_benchmark_suite.py \
    --model_path checkpoints/full_4modality_lingshu7b/checkpoint-100-merged \
    --category textqa \
    --output-dir results/4modality_checkpoint-100-merged

echo "=== Checkpoint-100: VQA === $(date)"
.venv/bin/python scripts/run_full_benchmark_suite.py \
    --model_path checkpoints/full_4modality_lingshu7b/checkpoint-100-merged \
    --category vqa \
    --output-dir results/4modality_checkpoint-100-merged

echo "=== Checkpoint-100: MedLFQA === $(date)"
.venv/bin/python scripts/run_full_benchmark_suite.py \
    --model_path checkpoints/full_4modality_lingshu7b/checkpoint-100-merged \
    --category medlfqa \
    --output-dir results/4modality_checkpoint-100-merged

echo "=== Checkpoint-100: EHR === $(date)"
.venv/bin/python scripts/run_full_benchmark_suite.py \
    --model_path checkpoints/full_4modality_lingshu7b/checkpoint-100-merged \
    --category ehr \
    --output-dir results/4modality_checkpoint-100-merged

echo "=== ALL DONE === $(date)"
