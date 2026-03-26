#!/bin/bash
# Step3-VL TextQA + MedLFQA sequential runner (B036 fix applied)
# GPU 4 only

set -e
export CUDA_VISIBLE_DEVICES=4
cd /data/project/private/minstar/workspace/BIOAgents

echo "=== Step3-VL TextQA starting ($(date)) ==="
.venv/bin/python scripts/run_full_benchmark_suite.py \
    --model step3vl \
    --category textqa \
    --output-dir logs/step3vl_textqa_20260318
echo "=== Step3-VL TextQA completed ($(date)) ==="

echo "=== Step3-VL MedLFQA starting ($(date)) ==="
.venv/bin/python scripts/run_full_benchmark_suite.py \
    --model step3vl \
    --category medlfqa \
    --output-dir logs/step3vl_medlfqa_20260318
echo "=== Step3-VL MedLFQA completed ($(date)) ==="
