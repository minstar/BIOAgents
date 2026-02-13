#!/bin/bash
# ============================================================================
# Healthcare AI GYM Coach — Continuous Autonomous Training
# ============================================================================
#
# Runs the GymCoach in CONTINUOUS mode:
#   EVALUATE → ANALYZE ERRORS → DETECT PATTERNS → GENERATE DATA → TRAIN → REPEAT
#
# Key Features:
#   - Training Memory: Records ALL actions, trajectories, errors
#   - Pattern Detection: Finds and prevents recurring errors
#   - Continuous Mode: Never stops — conquers domains, auto-expands
#   - Error Analysis: 9 failure categories with targeted data generation
#
# Usage:
#   # With default config
#   bash scripts/run_gym_coach.sh --model checkpoints/8gpu/sft_qwen3_8b/merged
#
#   # With custom config
#   bash scripts/run_gym_coach.sh --config configs/gym_coach.yaml --model checkpoints/qwen3_8b_sft
#
#   # On specific GPU
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_gym_coach.sh --model checkpoints/best_model
#
# Monitor:
#   tail -f logs/gym_coach/coach.log
#   cat logs/gym_coach/training_memory/snapshot_iter_*.json
#
# ============================================================================

set -e

PROJECT_ROOT="/data/project/private/minstar/workspace/BIOAgents"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/gym_coach/coach_${TIMESTAMP}.log"
mkdir -p logs/gym_coach

echo "============================================================================"
echo "  Healthcare AI GYM Coach — Continuous Autonomous Training"
echo "  Start: $TIMESTAMP"
echo "  Log: $LOG_FILE"
echo "============================================================================"

# Pass all arguments through
python -m bioagents.gym.gym_coach "$@" 2>&1 | tee "$LOG_FILE"
