#!/bin/bash
# ============================================================
# 3-Seed GRPO Baseline — for statistical significance
# ============================================================
# Runs vanilla GRPO baseline with 3 different seeds.
# Each run uses 8xA100, so must run sequentially.
# Seeds: 42, 123, 456
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERL_DIR="/data/project/private/minstar/workspace/verl"
BIOAGENTS_DIR="/data/project/private/minstar/workspace/BIOAgents"

export PATH="/data/project/private/minstar/miniconda3/envs/verl/bin:$PATH"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export VLLM_USE_V1=1
export LD_LIBRARY_PATH="/data/project/private/minstar/miniconda3/envs/verl/lib:${LD_LIBRARY_PATH:-}"
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false
export REWARD_DEBUG_LOG=1

# Load environment variables (wandb key, etc.)
if [ -f "${BIOAGENTS_DIR}/.env" ]; then
    set -a
    source "${BIOAGENTS_DIR}/.env"
    set +a
fi

SEEDS=(42 123 456)

run_grpo_seed() {
    local SEED=$1
    echo "============================================================"
    echo "Starting GRPO baseline seed=${SEED} at $(date)"
    echo "============================================================"

    export PYTHONHASHSEED=${SEED}

    cd "${VERL_DIR}"

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=${BIOAGENTS_DIR}/data/verl_parquet/full_4modality_vlm/train.parquet \
        data.val_files=${BIOAGENTS_DIR}/data/verl_parquet/full_4modality_vlm/test.parquet \
        data.train_batch_size=8 \
        data.max_prompt_length=8192 \
        data.max_response_length=12288 \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.image_key=images \
        data.return_raw_chat=True \
        data.tool_config_path=null \
        data.seed=${SEED} \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path=${BIOAGENTS_DIR}/checkpoints/models/Qwen3.5-9B \
        +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.data_loader_seed=${SEED} \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.n=3 \
        +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
        actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=${BIOAGENTS_DIR}/scripts/verl/tool_config_full.yaml \
        actor_rollout_ref.rollout.multi_turn.inject_tool_schemas=False \
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
        actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        reward.custom_reward_function.path=${BIOAGENTS_DIR}/scripts/verl/reward_fn.py \
        reward.custom_reward_function.name=compute_score \
        distillation.enabled=False \
        trainer.critic_warmup=0 \
        'trainer.logger=[console,wandb]' \
        trainer.project_name=bioagents-verl-grpo \
        trainer.experiment_name=qwen3_5_9b_grpo_baseline_seed${SEED} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=5 \
        trainer.test_freq=5 \
        trainer.rollout_data_dir=${BIOAGENTS_DIR}/rollout_dumps/grpo_baseline_seed${SEED} \
        trainer.total_epochs=3

    echo "GRPO baseline seed=${SEED} completed at $(date)"
}

# Run all seeds sequentially
for SEED in "${SEEDS[@]}"; do
    run_grpo_seed ${SEED}
done

echo "============================================================"
echo "All 3 GRPO baseline seeds completed at $(date)"
echo "============================================================"
