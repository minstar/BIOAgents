#!/bin/bash
# ============================================================
# Self-Distillation BT-OPD (v16) — EMA + Hints + Cosine Length Reward
# ============================================================
# Changes from v15 (EMA + hints, no length penalty):
# - COSINE_REWARD=1: cosine length-scaling reward (arXiv:2502.03373)
#   penalizes response explosion that caused v15 collapse
# - Hints ON (same as v15): v15 plateau (54.5% steps 10-20) was valuable
# - BT_OPD_MAX_TURN=0: apply OPD to ALL turns
# - max_response_length=12288: same as v15
#
# v15 findings: hints created best plateau (54.5% steps 10-20) but
#   collapsed to 49.0% by step 40 due to response explosion
#   (clip ratio 45%→91.7%, turns 7.5→4.3)
#
# v16 hypothesis: cosine length penalty + hints will:
# 1. Preserve v15's hint-driven plateau (54.5% steps 10-20)
# 2. Prevent response explosion via length penalty
# 3. Enable EMA recovery by keeping turns alive
#
# Cosine reward hyperparameters (env vars):
#   Correct: 1.1 (short) → 0.7 (long), exceed: -0.5
#   Wrong:   0.0 (short) → -0.3 (long), exceed: -0.5
# ============================================================

export PATH="/data/project/private/minstar/miniconda3/envs/verl/bin:$PATH"
export PYTHONPATH="/data/project/private/minstar/workspace/BIOAgents/scripts/verl:${PYTHONPATH:-}"
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export VLLM_USE_V1=1
export LD_LIBRARY_PATH="/data/project/private/minstar/miniconda3/envs/verl/lib:${LD_LIBRARY_PATH:-}"
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false
export REWARD_DEBUG_LOG=1

# Load environment variables (wandb key, etc.)
if [ -f /data/project/private/minstar/workspace/BIOAgents/.env ]; then
    set -a
    source /data/project/private/minstar/workspace/BIOAgents/.env
    set +a
fi

# ── BT-OPD configuration ──
export BT_OPD_MAX_TURN=0             # 0 = apply OPD to ALL turns (no truncation)
export BT_OPD_BIDIRECTIONAL=1         # Enable corrective gradient for negative trajs
export BT_OPD_MODEL_PATH=/data/project/private/minstar/workspace/BIOAgents/checkpoints/models/Qwen3.5-9B

# ── Cosine length reward (arXiv:2502.03373) ──
export COSINE_REWARD=1
export COSINE_L_MAX=12288             # Must match data.max_response_length
export COSINE_CHARS_PER_TOKEN=5.0     # Measured from actual rollouts: multi-turn XML tool calls inflate char count
export COSINE_R0_CORRECT=1.1          # Short correct → max reward
export COSINE_RL_CORRECT=0.7          # Long correct → reduced reward
export COSINE_R0_WRONG=0.0            # Short wrong → zero
export COSINE_RL_WRONG=-0.3           # Long wrong → penalty
export COSINE_R_EXCEED=-0.5           # Max length exceeded → strong penalty

# ── Hint-OPD configuration (same as v15) ──
export HINT_OPD_ENABLED=1
export HINT_OPD_DYNAMIC=1            # Teacher generates case-specific concern-style hints
# Static fallbacks (used when dynamic generation fails)
export HINT_OPD_CORRECT="The diagnostic reasoning appears sound, but could there be alternative diagnoses worth considering? Double-check whether the key findings truly support this conclusion."
export HINT_OPD_INCORRECT="Some aspects of this reasoning may need re-examination. Have you fully considered all the clinical findings? It might be worth revisiting the differential diagnosis and verifying each step of the reasoning process."

cd /data/project/private/minstar/workspace/verl

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/project/private/minstar/workspace/BIOAgents/data/verl_parquet/full_4modality_vlm/train.parquet \
    data.val_files=/data/project/private/minstar/workspace/BIOAgents/data/verl_parquet/full_4modality_vlm/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=12288 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key=images \
    data.return_raw_chat=True \
    data.tool_config_path=null \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=/data/project/private/minstar/workspace/BIOAgents/checkpoints/models/Qwen3.5-9B \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=3 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=/data/project/private/minstar/workspace/BIOAgents/scripts/verl/tool_config_full.yaml \
    actor_rollout_ref.rollout.multi_turn.inject_tool_schemas=False \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path=/data/project/private/minstar/workspace/BIOAgents/scripts/verl/reward_fn.py \
    reward.custom_reward_function.name=compute_score \
    distillation.enabled=True \
    +distillation.self_distillation=True \
    +distillation.teacher_update_interval=30 \
    +distillation.use_ema=True \
    +distillation.ema_decay=0.995 \
    +distillation.ema_update_interval=5 \
    distillation.teacher_model.model_path=/data/project/private/minstar/workspace/BIOAgents/checkpoints/models/Qwen3.5-9B \
    distillation.teacher_model.inference.name=sglang \
    distillation.teacher_model.enable_resource_pool=False \
    distillation.teacher_model.n_gpus_per_node=2 \
    distillation.teacher_model.nnodes=1 \
    distillation.teacher_model.inference.tensor_model_parallel_size=2 \
    distillation.teacher_model.inference.gpu_memory_utilization=0.25 \
    distillation.teacher_model.inference.enforce_eager=True \
    distillation.teacher_model.inference.free_cache_engine=True \
    +distillation.teacher_model.inference.engine_kwargs.sglang.attention_backend=triton \
    distillation.distillation_loss.loss_mode=bt_opd_kl \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.use_task_rewards=True \
    distillation.distillation_loss.distillation_loss_coef=4.0 \
    trainer.critic_warmup=0 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=bioagents-verl-grpo \
    trainer.experiment_name=qwen3_5_9b_self_distill_v16_ema_cosine \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.rollout_data_dir=/data/project/private/minstar/workspace/BIOAgents/rollout_dumps/v16 \
    trainer.total_epochs=3
