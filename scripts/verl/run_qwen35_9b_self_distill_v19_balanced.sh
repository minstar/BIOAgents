#!/bin/bash
# ============================================================
# TT-OPD v19 — v16 Distillation Settings + v17 LR Stability
# ============================================================
# Root cause analysis: v16 peaked at 61.1% but v18 only 54.8%
# Key difference: v16 had distill_coef=4.0 + ema_interval=5
# v18 weakened distillation to coef=1.0 + interval=10
#
# v19 = v16's distillation strength + v17's LR fix:
# 1. distill_coef: 1.0 → 4.0 (restore v16's strong distillation)
# 2. ema_update_interval: 10 → 5 (restore v16's frequent updates)
# 3. ema_decay: 0.99 → 0.995 (restore v16's conservative EMA)
# 4. teacher_update_interval: 100 → 30 (restore v16's hard-copy)
# 5. save_freq: 10 → 5 (match ema_update_interval, prevent skips)
# 6. test_freq: 20 → 10 (more frequent validation)
# 7. LR=2e-7 KEPT from v17 (v16 had 1e-6, caused grad_norm 25K)
#
# v16 results: peak 61.1% at step 60, but grad_norm exploded (25K)
# v17 results: 58.8% at step 20, grad_norm stable (max 110)
# v18 results: 54.8% at step 20, distill too weak (coef=1.0)
#
# Hypothesis: v16's distillation strength + v17's LR = best of both
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

# ── BT-OPD configuration (same as v16) ──
export BT_OPD_MAX_TURN=0             # 0 = apply OPD to ALL turns
export BT_OPD_BIDIRECTIONAL=1         # Bidirectional outcome-privileged conditioning
export BT_OPD_MODEL_PATH=/data/project/private/minstar/workspace/BIOAgents/checkpoints/models/Qwen3.5-9B

# ── Cosine length reward (same as v16) ──
export COSINE_REWARD=1
export COSINE_L_MAX=12288
export COSINE_CHARS_PER_TOKEN=5.0
export COSINE_R0_CORRECT=1.1
export COSINE_RL_CORRECT=0.7
export COSINE_R0_WRONG=0.0
export COSINE_RL_WRONG=-0.3
export COSINE_R_EXCEED=-0.5

# ── Hint-OPD configuration (same as v16) ──
export HINT_OPD_ENABLED=1
export HINT_OPD_DYNAMIC=1
export HINT_OPD_CORRECT="The diagnostic reasoning appears sound, but could there be alternative diagnoses worth considering? Double-check whether the key findings truly support this conclusion."
export HINT_OPD_INCORRECT="Some aspects of this reasoning may need re-examination. Have you fully considered all the clinical findings? It might be worth revisiting the differential diagnosis and verifying each step of the reasoning process."

cd /data/project/private/minstar/workspace/verl

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/project/private/minstar/workspace/BIOAgents/data/verl_parquet/full_4modality_vlm_textqa_sub/train.parquet \
    data.val_files=/data/project/private/minstar/workspace/BIOAgents/data/verl_parquet/full_4modality_vlm_textqa_sub/test.parquet \
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
    actor_rollout_ref.actor.optim.lr=2e-7 \
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
    trainer.experiment_name=qwen3_5_9b_ttopd_v19_balanced \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.rollout_data_dir=/data/project/private/minstar/workspace/BIOAgents/rollout_dumps/v19 \
    trainer.total_epochs=1
