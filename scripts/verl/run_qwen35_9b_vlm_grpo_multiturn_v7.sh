#!/bin/bash
# Qwen3.5-9B Vision GRPO — Multi-Turn Tool Use (v7)
# 8x A100 80GB, FSDP with CPU offload, SGLang rollout
#
# Changes from v6:
# 1. Uses FULL VLM model (Qwen3_5ForConditionalGeneration with vision encoder)
#    v6 used text-extracted model (Qwen3NextForCausalLM, no vision encoder)
# 2. Images processed as pixel data during RL rollout
#    v6 had IMAGE: path references but no vision processing
# 3. Lower gpu_memory_utilization for SGLang (vision needs more memory)
# 4. SGLang attention_backend=triton (required for Qwen3.5's 256-dim heads)
#
# This experiment isolates format learning (v6) from genuine visual understanding (v7)
# by comparing text-path vs pixel-data VQA performance on identical benchmarks.

export PATH="/data/project/private/minstar/miniconda3/envs/verl/bin:$PATH"
export PYTHONPATH="/data/project/private/minstar/workspace/BIOAgents/scripts/verl:${PYTHONPATH:-}"
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export VLLM_USE_V1=1
export LD_LIBRARY_PATH="/data/project/private/minstar/miniconda3/envs/verl/lib:${LD_LIBRARY_PATH:-}"
# Disable strict memory check — Qwen3.5 hybrid GatedDeltaNet triggers false positive mamba pool leak detection
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false
export REWARD_DEBUG_LOG=1

cd /data/project/private/minstar/workspace/verl

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/project/private/minstar/workspace/BIOAgents/data/verl_parquet/full_4modality_vlm/train.parquet \
    data.val_files=/data/project/private/minstar/workspace/BIOAgents/data/verl_parquet/full_4modality_vlm/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key=images \
    data.return_raw_chat=True \
    data.tool_config_path=null \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=/data/project/private/minstar/workspace/BIOAgents/checkpoints/models/Qwen3.5-9B \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
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
    trainer.critic_warmup=0 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=bioagents-verl-grpo \
    trainer.experiment_name=qwen3_5_9b_vlm_grpo_multiturn_v7 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=3
