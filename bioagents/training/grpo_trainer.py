"""BIOAgents GRPO Trainer — TRL-integrated multi-turn RL training.

Trains medical/biomedical agents using Group Relative Policy Optimization (GRPO)
with domain-specific reward functions (accuracy, format, process).

Usage:
    python bioagents/training/grpo_trainer.py --config configs/grpo_medical_qa.yaml
    accelerate launch bioagents/training/grpo_trainer.py --config configs/grpo_medical_qa.yaml

Reference: TRL GRPOTrainer, AgentGym-RL, MRPO framework
"""

import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from loguru import logger

# ============================================================
# Config
# ============================================================


@dataclass
class BioAgentGRPOConfig:
    """Full GRPO training configuration."""

    # Model
    model_name_or_path: str = "Qwen/Qwen3-1.7B"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # PEFT / LoRA (r=64 per MediX-R1/GSPO best practice)
    peft_enabled: bool = True
    peft_r: int = 64
    peft_lora_alpha: int = 128
    peft_lora_dropout: float = 0.05
    peft_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Dataset
    domain: str = "medical_qa"
    tasks_path: str = "data/domains/medical_qa/tasks.json"
    split_tasks_path: str = ""
    train_split: str = "train"
    eval_split: str = "test"
    max_prompt_length: int = 2048
    max_completion_length: int = 1024

    # Training
    output_dir: str = "checkpoints/grpo_medical_qa"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50
    save_total_limit: int = 3
    seed: int = 42

    # GRPO-specific (beta=0.01 per GSPO/MediX-R1 best practice)
    num_generations: int = 4
    beta: float = 0.01
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50

    # Reward
    reward_functions: list = field(
        default_factory=lambda: [
            {"name": "accuracy", "weight": 0.4},
            {"name": "format", "weight": 0.2},
            {"name": "process", "weight": 0.4},
        ]
    )
    bertscore_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

    # Reward Strategy (GRPO / MRPO / SARL / DRPO / CRPO / Adaptive)
    reward_strategy: str = "grpo"  # "grpo", "mrpo", "sarl", "drpo", "crpo", "adaptive"
    reward_strategy_config: dict = field(default_factory=dict)

    # Curriculum Learning
    use_curriculum: bool = False                  # Enable curriculum learning
    curriculum_min_tasks_per_level: int = 5       # Tasks before considering promotion
    curriculum_warmup_cycles: int = 2             # Warm-up at Level 1

    # Eval Trajectory Replay (WebRL-style, arXiv:2411.02337)
    # Pre-collected high-reward eval trajectories mixed into training batch
    # to bootstrap learning from successful demonstrations.
    eval_trajectories: list | None = field(default=None)
    eval_trajectory_mix_ratio: float = 0.3       # fraction of batch from eval pool
    eval_trajectory_min_reward: float = 0.5      # only replay trajectories above this reward

    # Task cap: limit number of training tasks per GRPO cycle to prevent
    # overfitting on small domain datasets. 0 = use all tasks.
    max_train_tasks: int = 20

    # Dr. GRPO (2025): remove per-token length normalization
    # Standard GRPO uses mean log-prob (biased toward short responses).
    # Dr. GRPO uses sum log-prob, eliminating length bias in medical answers.
    use_dr_grpo: bool = False

    # ── GSPO / DAPO / Dr.GRPO (TRL 0.28 native) ──────────────
    # loss_type: "grpo" (standard), "dapo" (DAPO), "dr_grpo" (Dr.GRPO)
    # DAPO = Clip-Higher + token-level PG without length normalization
    # Dr.GRPO = removes length/std normalization bias
    loss_type: str = "dapo"

    # Clip-Higher (DAPO): asymmetric clipping for exploration
    # epsilon_low (= epsilon) clips ratio below, epsilon_high clips above
    # Higher epsilon_high allows more upside exploration, preventing entropy collapse
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28

    # Importance sampling level: "token" (standard GRPO) or "sequence" (GSPO)
    # GSPO uses sequence-level ratios — eliminates high-variance token-level noise
    # that causes progressive model collapse (B041 root cause)
    importance_sampling_level: str = "sequence"

    # Reward scaling: "group" (standard), "none" (Dr.GRPO removes this)
    # Dr.GRPO paper shows group normalization introduces bias
    scale_rewards: str = "none"

    # Dynamic Sampling (DAPO technique #2):
    # Filter out prompts where all generations are correct or all wrong
    # Only train on prompts with mixed outcomes (informative gradients)
    dynamic_sampling: bool = True
    dynamic_sampling_min_mixed_ratio: float = 0.1  # min ratio of mixed prompts

    # Overlong reward shaping (DAPO technique #4):
    # Linear penalty for responses exceeding safe_length tokens
    overlong_reward_shaping: bool = True
    overlong_safe_length: int = 768       # tokens before penalty kicks in
    overlong_penalty_factor: float = 0.5  # penalty = factor * (len - safe) / max_len

    # Environment
    max_turns: int = 10
    use_gym_env: bool = True

    # Logging
    wandb_project: str = "pt2-minstar-gym-rl"
    run_name: str = "grpo_medical_qa"
    use_wandb: bool = True
    log_dir: str = "logs/runs"

    @classmethod
    def from_yaml(cls, path: str) -> "BioAgentGRPOConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        kwargs = {}
        # Flatten nested YAML into flat config
        if "model" in raw:
            kwargs["model_name_or_path"] = raw["model"].get("name_or_path", cls.model_name_or_path)
            kwargs["torch_dtype"] = raw["model"].get("torch_dtype", cls.torch_dtype)
            kwargs["attn_implementation"] = raw["model"].get("attn_implementation", cls.attn_implementation)
        if "peft" in raw:
            kwargs["peft_enabled"] = raw["peft"].get("enabled", cls.peft_enabled)
            kwargs["peft_r"] = raw["peft"].get("r", cls.peft_r)
            kwargs["peft_lora_alpha"] = raw["peft"].get("lora_alpha", cls.peft_lora_alpha)
            kwargs["peft_lora_dropout"] = raw["peft"].get("lora_dropout", cls.peft_lora_dropout)
            kwargs["peft_target_modules"] = raw["peft"].get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        if "dataset" in raw:
            kwargs["domain"] = raw["dataset"].get("domain", cls.domain)
            kwargs["tasks_path"] = raw["dataset"].get("tasks_path", cls.tasks_path)
            kwargs["split_tasks_path"] = raw["dataset"].get("split_tasks_path", "")
            kwargs["train_split"] = raw["dataset"].get("train_split", cls.train_split)
            kwargs["eval_split"] = raw["dataset"].get("eval_split", cls.eval_split)
            kwargs["max_prompt_length"] = raw["dataset"].get("max_prompt_length", cls.max_prompt_length)
            kwargs["max_completion_length"] = raw["dataset"].get("max_completion_length", cls.max_completion_length)
        if "training" in raw:
            t = raw["training"]
            for key in [
                "output_dir", "num_train_epochs", "per_device_train_batch_size",
                "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
                "warmup_ratio", "weight_decay", "max_grad_norm", "bf16",
                "logging_steps", "save_steps", "eval_steps", "save_total_limit", "seed",
                "num_generations", "beta", "temperature", "top_p", "top_k",
                # GSPO/DAPO/Dr.GRPO fields
                "loss_type", "epsilon_low", "epsilon_high",
                "importance_sampling_level", "scale_rewards",
                "dynamic_sampling", "dynamic_sampling_min_mixed_ratio",
                "overlong_reward_shaping", "overlong_safe_length", "overlong_penalty_factor",
                "use_dr_grpo",
            ]:
                if key in t:
                    kwargs[key] = t[key]
        if "rewards" in raw:
            kwargs["reward_functions"] = raw["rewards"].get("functions", [])
            kwargs["bertscore_model"] = raw["rewards"].get("bertscore_model", cls.bertscore_model)
            kwargs["reward_strategy"] = raw["rewards"].get("strategy", cls.reward_strategy)
            kwargs["reward_strategy_config"] = raw["rewards"].get("strategy_config", {})
        if "environment" in raw:
            kwargs["max_turns"] = raw["environment"].get("max_turns", cls.max_turns)
            kwargs["use_gym_env"] = raw["environment"].get("use_gym_env", cls.use_gym_env)
        if "logging" in raw:
            kwargs["wandb_project"] = raw["logging"].get("project", cls.wandb_project)
            kwargs["run_name"] = raw["logging"].get("run_name", cls.run_name)
            kwargs["use_wandb"] = raw["logging"].get("use_wandb", cls.use_wandb)
            kwargs["log_dir"] = raw["logging"].get("log_dir", cls.log_dir)

        return cls(**kwargs)


# ============================================================
# Dataset preparation
# ============================================================


def build_grpo_dataset(config: BioAgentGRPOConfig, split: str = "train"):
    """Build a HuggingFace Dataset from BIOAgents tasks for GRPO training.

    Each example becomes a prompt that the model generates completions for.
    The reward is computed by the reward functions after generation.

    Returns:
        datasets.Dataset with 'prompt' and metadata columns
    """
    from datasets import Dataset

    # Load tasks
    tasks_path = Path(config.tasks_path)
    with open(tasks_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    # Apply split filtering
    if config.split_tasks_path and split:
        split_file = Path(config.split_tasks_path)
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                splits = json.load(f)
            if split in splits:
                valid_ids = set(splits[split])
                all_tasks = [t for t in all_tasks if t["id"] in valid_ids]
                logger.info(f"Filtered to {len(all_tasks)} tasks for split '{split}'")

    if not all_tasks:
        raise ValueError(f"No tasks found for split '{split}' in {tasks_path}")

    # Build prompts from tasks
    records = []
    for task in all_tasks:
        prompt = _build_prompt_from_task(task, config.domain)
        correct_answer = task.get("correct_answer", "")
        task_id = task.get("id", "")

        records.append({
            "prompt": prompt,
            "solution": correct_answer,
            "task_id": task_id,
            "domain": config.domain,
        })

    dataset = Dataset.from_list(records)
    logger.info(f"Built {split} dataset: {len(dataset)} examples")
    return dataset


def _build_prompt_from_task(task: dict, domain: str) -> list[dict]:
    """Build a chat-format prompt from a task dict.

    Returns a list of message dicts for the tokenizer.apply_chat_template().
    """
    ticket = task.get("ticket", "")
    description = task.get("description", {})

    if domain == "medical_qa":
        system_msg = (
            "You are a medical AI assistant that answers medical questions using "
            "evidence-based reasoning. Use tools to search for evidence, then "
            "submit your answer with clear reasoning.\n\n"
            "Available tools: search_pubmed, browse_article, search_medical_wiki, "
            "browse_wiki_entry, retrieve_evidence, analyze_answer_options, think, submit_answer.\n\n"
            "To call a tool, respond with JSON: {\"name\": \"tool_name\", \"arguments\": {...}}\n"
            "When ready, use submit_answer to provide your final answer."
        )
    elif domain == "drug_interaction":
        system_msg = (
            "You are a clinical pharmacology AI assistant specializing in drug-drug "
            "interaction assessment. Review medication profiles, check interactions, "
            "and provide management recommendations.\n\n"
            "Available tools: get_patient_medications, get_drug_info, check_interaction, "
            "check_all_interactions, search_alternatives, check_dosage, "
            "search_drugs_by_class, think, submit_answer.\n\n"
            "To call a tool, respond with JSON: {\"name\": \"tool_name\", \"arguments\": {...}}\n"
            "When done, use submit_answer to provide your recommendation."
        )
    elif domain == "visual_diagnosis":
        system_msg = (
            "You are a medical AI assistant specializing in visual diagnosis. "
            "Analyze medical images, interpret reports, and answer visual questions.\n\n"
            "Available tools: get_image_metadata, get_image_report, analyze_image, "
            "compare_images, search_similar_cases, answer_visual_question, think.\n\n"
            "To call a tool, respond with JSON: {\"name\": \"tool_name\", \"arguments\": {...}}"
        )
    elif domain == "clinical_diagnosis":
        system_msg = (
            "You are a medical AI assistant for clinical diagnosis. Use tools to "
            "review patient records, order tests, and make clinical recommendations.\n\n"
            "To call a tool, respond with JSON: {\"name\": \"tool_name\", \"arguments\": {...}}"
        )
    else:
        system_msg = "You are a medical AI assistant."

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": ticket},
    ]


# ============================================================
# Reward function integration
# ============================================================


def build_reward_functions(config: BioAgentGRPOConfig) -> list:
    """Build GRPO-compatible reward functions from config.

    Supports two modes:
    1. Legacy: list of named reward functions from GRPO_REWARD_REGISTRY
    2. Strategy-based: use reward_strategy to select GRPO/MRPO/SARL/Adaptive

    Returns:
        List of callables matching TRL GRPOTrainer reward_funcs signature:
            fn(completions, **kwargs) -> list[float]
    """
    # --- Strategy-based reward (new system) ---
    if config.reward_strategy and config.reward_strategy != "grpo":
        return _build_strategy_reward_functions(config)

    # --- Legacy: individual reward functions ---
    from bioagents.evaluation.grpo_rewards import GRPO_REWARD_REGISTRY

    reward_fns = []
    for rw_spec in config.reward_functions:
        name = rw_spec["name"]
        if name not in GRPO_REWARD_REGISTRY:
            raise ValueError(f"Unknown reward function '{name}'. Available: {list(GRPO_REWARD_REGISTRY.keys())}")

        fn = GRPO_REWARD_REGISTRY[name]
        reward_fns.append(fn)
        logger.info(f"  Reward function: {name} (weight applied inside composite)")

    return reward_fns


def _build_strategy_reward_functions(config: BioAgentGRPOConfig) -> list:
    """Build reward functions using the adaptive reward strategy system.

    Creates a single reward function from the selected strategy (MRPO, SARL, etc.)
    that encapsulates the entire reward computation pipeline.
    """
    from bioagents.evaluation.reward_strategies import (
        RewardStrategyConfig,
        create_reward_strategy,
        make_grpo_reward_fn,
    )

    # Build strategy config from YAML overrides
    strategy_kwargs = {}
    sc = config.reward_strategy_config or {}

    # Map YAML keys to RewardStrategyConfig fields
    if "grpo_weights" in sc:
        strategy_kwargs["grpo_weights"] = sc["grpo_weights"]
    if "mrpo_alignment_weight" in sc:
        strategy_kwargs["mrpo_alignment_weight"] = sc["mrpo_alignment_weight"]
    if "mrpo_relevance_weight" in sc:
        strategy_kwargs["mrpo_relevance_weight"] = sc["mrpo_relevance_weight"]
    if "mrpo_factuality_weight" in sc:
        strategy_kwargs["mrpo_factuality_weight"] = sc["mrpo_factuality_weight"]
    if "sarl_alpha" in sc:
        strategy_kwargs["sarl_alpha"] = sc["sarl_alpha"]
    if "sarl_lambda" in sc:
        strategy_kwargs["sarl_lambda"] = sc["sarl_lambda"]
    if "sarl_tool_bonus_per_call" in sc:
        strategy_kwargs["sarl_tool_bonus_per_call"] = sc["sarl_tool_bonus_per_call"]
    if "sarl_tool_bonus_cap" in sc:
        strategy_kwargs["sarl_tool_bonus_cap"] = sc["sarl_tool_bonus_cap"]

    # Also pull weights from reward_functions list if provided
    if config.reward_functions and "grpo_weights" not in strategy_kwargs:
        weights = {}
        for rf in config.reward_functions:
            weights[rf["name"]] = rf["weight"]
        if weights:
            strategy_kwargs["grpo_weights"] = weights

    strategy_config = RewardStrategyConfig(**strategy_kwargs)
    strategy = create_reward_strategy(config.reward_strategy, config=strategy_config)

    logger.info(f"  Reward strategy: {strategy.name}")
    logger.info(f"  Strategy type: {config.reward_strategy}")
    if strategy_kwargs:
        logger.info(f"  Strategy config overrides: {strategy_kwargs}")

    # Create TRL-compatible reward function
    reward_fn = make_grpo_reward_fn(strategy)
    return [reward_fn]


# ============================================================
# DAPO: Dynamic Sampling Callback & Overlong Reward Shaping
# ============================================================


def _wrap_overlong_reward(reward_funcs, safe_length: int, max_length: int, penalty_factor: float):
    """Wrap reward functions with overlong penalty (DAPO technique #4).

    Applies a linear penalty to completions exceeding safe_length tokens.
    penalty = -penalty_factor * (completion_len - safe_length) / max_length
    """
    wrapped = []
    for fn in reward_funcs:
        def make_wrapper(original_fn):
            def wrapper(completions, **kwargs):
                rewards = original_fn(completions, **kwargs)
                for i, completion in enumerate(completions):
                    # Estimate token count from characters (rough: 1 token ≈ 4 chars)
                    comp_text = completion if isinstance(completion, str) else str(completion)
                    est_tokens = len(comp_text) // 4
                    if est_tokens > safe_length:
                        overshoot = (est_tokens - safe_length) / max_length
                        penalty = -penalty_factor * overshoot
                        rewards[i] = rewards[i] + penalty
                return rewards
            return wrapper
        wrapped.append(make_wrapper(fn))
    return wrapped


class DynamicSamplingCallback:
    """DAPO Dynamic Sampling (technique #2).

    Monitors reward distributions within generation groups and logs
    statistics about mixed vs. homogeneous groups. TRL handles the
    actual training loop, but we can use this callback to track
    how many prompts produce informative (mixed) vs. uninformative
    (all-correct or all-wrong) batches.

    In DAPO, uninformative groups are filtered out. TRL doesn't
    support per-group filtering natively, but with sequence-level
    importance sampling + Clip-Higher, the gradient contribution
    from homogeneous groups is naturally minimized (advantages ≈ 0).
    This callback logs the ratio for monitoring.
    """

    def __init__(self, min_mixed_ratio: float = 0.1):
        self.min_mixed_ratio = min_mixed_ratio
        self.step_count = 0
        self.total_groups = 0
        self.mixed_groups = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log dynamic sampling statistics."""
        if logs and self.total_groups > 0:
            mixed_ratio = self.mixed_groups / self.total_groups
            logs["dynamic_sampling/mixed_ratio"] = mixed_ratio
            logs["dynamic_sampling/total_groups"] = self.total_groups
            if mixed_ratio < self.min_mixed_ratio:
                logger.warning(
                    f"Dynamic sampling: mixed ratio {mixed_ratio:.2%} below "
                    f"threshold {self.min_mixed_ratio:.2%}. Consider adjusting "
                    f"task difficulty or num_generations."
                )


# ============================================================
# Main trainer
# ============================================================


def train(config: BioAgentGRPOConfig):
    """Run GRPO training with the given configuration.

    Compatible with TRL >= 0.28.0 GRPOTrainer API.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    logger.info("=" * 60)
    logger.info("BIOAgents GRPO Trainer (TRL 0.28+)")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name_or_path}")
    logger.info(f"Domain: {config.domain}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Epochs: {config.num_train_epochs}")
    logger.info(f"Batch: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps}")
    logger.info(f"Num generations (G): {config.num_generations}")
    logger.info(f"Beta (KL): {config.beta}")
    logger.info(f"Temperature: {config.temperature}")

    # --- Tokenizer ---
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model ---
    logger.info("Loading model...")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    # Detect model type to choose correct Auto class
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model_type = getattr(model_config, "model_type", "")
    is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl")

    load_kwargs = dict(torch_dtype=model_dtype, trust_remote_code=True)
    if config.attn_implementation and not is_qwen_vl:
        load_kwargs["attn_implementation"] = config.attn_implementation

    if is_qwen_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration
        logger.info(f"Detected VL model (type={model_type}), using Qwen2_5_VLForConditionalGeneration")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name_or_path, **load_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, **load_kwargs,
        )

    # --- PEFT / LoRA ---
    peft_config = None
    if config.peft_enabled:
        from peft import LoraConfig, TaskType

        peft_config = LoraConfig(
            r=config.peft_r,
            lora_alpha=config.peft_lora_alpha,
            lora_dropout=config.peft_lora_dropout,
            target_modules=config.peft_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        logger.info(f"LoRA config: r={config.peft_r}, alpha={config.peft_lora_alpha}")

    # --- Dataset ---
    logger.info("Building datasets...")
    train_dataset = build_grpo_dataset(config, split=config.train_split)

    eval_dataset = None
    if config.eval_split:
        try:
            eval_dataset = build_grpo_dataset(config, split=config.eval_split)
        except ValueError:
            logger.warning(f"No eval dataset for split '{config.eval_split}', skipping eval.")

    # --- Reward Functions ---
    logger.info("Setting up reward functions...")
    reward_funcs = build_reward_functions(config)

    # --- W&B Init for TRL (set env vars before GRPOConfig) ---
    if config.use_wandb:
        os.environ["WANDB_PROJECT"] = config.wandb_project
        # Also init via our centralized logger for consistency
        from bioagents.utils.wandb_logger import GymWandbLogger, WANDB_PROJECT
        _wb_project = config.wandb_project if config.wandb_project != "bioagents-grpo" else WANDB_PROJECT
        os.environ["WANDB_PROJECT"] = _wb_project
        logger.info(f"W&B project: {_wb_project}, run: {config.run_name}")

    # --- GRPO Training Config (TRL 0.28+) ---
    os.makedirs(config.output_dir, exist_ok=True)

    # Resolve loss_type: use_dr_grpo flag overrides for backward compat
    effective_loss_type = config.loss_type
    if config.use_dr_grpo and effective_loss_type == "grpo":
        effective_loss_type = "dr_grpo"

    logger.info(f"Loss type: {effective_loss_type}")
    logger.info(f"Importance sampling: {config.importance_sampling_level}")
    logger.info(f"Epsilon (low/high): {config.epsilon_low}/{config.epsilon_high}")
    logger.info(f"Scale rewards: {config.scale_rewards}")
    logger.info(f"Dynamic sampling: {config.dynamic_sampling}")

    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        # GRPO-specific (TRL 0.28)
        num_generations=config.num_generations,
        beta=config.beta,
        temperature=config.temperature,
        max_completion_length=config.max_completion_length,
        # ── GSPO/DAPO/Dr.GRPO (TRL 0.28 native) ──
        loss_type=effective_loss_type,
        epsilon=config.epsilon_low,
        epsilon_high=config.epsilon_high,
        importance_sampling_level=config.importance_sampling_level,
        scale_rewards=config.scale_rewards,
        # Logging
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.run_name,
        logging_dir=config.log_dir,
    )

    # --- Overlong Reward Shaping (DAPO technique #4) ---
    # Wrap reward functions to penalize excessively long completions
    if config.overlong_reward_shaping:
        reward_funcs = _wrap_overlong_reward(
            reward_funcs,
            safe_length=config.overlong_safe_length,
            max_length=config.max_completion_length,
            penalty_factor=config.overlong_penalty_factor,
        )
        logger.info(
            f"Overlong reward shaping: safe={config.overlong_safe_length}, "
            f"max={config.max_completion_length}, penalty={config.overlong_penalty_factor}"
        )

    # --- Dynamic Sampling Callback (DAPO technique #2) ---
    callbacks = []
    if config.dynamic_sampling:
        callbacks.append(
            DynamicSamplingCallback(
                min_mixed_ratio=config.dynamic_sampling_min_mixed_ratio,
            )
        )
        logger.info(f"Dynamic sampling enabled (min_mixed_ratio={config.dynamic_sampling_min_mixed_ratio})")

    # --- Trainer (TRL 0.28 API) ---
    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks if callbacks else None,
    )

    # --- Train ---
    logger.info("Starting GRPO training...")
    trainer.train()

    # --- Save final model ---
    logger.info(f"Saving final model to {config.output_dir}/final")
    trainer.save_model(os.path.join(config.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final"))

    # --- Save config for reproducibility ---
    config_save_path = os.path.join(config.output_dir, "training_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(vars(config), f, default_flow_style=False)
    logger.info(f"Config saved to {config_save_path}")

    logger.info("✅ GRPO training complete!")
    return trainer


# ============================================================
# FairGRPO Training (Fairness-Aware GRPO)
# ============================================================
# Reference: FairGRPO (arXiv:2510.19893) — Hierarchical RL approach
# for equitable clinical reasoning with demographic-aware weighting.


@dataclass
class FairGRPOConfig(BioAgentGRPOConfig):
    """FairGRPO configuration — extends BioAgentGRPOConfig with fairness params."""

    # Fairness parameters
    fairness_enabled: bool = True
    fairness_weight: float = 0.1        # Weight of fairness component in composite reward
    alpha_repr: float = 0.5             # Representation-aware weight (upweight rare groups)
    alpha_perf: float = 0.5             # Performance-aware weight (upweight struggling groups)
    fairness_log_interval: int = 50     # Steps between fairness metric logging
    fairness_reset_per_epoch: bool = True  # Reset tracker each epoch
    demographic_features: list = field(
        default_factory=lambda: ["age_group", "sex", "ethnicity"]
    )
    # Fairness targets
    max_fairness_gap: float = 0.15      # Target max gap between groups
    fairness_penalty_scale: float = 0.5  # Scale of penalty when gap exceeds target


def train_fair_grpo(config: FairGRPOConfig):
    """Run FairGRPO training with demographic-aware reward weighting.

    Extends standard GRPO training with:
    1. Demographic group extraction from patient data
    2. Fairness-aware reward weighting (representation + performance)
    3. Fairness gap monitoring and logging
    4. Adaptive emphasis on underperforming demographic groups

    This implements the FairGRPO framework from arXiv:2510.19893:
    - Adaptive importance weighting of GRPO advantages
    - Unsupervised demographic group discovery (when labels missing)
    - Progressive fairness improvement during training
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    logger.info("=" * 60)
    logger.info("BIOAgents FairGRPO Trainer (Fairness-Aware)")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name_or_path}")
    logger.info(f"Domain: {config.domain}")
    logger.info(f"Fairness weight: {config.fairness_weight}")
    logger.info(f"Alpha (repr): {config.alpha_repr}, Alpha (perf): {config.alpha_perf}")
    logger.info(f"Max fairness gap target: {config.max_fairness_gap}")

    # --- Setup model + tokenizer (same as standard GRPO) ---
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        attn_implementation=config.attn_implementation if config.attn_implementation else None,
    )

    # --- PEFT ---
    peft_config = None
    if config.peft_enabled:
        from peft import LoraConfig, TaskType
        peft_config = LoraConfig(
            r=config.peft_r,
            lora_alpha=config.peft_lora_alpha,
            lora_dropout=config.peft_lora_dropout,
            target_modules=config.peft_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

    # --- Dataset with demographic metadata ---
    train_dataset = build_grpo_dataset(config, split=config.train_split)

    # Load patient data for demographic extraction
    patient_data_map = _load_patient_demographics(config)
    logger.info(f"Loaded demographics for {len(patient_data_map)} patients")

    # --- FairGRPO Reward Functions ---
    from bioagents.evaluation.grpo_rewards import (
        grpo_fair_composite_reward,
        get_fairness_tracker,
    )

    tracker = get_fairness_tracker()
    tracker.reset()

    # Create a fairness-aware reward function that wraps the composite reward
    def fair_reward_fn(completions, solution=None, **kwargs):
        """Fairness-aware composite reward for GRPO training."""
        # Extract patient demographics from task metadata
        task_ids = kwargs.get("task_id", [])
        pd_list = []
        for tid in task_ids:
            pd_list.append(patient_data_map.get(tid, {}))

        return grpo_fair_composite_reward(
            completions,
            solution=solution,
            patient_data=pd_list if pd_list else None,
            fairness_weight=config.fairness_weight,
            alpha_repr=config.alpha_repr,
            alpha_perf=config.alpha_perf,
            **kwargs,
        )

    # --- GRPO Config (with GSPO/DAPO support) ---
    os.makedirs(config.output_dir, exist_ok=True)
    effective_loss_type = config.loss_type
    if config.use_dr_grpo and effective_loss_type == "grpo":
        effective_loss_type = "dr_grpo"

    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        num_generations=config.num_generations,
        beta=config.beta,
        temperature=config.temperature,
        max_completion_length=config.max_completion_length,
        # GSPO/DAPO/Dr.GRPO
        loss_type=effective_loss_type,
        epsilon=config.epsilon_low,
        epsilon_high=config.epsilon_high,
        importance_sampling_level=config.importance_sampling_level,
        scale_rewards=config.scale_rewards,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.run_name + "_fairgrpo",
        logging_dir=config.log_dir,
    )

    # --- Trainer ---
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[fair_reward_fn],
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # --- Train with fairness monitoring ---
    logger.info("Starting FairGRPO training...")
    trainer.train()

    # --- Log final fairness metrics ---
    fairness_summary = tracker.get_summary()
    logger.info("=" * 40)
    logger.info("FairGRPO Final Fairness Metrics:")
    for feature, groups in fairness_summary.items():
        if feature == "fairness_gaps":
            logger.info(f"  Fairness Gaps: {groups}")
        else:
            for group, stats in groups.items():
                logger.info(f"  {feature}/{group}: count={stats['count']}, mean_reward={stats['mean_reward']}")

    gaps = fairness_summary.get("fairness_gaps", {})
    for feature, gap in gaps.items():
        if gap > config.max_fairness_gap:
            logger.warning(
                f"  ⚠ Fairness gap for '{feature}' ({gap:.4f}) exceeds "
                f"target ({config.max_fairness_gap}). Consider additional training."
            )
        else:
            logger.info(f"  ✓ Fairness gap for '{feature}' ({gap:.4f}) within target.")

    # --- Save ---
    trainer.save_model(os.path.join(config.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final"))

    # Save fairness report
    fairness_path = os.path.join(config.output_dir, "fairness_report.json")
    with open(fairness_path, "w") as f:
        json.dump(fairness_summary, f, indent=2)
    logger.info(f"Fairness report saved to {fairness_path}")

    logger.info("✅ FairGRPO training complete!")
    return trainer


def _load_patient_demographics(config: BioAgentGRPOConfig) -> dict:
    """Load patient demographic data from domain db.json for fairness tracking.

    Maps task_id -> patient demographic dict with age, sex, ethnicity, etc.
    """
    demo_map = {}
    domain_dir = Path(f"data/domains/{config.domain}")
    db_path = domain_dir / "db.json"

    if not db_path.exists():
        logger.debug(f"No db.json found at {db_path}, fairness tracking will use empty demographics")
        return demo_map

    try:
        with open(db_path, "r", encoding="utf-8") as f:
            db = json.load(f)

        # Extract patients from various db structures
        patients = {}
        if "patients" in db:
            patients = db["patients"]
        elif "patient_profiles" in db:
            patients = db["patient_profiles"]
        elif "records" in db:
            # EHR format: extract demographics from records
            for rid, record in db["records"].items():
                if "demographics" in record:
                    pid = record["demographics"].get("patient_id", rid)
                    patients[pid] = record["demographics"]

        # Build task_id -> demographics mapping
        # Load tasks to get patient_id per task
        tasks_path = Path(config.tasks_path)
        if tasks_path.exists():
            with open(tasks_path, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            for task in tasks:
                tid = task.get("id", "")
                pid = task.get("patient_id", "")
                hadm_id = task.get("hadm_id", "")

                # Try to find patient demographics
                if pid and pid in patients:
                    demo_map[tid] = patients[pid]
                elif hadm_id and hadm_id in (db.get("records", {})):
                    record = db["records"][hadm_id]
                    if "demographics" in record:
                        demo_map[tid] = record["demographics"]

        logger.info(f"Loaded demographics for {len(demo_map)}/{len(patients)} patient-task pairs")

    except Exception as e:
        logger.warning(f"Failed to load patient demographics: {e}")

    return demo_map


# ============================================================
# Multi-turn GRPO (environment-in-the-loop)
# ============================================================


@dataclass
class MultiTurnGRPOConfig(BioAgentGRPOConfig):
    """Config for multi-turn GRPO with environment-in-the-loop."""
    # Multi-turn rollout
    num_rollouts_per_task: int = 4     # G: rollouts per prompt (for group relative)
    max_turns: int = 10                # Max environment interaction turns
    rollout_temperature: float = 0.8   # Higher for diverse rollouts
    # Trajectory filtering
    min_trajectory_reward: float = 0.0  # Discard trajectories below this
    max_trajectory_length: int = 4096   # Max tokens per trajectory
    # Training from trajectories
    trajectory_epochs: int = 2          # Epochs over collected trajectories
    grpo_mini_batch_size: int = 4       # Mini-batch for GRPO update
    # Task sampling — cap tasks per training cycle to prevent overfitting
    max_train_tasks: int = 20           # 0 = use all tasks (original behavior)
    # Logging
    save_trajectories: bool = True      # Save all trajectories to disk


def train_multiturn(config: BioAgentGRPOConfig):
    """Run multi-turn GRPO training with environment-in-the-loop.

    This is the CORE training loop of Healthcare AI GYM. It performs
    actual agent-environment interaction to collect multi-turn trajectories,
    then trains using the Group Relative Policy Optimization (GRPO) approach.

    Algorithm:
    For each training epoch:
        1. Sample a batch of tasks from the training set
        2. For each task, run G rollouts through the GYM environment:
           - Agent generates tool calls → environment executes → returns observation
           - Repeat for max_turns or until agent submits final answer
        3. Score each trajectory using the 5D composite reward:
           accuracy + format + process + safety + coherence
        4. Within each task's G trajectories, compute group-relative advantages:
           advantage_i = reward_i - mean(rewards_in_group)
        5. Build (prompt, trajectory) pairs weighted by advantages
        6. Update the policy model via GRPO objective

    Reference: AgentGym-RL (arXiv:2509.08755), GRPO (arXiv:2402.03300)
    """
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mt_config = MultiTurnGRPOConfig(**vars(config))

    logger.info("=" * 60)
    logger.info("BIOAgents Multi-Turn GRPO Trainer (Environment-in-the-Loop)")
    logger.info("=" * 60)
    logger.info(f"Model: {mt_config.model_name_or_path}")
    logger.info(f"Domain: {mt_config.domain}")
    logger.info(f"Max turns: {mt_config.max_turns}")
    logger.info(f"Rollouts per task (G): {mt_config.num_rollouts_per_task}")

    # --- Setup tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        mt_config.model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Setup model ---
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map.get(mt_config.torch_dtype, torch.bfloat16)

    # Detect model type for correct Auto class
    from transformers import AutoConfig
    _model_config = AutoConfig.from_pretrained(mt_config.model_name_or_path, trust_remote_code=True)
    _model_type = getattr(_model_config, "model_type", "")
    _is_qwen_vl = _model_type in ("qwen2_5_vl", "qwen2_vl") or "qwen2" in _model_type.lower() and "vl" in _model_type.lower()

    if _is_qwen_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration
        logger.info(f"Loading VL model ({_model_type}) with Qwen2_5_VLForConditionalGeneration")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mt_config.model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            mt_config.model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.gradient_checkpointing_enable()  # Save VRAM

    # --- PEFT ---
    if mt_config.peft_enabled:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            r=mt_config.peft_r,
            lora_alpha=mt_config.peft_lora_alpha,
            lora_dropout=mt_config.peft_lora_dropout,
            target_modules=mt_config.peft_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        logger.info(f"LoRA applied: r={mt_config.peft_r}, alpha={mt_config.peft_lora_alpha}")
        model.print_trainable_parameters()

    # --- Load environment ---
    from bioagents.gym.agent_env import register_bioagent_gym
    register_bioagent_gym()

    import gymnasium as gym

    # --- Load tasks ---
    tasks_path = Path(mt_config.tasks_path)
    with open(tasks_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    # Apply split filtering
    if mt_config.split_tasks_path and mt_config.train_split:
        split_file = Path(mt_config.split_tasks_path)
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                splits = json.load(f)
            if mt_config.train_split in splits:
                valid_ids = set(splits[mt_config.train_split])
                all_tasks = [t for t in all_tasks if t["id"] in valid_ids]

    # Cap training tasks to prevent overfitting on small domains
    if mt_config.max_train_tasks > 0 and len(all_tasks) > mt_config.max_train_tasks:
        import random as _rng
        original_count = len(all_tasks)
        all_tasks = _rng.sample(all_tasks, mt_config.max_train_tasks)
        logger.info(f"Capped training tasks to {mt_config.max_train_tasks} (from {original_count})")

    logger.info(f"Training on {len(all_tasks)} tasks, {mt_config.num_rollouts_per_task} rollouts each")

    # --- Optimizer + LR Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=mt_config.learning_rate,
        weight_decay=mt_config.weight_decay,
    )
    # Estimate total training steps for cosine scheduler
    _est_tasks = min(len(all_tasks), mt_config.max_train_tasks) if mt_config.max_train_tasks > 0 else len(all_tasks)
    _est_steps_per_epoch = max(_est_tasks * mt_config.num_rollouts_per_task // mt_config.grpo_mini_batch_size, 1)
    _est_total_steps = _est_steps_per_epoch * mt_config.num_train_epochs * mt_config.trajectory_epochs
    _warmup_steps = max(int(_est_total_steps * 0.1), 1)
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
    _warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=_warmup_steps)
    _cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(_est_total_steps - _warmup_steps, 1))
    lr_scheduler = SequentialLR(optimizer, schedulers=[_warmup_scheduler, _cosine_scheduler], milestones=[_warmup_steps])
    logger.info(f"LR scheduler: warmup {_warmup_steps} steps → cosine decay over {_est_total_steps} total steps")

    # --- Reward function (with adaptive weights from config) ---
    from bioagents.evaluation.rewards import compute_composite_reward

    # Check if using adaptive reward strategy
    _use_strategy = mt_config.reward_strategy and mt_config.reward_strategy != "grpo"
    _reward_strategy = None
    if _use_strategy:
        from bioagents.evaluation.reward_strategies import create_reward_strategy
        _reward_strategy = create_reward_strategy(mt_config.reward_strategy)
        logger.info(f"Using reward strategy: {_reward_strategy.name}")

    # Convert reward_functions list to weights dict for compute_composite_reward
    _reward_weights = {}
    for rw_spec in mt_config.reward_functions:
        _reward_weights[rw_spec["name"]] = rw_spec["weight"]
    if not _reward_weights:
        _reward_weights = {"accuracy": 0.4, "format": 0.2, "process": 0.4}
    logger.info(f"Reward weights: {_reward_weights}")

    # --- Output directory ---
    os.makedirs(mt_config.output_dir, exist_ok=True)

    # --- W&B Logging ---
    from bioagents.utils.wandb_logger import GymWandbLogger
    wb = GymWandbLogger.init_run(
        agent_id=mt_config.run_name or "multiturn_grpo",
        run_type="multiturn_grpo",
        domain=mt_config.domain,
        reward_strategy=mt_config.reward_strategy,
        model_name=mt_config.model_name_or_path,
        config={
            "model": mt_config.model_name_or_path,
            "domain": mt_config.domain,
            "reward_strategy": mt_config.reward_strategy,
            "max_turns": mt_config.max_turns,
            "num_rollouts_per_task": mt_config.num_rollouts_per_task,
            "num_train_epochs": mt_config.num_train_epochs,
            "learning_rate": mt_config.learning_rate,
            "beta": mt_config.beta,
            "temperature": mt_config.rollout_temperature,
            "peft_r": mt_config.peft_r,
            "reward_weights": _reward_weights,
        },
        tags=["multiturn", f"domain:{mt_config.domain}", f"strategy:{mt_config.reward_strategy}"],
        enabled=mt_config.use_wandb,
    )

    # ========================================
    # Main Training Loop
    # ========================================
    all_trajectories = []
    best_mean_reward = -1.0
    global_step = 0

    for epoch in range(mt_config.num_train_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{mt_config.num_train_epochs}")
        logger.info(f"{'='*50}")

        epoch_rewards = []
        epoch_trajectories = []

        # Shuffle tasks each epoch
        random.shuffle(all_tasks)

        for task_idx, task in enumerate(all_tasks):
            task_id = task["id"]
            ticket = task.get("ticket", "")
            correct_answer = task.get("correct_answer", "")
            eval_criteria = task.get("evaluation_criteria", {})
            expected_actions = eval_criteria.get("actions", [])

            # --- Collect G rollouts for this task ---
            task_rollouts = []

            for g in range(mt_config.num_rollouts_per_task):
                trajectory = _run_single_rollout(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    domain=mt_config.domain,
                    task=task,
                    max_turns=mt_config.max_turns,
                    temperature=mt_config.rollout_temperature,
                )
                task_rollouts.append(trajectory)

            # --- Compute rewards for each rollout ---
            task_rewards = []
            for traj in task_rollouts:
                if _use_strategy and _reward_strategy is not None:
                    # Use adaptive reward strategy (MRPO, SARL, etc.)
                    completions = [[{"content": traj["final_response"], "role": "assistant"}]]
                    strategy_rewards = _reward_strategy.compute_reward(
                        completions, solution=[correct_answer], task=task,
                    )
                    strategy_breakdowns = _reward_strategy.get_reward_breakdown(
                        completions, solution=[correct_answer], task=task,
                    )
                    traj["reward"] = strategy_rewards[0]
                    traj["reward_detail"] = strategy_breakdowns[0] if strategy_breakdowns else {"total": strategy_rewards[0]}
                    task_rewards.append(strategy_rewards[0])
                else:
                    # Standard composite reward
                    reward_result = compute_composite_reward(
                        response=traj["final_response"],
                        correct_answer=correct_answer,
                        tool_call_log=traj["tool_calls"],
                        expected_actions=expected_actions,
                        is_final=True,
                        weights=_reward_weights,
                    )
                    traj["reward"] = reward_result["total"]
                    traj["reward_detail"] = reward_result
                    task_rewards.append(reward_result["total"])

            # --- Compute GRPO group-relative advantages ---
            mean_reward = sum(task_rewards) / max(len(task_rewards), 1)
            std_reward = (sum((r - mean_reward)**2 for r in task_rewards) / max(len(task_rewards), 1)) ** 0.5
            std_reward = max(std_reward, 1e-6)  # prevent division by zero

            for traj, reward in zip(task_rollouts, task_rewards):
                traj["advantage"] = (reward - mean_reward) / std_reward
                epoch_trajectories.append(traj)

            epoch_rewards.extend(task_rewards)

            if (task_idx + 1) % 5 == 0:
                logger.info(
                    f"  Task {task_idx+1}/{len(all_tasks)}: "
                    f"mean_R={mean_reward:.3f}, "
                    f"best_R={max(task_rewards):.3f}, "
                    f"worst_R={min(task_rewards):.3f}"
                )

            # W&B per-task logging
            global_step += 1
            wb.log_step({
                "task/mean_reward": mean_reward,
                "task/best_reward": max(task_rewards),
                "task/worst_reward": min(task_rewards),
                "task/std_reward": std_reward,
                "task/num_turns_avg": sum(t.get("num_turns", 0) for t in task_rollouts) / max(len(task_rollouts), 1),
            }, step=global_step)

        # --- Mix in eval trajectories (WebRL-style replay, arXiv:2411.02337) ---
        # High-reward trajectories from the previous evaluation phase are replayed
        # into the training batch to bootstrap learning from successful strategies.
        if mt_config.eval_trajectories:
            valid_eval = [
                t for t in mt_config.eval_trajectories
                if t.get("reward", t.get("final_reward", 0.0)) >= mt_config.eval_trajectory_min_reward
            ]
            mix_n = int(len(epoch_trajectories) * mt_config.eval_trajectory_mix_ratio)
            if valid_eval and mix_n > 0:
                sampled = random.sample(valid_eval, min(mix_n, len(valid_eval)))
                # Set replay advantages to match the positive advantage
                # distribution of the current GRPO batch. Fixed advantage=1.0
                # creates a scale mismatch with normalized GRPO advantages.
                rollout_advantages = [
                    t.get("advantage", 0.0) for t in epoch_trajectories
                    if t.get("advantage", 0.0) > 0
                ]
                replay_advantage = (
                    sum(rollout_advantages) / len(rollout_advantages)
                    if rollout_advantages else 1.0
                )
                for t in sampled:
                    t["advantage"] = replay_advantage
                    if "reward" not in t and "final_reward" in t:
                        t["reward"] = t["final_reward"]
                epoch_trajectories.extend(sampled)
                logger.info(
                    f"  [Replay] Mixed {len(sampled)} eval trajectories "
                    f"(pool={len(valid_eval)}, ratio={mt_config.eval_trajectory_mix_ratio:.0%}, "
                    f"replay_adv={replay_advantage:.3f})"
                )
                wb.log_step({"replay/eval_traj_count": len(sampled),
                             "replay/eval_pool_size": len(valid_eval),
                             "replay/advantage": replay_advantage}, step=global_step)

        # --- Filter trajectories ---
        positive_trajs = [t for t in epoch_trajectories if t.get("advantage", 0) > 0]
        logger.info(
            f"Epoch {epoch+1}: {len(epoch_trajectories)} total trajectories, "
            f"{len(positive_trajs)} positive advantage"
        )

        # --- Compute reference log-probs (KL anchor) ---
        # Use the base model (LoRA adapters disabled) as the reference.
        # This prevents catastrophic forgetting by penalizing policy drift.
        ref_log_probs = None
        if mt_config.peft_enabled and hasattr(model, "disable_adapter_layers"):
            logger.info("  Computing reference log-probs (LoRA disabled)...")
            model.disable_adapter_layers()
            ref_log_probs = _compute_ref_log_probs(
                ref_model=model,
                tokenizer=tokenizer,
                trajectories=epoch_trajectories,
                device=device,
                max_length=mt_config.max_trajectory_length,
                use_dr_grpo=mt_config.use_dr_grpo,
            )
            model.enable_adapter_layers()
            logger.info(f"  Reference log-probs computed for {len(ref_log_probs)} trajectories")

        # --- GRPO Policy Update ---
        if positive_trajs:
            loss_total = _grpo_policy_update(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                trajectories=epoch_trajectories,
                device=device,
                beta=mt_config.beta,
                max_length=mt_config.max_trajectory_length,
                mini_batch_size=mt_config.grpo_mini_batch_size,
                num_epochs=mt_config.trajectory_epochs,
                use_dr_grpo=mt_config.use_dr_grpo,
                ref_log_probs=ref_log_probs,
                lr_scheduler=lr_scheduler,
            )
            logger.info(f"  GRPO loss: {loss_total:.4f}")
        else:
            logger.warning("  No positive-advantage trajectories, skipping update")

        # --- Epoch statistics ---
        mean_epoch_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        logger.info(f"Epoch {epoch+1} mean reward: {mean_epoch_reward:.4f}")

        # W&B epoch logging
        wb.log_epoch(
            epoch=epoch + 1,
            mean_reward=mean_epoch_reward,
            num_trajectories=len(epoch_trajectories),
            positive_trajectories=len(positive_trajs),
            loss=loss_total if positive_trajs else 0.0,
        )

        if mean_epoch_reward > best_mean_reward:
            best_mean_reward = mean_epoch_reward
            # Save best checkpoint
            best_path = os.path.join(mt_config.output_dir, "best")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            logger.info(f"  New best model saved (reward={best_mean_reward:.4f})")
            wb.set_summary("best_mean_reward", best_mean_reward)

        # --- Save trajectories ---
        if mt_config.save_trajectories:
            traj_path = os.path.join(mt_config.output_dir, f"trajectories_epoch_{epoch+1}.json")
            _save_trajectories(epoch_trajectories, traj_path)

        all_trajectories.extend(epoch_trajectories)

    # --- Save final model ---
    final_path = os.path.join(mt_config.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # --- Save training summary ---
    summary = {
        "total_trajectories": len(all_trajectories),
        "best_mean_reward": best_mean_reward,
        "epochs": mt_config.num_train_epochs,
        "tasks": len(all_tasks),
        "rollouts_per_task": mt_config.num_rollouts_per_task,
        "domain": mt_config.domain,
    }
    summary_path = os.path.join(mt_config.output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Multi-turn GRPO training complete!")
    logger.info(f"Best mean reward: {best_mean_reward:.4f}")
    logger.info(f"Total trajectories: {len(all_trajectories)}")
    logger.info(f"Models saved to: {mt_config.output_dir}")
    logger.info(f"{'='*50}")

    # W&B final summary
    wb.set_summary("total_trajectories", len(all_trajectories))
    wb.set_summary("best_mean_reward", best_mean_reward)
    wb.set_summary("domain", mt_config.domain)
    wb.set_summary("reward_strategy", mt_config.reward_strategy)
    wb.finish()

    return all_trajectories


def _run_single_rollout(
    model,
    tokenizer,
    device,
    domain: str,
    task: dict,
    max_turns: int = 10,
    temperature: float = 0.8,
) -> dict:
    """Run a single multi-turn rollout through the GYM environment.

    Returns a trajectory dict with:
    - messages: full conversation history
    - tool_calls: list of tool call records
    - final_response: agent's final text response
    - num_turns: number of turns taken
    """
    import gymnasium as gym

    task_id = task["id"]

    # Create environment (BUG-009: use constant, not string literal)
    from bioagents.utils.model_loader import GYM_ENV_ID
    env = gym.make(
        GYM_ENV_ID,
        domain=domain,
        task_id=task_id,
    )

    obs, info = env.reset()
    # obs is a string (the initial observation text), info is a dict with metadata
    system_prompt = "You are a medical AI assistant. Use available tools to complete the task."
    if isinstance(obs, dict):
        # Legacy format
        observation_text = obs.get("ticket", str(obs))
        system_prompt = obs.get("system_prompt", system_prompt)
    else:
        # Current format: obs is a string containing the full initial observation
        observation_text = str(obs)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation_text},
    ]

    tool_calls = []
    final_response = ""

    for turn in range(max_turns):
        # Generate model response
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=max(temperature, 0.01),
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not response:
            response = '{"name": "think", "arguments": {"thought": "Let me analyze this carefully."}}'

        # Try to parse as tool call
        parsed_tool = _parse_tool_call_from_response(response)

        if parsed_tool:
            # Execute tool in environment
            action = json.dumps(parsed_tool)
            obs_new, reward, terminated, truncated, step_info = env.step(action)

            # Normalize observation
            if isinstance(obs_new, dict):
                tool_response_str = obs_new.get("tool_response", json.dumps(obs_new, ensure_ascii=False))
            elif isinstance(obs_new, str):
                tool_response_str = obs_new
            else:
                tool_response_str = str(obs_new) if obs_new is not None else ""

            tool_calls.append({
                "tool_name": parsed_tool.get("name", ""),
                "arguments": parsed_tool.get("arguments", {}),
                "response": tool_response_str,
            })

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": tool_response_str})

            if terminated or truncated:
                final_response = response
                break
        else:
            # Non-tool response = final answer
            final_response = response
            messages.append({"role": "assistant", "content": response})
            break

    env.close()

    return {
        "task_id": task_id,
        "messages": messages,
        "tool_calls": tool_calls,
        "final_response": final_response,
        "num_turns": len(tool_calls) + 1,
        "full_text": "\n".join(m["content"] for m in messages if m["role"] == "assistant"),
    }


def _parse_tool_call_from_response(text: str) -> Optional[dict]:
    """Parse a tool call from model output text.

    Delegates to the canonical ``parse_tool_call`` in agent_runner which
    supports JSON, code-blocks, XML tags, ReAct format, and alternative
    key names (function/tool/action).
    """
    try:
        from bioagents.evaluation.agent_runner import parse_tool_call
        return parse_tool_call(text)
    except ImportError:
        pass

    # ── inline fallback (should never be reached) ──
    import re
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "name" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    code_match = re.search(r'```(?:json)?\s*\n?({.*?})\s*\n?```', text, re.DOTALL)
    if code_match:
        try:
            parsed = json.loads(code_match.group(1))
            if "name" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def _compute_ref_log_probs(
    ref_model,
    tokenizer,
    trajectories: list[dict],
    device,
    max_length: int = 4096,
    use_dr_grpo: bool = False,
) -> dict[int, float]:
    """Pre-compute log-probabilities under the frozen reference model.

    Returns a dict mapping trajectory index → reference log-prob.
    This is used for the KL divergence penalty in the GRPO objective:
        KL(π_θ || π_ref) ≈ log π_θ(y|x) - log π_ref(y|x)
    """
    ref_log_probs = {}
    ref_model.eval()
    with torch.no_grad():
        for idx, traj in enumerate(trajectories):
            full_text = traj.get("full_text", "")
            if not full_text:
                continue
            encoding = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            input_ids = encoding["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue
            outputs = ref_model(input_ids=input_ids, labels=input_ids)
            if use_dr_grpo:
                seq_len = max((input_ids != tokenizer.pad_token_id).sum().item(), 1)
                ref_log_probs[idx] = (-outputs.loss * seq_len).item()
            else:
                ref_log_probs[idx] = (-outputs.loss).item()
    return ref_log_probs


def _grpo_policy_update(
    model,
    tokenizer,
    optimizer,
    trajectories: list[dict],
    device,
    beta: float = 0.04,
    max_length: int = 4096,
    mini_batch_size: int = 4,
    num_epochs: int = 2,
    use_dr_grpo: bool = False,
    ref_log_probs: dict[int, float] | None = None,
    lr_scheduler=None,
) -> float:
    """GRPO policy gradient update from collected trajectories.

    Implements the core GRPO objective with proper KL penalty:
        L = -E[advantage * log π_θ(y|x)] + β * E[log π_θ(y|x) - log π_ref(y|x)]

    The KL term anchors the policy to the reference (pretrained) model,
    preventing catastrophic forgetting during RL fine-tuning.

    Dr. GRPO variant (use_dr_grpo=True): uses sum log-prob instead of mean,
    eliminating the length normalization bias in the original GRPO objective.
    Reference: Dr. GRPO (2025) — removes per-token averaging that penalizes
    appropriately detailed medical answers.

    where advantages are group-relative (normalized within each task's rollouts).

    For trajectories with positive advantage, we increase log probability.
    For those with negative advantage, we decrease it.
    """
    import torch.nn.functional as F

    model.train()
    total_loss = 0.0
    total_kl = 0.0
    num_updates = 0

    for epoch in range(num_epochs):
        # Shuffle trajectories
        import random
        indices = list(range(len(trajectories)))
        random.shuffle(indices)

        for batch_start in range(0, len(indices), mini_batch_size):
            batch_indices = indices[batch_start:batch_start + mini_batch_size]
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            batch_kl = 0.0
            valid_samples = 0

            for idx in batch_indices:
                traj = trajectories[idx]
                advantage = traj.get("advantage", 0.0)

                if abs(advantage) < 1e-6:
                    continue

                # Build the full trajectory text
                full_text = traj.get("full_text", "")
                if not full_text:
                    continue

                # Tokenize
                encoding = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )
                input_ids = encoding["input_ids"].to(device)

                if input_ids.shape[1] < 2:
                    continue

                # Forward pass — compute log probabilities
                outputs = model(input_ids=input_ids, labels=input_ids)
                if use_dr_grpo:
                    # Dr. GRPO: use sum (not mean) of log-probs to remove
                    # length normalization bias. outputs.loss is mean NLL,
                    # so multiply by sequence length to get sum.
                    seq_len = max((input_ids != tokenizer.pad_token_id).sum().item(), 1)
                    log_probs = -outputs.loss * seq_len
                else:
                    log_probs = -outputs.loss  # standard: mean NLL → log prob

                # GRPO objective: advantage-weighted log probability
                # Positive advantage → increase log prob (good trajectory)
                # Negative advantage → decrease log prob (bad trajectory)
                policy_loss = -advantage * log_probs

                # KL divergence penalty: KL(π_θ || π_ref) ≈ log π_θ - log π_ref
                # This prevents the policy from drifting too far from the
                # pretrained model, avoiding catastrophic forgetting.
                kl_penalty = torch.tensor(0.0, device=device)
                if ref_log_probs is not None and idx in ref_log_probs:
                    ref_lp = ref_log_probs[idx]
                    kl_penalty = log_probs - ref_lp  # approx KL per sample
                    batch_kl += kl_penalty.item()

                # Combined loss: policy gradient + β * KL penalty
                sample_loss = policy_loss + beta * kl_penalty

                batch_loss = batch_loss + sample_loss
                valid_samples += 1

            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                total_loss += batch_loss.item()
                total_kl += batch_kl / valid_samples
                num_updates += 1

    model.eval()
    avg_loss = total_loss / max(num_updates, 1)
    avg_kl = total_kl / max(num_updates, 1)
    if num_updates > 0:
        logger.info(f"  GRPO update: {num_updates} steps, avg_loss={avg_loss:.4f}, avg_kl={avg_kl:.4f}")
    return avg_loss


def _save_trajectories(trajectories: list[dict], path: str):
    """Save trajectories to disk (JSON serializable subset)."""
    serializable = []
    for traj in trajectories:
        serializable.append({
            "task_id": traj.get("task_id"),
            "num_turns": traj.get("num_turns"),
            "reward": traj.get("reward"),
            "advantage": traj.get("advantage"),
            "reward_detail": traj.get("reward_detail"),
            "tool_calls": traj.get("tool_calls"),
            "final_response": traj.get("final_response", "")[:500],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(serializable)} trajectories to {path}")


# ============================================================
# CLI
# ============================================================


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="BIOAgents GRPO Trainer")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to GRPO config YAML file",
    )
    parser.add_argument(
        "--mode", type=str, default="single_turn",
        choices=["single_turn", "multi_turn", "fair_grpo"],
        help="Training mode: single_turn (TRL GRPO), multi_turn (env-in-the-loop), or fair_grpo (FairGRPO)",
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        choices=["grpo", "mrpo", "sarl", "adaptive"],
        help="Reward strategy override: grpo (standard), mrpo (token shaping), sarl (search agent RL), adaptive (auto-select per task)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Build datasets and reward functions without training",
    )
    args = parser.parse_args()

    # Load config
    config = BioAgentGRPOConfig.from_yaml(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Override strategy from CLI if provided
    if args.strategy:
        config.reward_strategy = args.strategy
        logger.info(f"Reward strategy overridden to: {args.strategy}")

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info(f"Model: {config.model_name_or_path}")
        logger.info(f"Domain: {config.domain}")
        logger.info(f"Reward strategy: {config.reward_strategy}")

        # Build and validate dataset
        train_ds = build_grpo_dataset(config, split=config.train_split)
        logger.info(f"Train dataset: {len(train_ds)} examples")
        logger.info(f"Sample prompt:\n{json.dumps(train_ds[0]['prompt'], indent=2)}")

        # Validate reward functions
        reward_fns = build_reward_functions(config)
        logger.info(f"Reward functions: {len(reward_fns)}")

        # Test reward computation
        test_completions = [[{"content": "The answer is B", "role": "assistant"}]]
        for fn in reward_fns:
            scores = fn(test_completions, solution=["B"])
            fn_name = getattr(fn, "__name__", str(fn))
            logger.info(f"  {fn_name}: test_score={scores}")

        # If adaptive strategy, show selection analysis
        if config.reward_strategy == "adaptive":
            from bioagents.evaluation.reward_strategies import (
                create_reward_strategy,
                AdaptiveRewardStrategy,
            )
            strategy = create_reward_strategy("adaptive")
            if isinstance(strategy, AdaptiveRewardStrategy):
                # Analyze first few tasks
                tasks_path = Path(config.tasks_path)
                with open(tasks_path, "r", encoding="utf-8") as f:
                    sample_tasks = json.load(f)[:10]
                for task in sample_tasks:
                    strategy.select_strategy(task)
                summary = strategy.get_selection_summary()
                logger.info(f"  Adaptive strategy selection preview: {summary}")

        logger.info("✅ Dry run complete!")
        return

    if args.mode == "single_turn":
        train(config)
    elif args.mode == "multi_turn":
        train_multiturn(config)
    elif args.mode == "fair_grpo":
        fair_config = FairGRPOConfig(**vars(config))
        train_fair_grpo(fair_config)


if __name__ == "__main__":
    main()
