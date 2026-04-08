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

import threading as _threading

import torch
import yaml
from loguru import logger

# ── Monkey-patch Triton autotuner for thread safety ──
# Triton's in-memory cache is not thread-safe; concurrent autotuner runs
# corrupt cache entries (NoneType mapping error). This lock serializes
# autotuning — only affects the first call per kernel config.
try:
    import triton.runtime.autotuner as _triton_autotuner
    _triton_autotuner_lock = _threading.Lock()
    _triton_original_run = _triton_autotuner.Autotuner.run

    def _triton_thread_safe_run(self, *args, **kwargs):
        with _triton_autotuner_lock:
            return _triton_original_run(self, *args, **kwargs)

    _triton_autotuner.Autotuner.run = _triton_thread_safe_run
    logger.info("[Triton] Autotuner monkey-patched for thread safety")
except Exception as _e:
    logger.warning(f"[Triton] Could not patch autotuner: {_e}")

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

    # ── On-Policy Distillation (OPD) ──────────────────────────
    # Cross-stage OPD (GLM-5 style): teacher provides per-token dense
    # supervision on student-sampled trajectories.
    # Reference: GLM-5 (arXiv:2602.15763), G-OPD (arXiv:2602.12125)
    use_opd: bool = False                          # Enable OPD
    opd_teacher_path: str = ""                     # Path to teacher model (merged)
    opd_mode: str = "cross_stage"                  # "cross_stage" | "self_evolving" | "hybrid"
    opd_alpha: float = 0.5                         # Blend: α*RL_advantage + (1-α)*OPD_advantage
    opd_lambda: float = 1.0                        # G-OPD reward extrapolation (λ>1 = surpass teacher)
    opd_warmup_steps: int = 50                     # Steps before OPD kicks in (let RL stabilize first)

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
                "use_dr_grpo", "max_train_tasks",
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
        if "opd" in raw:
            o = raw["opd"]
            for key in [
                "use_opd", "opd_teacher_path", "opd_mode",
                "opd_alpha", "opd_lambda", "opd_warmup_steps",
            ]:
                if key in o:
                    kwargs[key] = o[key]

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
    ticket = task.get("ticket", task.get("question", ""))
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
    elif domain == "medical_qa_direct":
        system_msg = (
            "You are a medical expert. Answer the following medical question by "
            "selecting the best option. Reply with ONLY the letter (A, B, C, or D) "
            "wrapped in <answer> tags, e.g. <answer>A</answer>."
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
# Entropy Monitoring Callback (GTPO-style)
# ============================================================


from transformers import TrainerCallback


class EntropyMonitorCallback(TrainerCallback):
    """Monitors policy entropy during training to detect collapse early.

    Based on GTPO (arXiv:2508.03772): tracks generation entropy and
    applies adaptive entropy bonus when entropy drops below threshold.

    Entropy collapse is the #1 early warning sign for B041-style model
    destruction. If entropy drops below critical_threshold, training
    should be stopped immediately.
    """

    def __init__(
        self,
        warning_threshold: float = 0.15,   # Log warning when entropy drops below this
        critical_threshold: float = 0.05,  # STOP training when entropy drops below this
        min_steps_before_stop: int = 20,   # Don't auto-stop before this many steps
        window_size: int = 50,             # Smoothing window for entropy tracking
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.min_steps_before_stop = min_steps_before_stop
        self.window_size = window_size
        self.entropy_history = []
        self.warned = False

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """Track entropy metrics from training logs."""
        if logs is None:
            return

        # TRL logs various metrics; look for entropy-related ones
        entropy = logs.get("entropy", logs.get("policy_entropy", None))

        if entropy is not None:
            self.entropy_history.append(entropy)
            # Compute smoothed entropy
            window = self.entropy_history[-self.window_size:]
            avg_entropy = sum(window) / len(window)

            logs["entropy_monitor/avg_entropy"] = avg_entropy
            logs["entropy_monitor/min_entropy"] = min(window)

            if avg_entropy < self.critical_threshold and len(self.entropy_history) >= self.min_steps_before_stop:
                logger.error(
                    f"🚨 CRITICAL: Entropy collapsed to {avg_entropy:.4f} "
                    f"(threshold: {self.critical_threshold}) after {len(self.entropy_history)} steps. "
                    f"Training stopped to prevent irreversible damage!"
                )
                # Signal to stop training
                if hasattr(control, "should_training_stop"):
                    control.should_training_stop = True

            elif avg_entropy < self.warning_threshold and not self.warned:
                logger.warning(
                    f"⚠ Entropy dropping: {avg_entropy:.4f} "
                    f"(warning threshold: {self.warning_threshold}). "
                    f"Monitor closely for potential collapse."
                )
                self.warned = True


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


class DynamicSamplingCallback(TrainerCallback):
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
# Force-save callback (workaround for TRL resume save bug)
# ============================================================


class ForceSaveCallback(TrainerCallback):
    """Force checkpoint saving every N steps.

    Workaround for a bug where TRL GRPOTrainer doesn't save checkpoints
    after resuming from a checkpoint. This callback explicitly sets
    should_save=True on the correct step boundaries.
    """

    def __init__(self, save_steps: int = 50):
        self.save_steps = save_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_steps == 0:
            logger.info(f"ForceSaveCallback: triggering save at step {state.global_step}")
            control.should_save = True
        return control


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

    # --- Callbacks ---
    callbacks = []

    # Force-save callback (workaround for TRL resume save bug)
    callbacks.append(ForceSaveCallback(save_steps=config.save_steps))
    logger.info(f"ForceSaveCallback enabled (save every {config.save_steps} steps)")

    # Entropy monitoring (always enabled — critical for detecting collapse)
    callbacks.append(EntropyMonitorCallback())
    logger.info("Entropy monitoring enabled (GTPO-style collapse detection)")

    # --- Dynamic Sampling Callback (DAPO technique #2) ---
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

    # --- Train (auto-resume from latest checkpoint if available) ---
    import glob as _glob
    _ckpts = sorted(_glob.glob(os.path.join(config.output_dir, "checkpoint-*")),
                    key=lambda p: int(p.split("-")[-1]) if p.split("-")[-1].isdigit() else 0)
    _resume = _ckpts[-1] if _ckpts else None
    if _resume:
        logger.info(f"Resuming training from checkpoint: {_resume}")
    else:
        logger.info("Starting GRPO training from scratch...")
    trainer.train(resume_from_checkpoint=_resume)

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
    grpo_mini_batch_size: int = 2       # Mini-batch for GRPO update (reduced for 9B + 4k tokens)
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

    # --- Setup tokenizer (and processor for VL models) ---
    tokenizer = AutoTokenizer.from_pretrained(
        mt_config.model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load processor for VL models (needed for image inputs)
    _processor = None
    from transformers import AutoConfig as _AC
    _pre_config = _AC.from_pretrained(mt_config.model_name_or_path, trust_remote_code=True)
    _pre_model_type = getattr(_pre_config, "model_type", "")
    if _pre_model_type in ("qwen2_5_vl", "qwen2_vl", "qwen3_5") or (
        "qwen2" in _pre_model_type.lower() and "vl" in _pre_model_type.lower()
    ):
        from transformers import AutoProcessor
        _processor = AutoProcessor.from_pretrained(
            mt_config.model_name_or_path, trust_remote_code=True,
        )
        logger.info(f"Loaded VL processor for {_pre_model_type}")

    # --- Setup model ---
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map.get(mt_config.torch_dtype, torch.bfloat16)

    # Detect model type for correct Auto class
    from transformers import AutoConfig
    _model_config = AutoConfig.from_pretrained(mt_config.model_name_or_path, trust_remote_code=True)
    _model_type = getattr(_model_config, "model_type", "")
    _is_qwen_vl = (
        _model_type in ("qwen2_5_vl", "qwen2_vl", "qwen3_5")
        or ("qwen2" in _model_type.lower() and "vl" in _model_type.lower())
    )

    # Determine correct model class and attention implementation
    _is_qwen3_5 = _model_type == "qwen3_5"
    # Qwen3.5 requires eager attention (sdpa causes hangs)
    _attn_impl = "eager" if _is_qwen3_5 else "sdpa"

    if _is_qwen3_5:
        from transformers import Qwen3_5ForConditionalGeneration
        logger.info(f"Loading VL model ({_model_type}) with Qwen3_5ForConditionalGeneration (attn={_attn_impl})")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            mt_config.model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            attn_implementation=_attn_impl,
        )
    elif _is_qwen_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration
        logger.info(f"Loading VL model ({_model_type}) with Qwen2_5_VLForConditionalGeneration")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mt_config.model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            attn_implementation=_attn_impl,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            mt_config.model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            attn_implementation="sdpa",
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

    # --- OPD: Load teacher model ---
    teacher_model = None
    if mt_config.use_opd and mt_config.opd_teacher_path:
        logger.info(f"[OPD] Loading teacher model from: {mt_config.opd_teacher_path}")
        logger.info(f"[OPD] Mode: {mt_config.opd_mode}, alpha: {mt_config.opd_alpha}, lambda: {mt_config.opd_lambda}")
        if _is_qwen3_5:
            teacher_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                mt_config.opd_teacher_path,
                torch_dtype=model_dtype,
                trust_remote_code=True,
                attn_implementation=_attn_impl,
            )
        elif _is_qwen_vl:
            teacher_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                mt_config.opd_teacher_path,
                torch_dtype=model_dtype,
                trust_remote_code=True,
            )
        else:
            teacher_model = AutoModelForCausalLM.from_pretrained(
                mt_config.opd_teacher_path,
                torch_dtype=model_dtype,
                trust_remote_code=True,
            )
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        logger.info(f"[OPD] Teacher model loaded ({sum(p.numel() for p in teacher_model.parameters())/1e9:.1f}B params, frozen)")

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

    # --- Pre-load GYM domains in main thread (avoid race condition in workers) ---
    from bioagents.gym.agent_env import _load_default_domains, register_bioagent_gym
    _load_default_domains()
    register_bioagent_gym()

    # --- Rollout backend: vLLM server or HuggingFace model replicas ---
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    _num_rollout_gpus = len(_visible_gpus)
    _DOMAIN_ALIASES = {
        "text_qa": "medical_qa",
        "multimodal_vqa": "visual_diagnosis",
    }

    # ── vLLM Rollout Server Setup ──
    # Uses vLLM for inference (CUDA graphs, PagedAttention, continuous batching)
    # instead of raw HuggingFace model.generate() on each GPU.
    _use_vllm_rollout = True  # Enabled: flash_attn built from source for GLIBC 2.31
    _vllm_client = None
    _vllm_process = None
    _vllm_port = 8199
    _lora_base_dir = os.path.join(mt_config.output_dir, "_vllm_lora")
    _lora_version = [0]

    if _use_vllm_rollout:
        import subprocess
        import time as _setup_time
        try:
            import openai as _openai_mod
            import requests as _requests_mod
        except ImportError:
            logger.warning("[vLLM] openai/requests not available, falling back to HF replicas")
            _use_vllm_rollout = False

    if _use_vllm_rollout:
        logger.info(f"[vLLM] Setting up vLLM rollout server (replaces {_num_rollout_gpus-1} HF replicas)")

        # Save initial LoRA adapter for vLLM
        os.makedirs(_lora_base_dir, exist_ok=True)
        _lora_path_v0 = os.path.join(_lora_base_dir, "v0")
        model.save_pretrained(_lora_path_v0)
        logger.info(f"[vLLM] Saved initial LoRA adapter → {_lora_path_v0}")

        # Determine how many vLLM instances to launch (1 per GPU, GPU1..N-1)
        _vllm_num_instances = _num_rollout_gpus - 1  # Use all available GPUs for vLLM
        _vllm_processes = []
        _vllm_clients = []

        for _vi in range(_vllm_num_instances):
            _gpu_id = _vi + 1  # GPU1, GPU2, ...
            _port = _vllm_port + _vi
            # Use vLLM-converted model (interleaved qwen3_next format)
            _vllm_model_path = mt_config.model_name_or_path + "-vllm"
            if not os.path.exists(_vllm_model_path):
                _vllm_model_path = mt_config.model_name_or_path + "-text-only"
            if not os.path.exists(_vllm_model_path):
                _vllm_model_path = mt_config.model_name_or_path  # fallback
            _vllm_cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", _vllm_model_path,
                "--served-model-name", "current",
                "--tensor-parallel-size", "1",
                "--gpu-memory-utilization", "0.85",
                "--max-model-len", str(max(mt_config.max_prompt_length + mt_config.max_completion_length, 32768)),
                "--port", str(_port),
                "--trust-remote-code",
                "--dtype", "bfloat16",
                "--disable-log-requests",
                "--enable-prefix-caching",
                "--enforce-eager",
                "--enable-auto-tool-choice",
                "--tool-call-parser", "hermes",
                "--enable-lora",
                "--lora-modules", f"active={_lora_path_v0}",
                "--max-lora-rank", "64",
            ]
            _vllm_env = os.environ.copy()
            _vllm_env["CUDA_VISIBLE_DEVICES"] = str(_gpu_id)
            _vllm_env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
            _vllm_env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
            _proc = subprocess.Popen(
                _vllm_cmd, env=_vllm_env,
                stdout=open(os.path.join(mt_config.output_dir, f"vllm_gpu{_gpu_id}.log"), "w"),
                stderr=subprocess.STDOUT,
            )
            _vllm_processes.append(_proc)
            logger.info(f"[vLLM] Launching instance {_vi} on GPU{_gpu_id}, port {_port} (PID={_proc.pid})")

        # Wait for all servers to be ready
        for _vi in range(_vllm_num_instances):
            _port = _vllm_port + _vi
            _client = _openai_mod.OpenAI(
                base_url=f"http://localhost:{_port}/v1", api_key="dummy",
                timeout=300.0,
            )
            _ready = False
            for _attempt in range(300):  # up to 5 minutes per server
                try:
                    _client.models.list()
                    _ready = True
                    break
                except Exception:
                    _setup_time.sleep(1)

            if not _ready:
                logger.error(f"[vLLM] Instance {_vi} (port {_port}) failed to start!")
                # Kill all and fallback
                for p in _vllm_processes:
                    p.kill()
                _use_vllm_rollout = False
                break

            # LoRA pre-loaded via --lora-modules flag at server startup
            logger.info(f"[vLLM] Instance {_vi} ready, LoRA pre-loaded via --lora-modules")

            _vllm_clients.append(_client)

        if _use_vllm_rollout:
            logger.info(f"[vLLM] {len(_vllm_clients)} inference servers ready (continuous batching + CUDA graphs)")

    def _reload_vllm_lora(lora_path: str, lora_version: int) -> bool:
        """Hot-reload LoRA adapter on all vLLM inference servers.

        Uses POST /v1/load_lora_adapter (vLLM 0.15.1+) to dynamically
        update the LoRA weights without restarting the servers. Requires
        VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 (set in server env).

        Args:
            lora_path: Absolute path to the saved LoRA adapter directory.
            lora_version: Version number for logging.

        Returns:
            True if all servers reloaded successfully, False otherwise.
        """
        import requests as _req_mod

        _reload_success = True
        for _ri, _rc in enumerate(_vllm_clients):
            _port = _vllm_port + _ri
            _url = f"http://localhost:{_port}/v1/load_lora_adapter"
            _payload = {
                "lora_name": "active",
                "lora_path": lora_path,
                "load_inplace": True,
            }
            try:
                _resp = _req_mod.post(_url, json=_payload, timeout=60)
                if _resp.status_code == 200:
                    logger.info(
                        f"[vLLM] Instance {_ri} (port {_port}): "
                        f"LoRA v{lora_version} loaded successfully"
                    )
                else:
                    logger.warning(
                        f"[vLLM] Instance {_ri} (port {_port}): "
                        f"LoRA reload returned {_resp.status_code}: {_resp.text}"
                    )
                    _reload_success = False
            except Exception as _reload_err:
                logger.warning(
                    f"[vLLM] Instance {_ri} (port {_port}): "
                    f"LoRA reload failed: {_reload_err}"
                )
                _reload_success = False

        if _reload_success:
            logger.info(
                f"[vLLM] LoRA v{lora_version} hot-reloaded on all "
                f"{len(_vllm_clients)} servers"
            )
        else:
            logger.warning(
                f"[vLLM] LoRA v{lora_version} reload had failures on some "
                f"servers — rollouts may use stale weights"
            )
        return _reload_success

    # Fallback: HuggingFace model replicas (original path)
    _rollout_models = [model]
    _rollout_devices = [device]
    if not _use_vllm_rollout and _num_rollout_gpus > 1:
        logger.info(f"[Multi-GPU] Setting up {_num_rollout_gpus} HF replicas for parallel rollouts")
        for gpu_i in range(1, _num_rollout_gpus):
            _dev = torch.device(f"cuda:{gpu_i}")
            if _is_qwen3_5:
                _replica = Qwen3_5ForConditionalGeneration.from_pretrained(
                    mt_config.model_name_or_path, torch_dtype=model_dtype,
                    trust_remote_code=True, attn_implementation=_attn_impl,
                ).to(_dev)
            elif _is_qwen_vl:
                _replica = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    mt_config.model_name_or_path, torch_dtype=model_dtype,
                    trust_remote_code=True, attn_implementation="sdpa",
                ).to(_dev)
            else:
                _replica = AutoModelForCausalLM.from_pretrained(
                    mt_config.model_name_or_path, torch_dtype=model_dtype,
                    trust_remote_code=True, attn_implementation="sdpa",
                ).to(_dev)
            _replica.load_state_dict(model.state_dict(), strict=False)
            _replica.eval()
            _rollout_models.append(_replica)
            logger.info(f"[Multi-GPU] Replica {gpu_i} loaded on cuda:{gpu_i}")
        _rollout_devices = [torch.device(f"cuda:{i}") for i in range(_num_rollout_gpus)]

    def _process_single_task(task, gpu_idx):
        """Process all rollouts for a single task."""
        task_domain = task.get("_source_domain", mt_config.domain)
        task_domain = _DOMAIN_ALIASES.get(task_domain, task_domain)
        task_id = task.get("id", "?")

        rollouts = []
        for g in range(mt_config.num_rollouts_per_task):
            logger.info(f"[GPU{gpu_idx}] Task {task_id} rollout {g+1}/{mt_config.num_rollouts_per_task} starting (domain={task_domain})")
            if _use_vllm_rollout:
                # Route to one of the vLLM instances (round-robin by gpu_idx)
                _vllm_idx = gpu_idx % len(_vllm_clients)
                trajectory = _run_single_rollout(
                    model=None,
                    tokenizer=tokenizer,
                    device=None,
                    domain=task_domain,
                    task=task,
                    max_turns=mt_config.max_turns,
                    temperature=mt_config.rollout_temperature,
                    max_completion_length=mt_config.max_completion_length,
                    max_prompt_length=mt_config.max_prompt_length,
                    rollout_timeout=6000.0,
                    vllm_client=_vllm_clients[_vllm_idx],
                    processor=_processor,
                )
            else:
                _model = _rollout_models[gpu_idx]
                _dev = _rollout_devices[gpu_idx]
                trajectory = _run_single_rollout(
                    model=_model,
                    tokenizer=tokenizer,
                    device=_dev,
                    domain=task_domain,
                    task=task,
                    max_turns=mt_config.max_turns,
                    temperature=mt_config.rollout_temperature,
                    max_completion_length=mt_config.max_completion_length,
                    max_prompt_length=mt_config.max_prompt_length,
                    rollout_timeout=6000.0,
                    processor=_processor,
                )
            rollouts.append(trajectory)
            logger.info(f"[GPU{gpu_idx}] Task {task_id} rollout {g+1}/{mt_config.num_rollouts_per_task} done — resp_len={len(trajectory.get('final_response',''))}")
        return rollouts

    # ── Triton warm-up: compile kernels single-threaded to avoid cache races ──
    if _use_vllm_rollout:
        # vLLM handles its own kernel compilation; only warm up training model on GPU0
        logger.info("[Triton] vLLM mode: warming up training model only (GPU0)...")
        _wu_texts = ["Hello", "Hello world " * 50, "Hello world " * 400]
        for _wu_text in _wu_texts:
            _wu_ids = tokenizer(_wu_text, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)
            _wu_out = model(_wu_ids, labels=_wu_ids)
            _wu_out.loss.backward()
            optimizer.zero_grad()
        logger.info("[Triton] Training forward+backward warm-up done")
    else:
        # HF replica mode: warm up all GPUs
        logger.info("[Triton] Warming up kernels on each GPU (single-threaded)...")
        _wu_texts = ["Hello", "Hello world " * 50, "Hello world " * 400]
        for _wu_i, (_wu_model, _wu_dev) in enumerate(zip(_rollout_models, _rollout_devices)):
            with torch.no_grad():
                for _wu_text in _wu_texts:
                    _wu_ids = tokenizer(_wu_text, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(_wu_dev)
                    _wu_model(_wu_ids)
                _wu_short = tokenizer("Hello", return_tensors="pt").input_ids.to(_wu_dev)
                _wu_model.generate(_wu_short, max_new_tokens=4, do_sample=True, temperature=0.9)
                _wu_model.generate(_wu_short, max_new_tokens=4, do_sample=False)
            logger.info(f"[Triton] GPU{_wu_i} warm-up done")
        for _wu_text in _wu_texts:
            _wu_ids = tokenizer(_wu_text, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)
            _wu_out = model(_wu_ids, labels=_wu_ids)
            _wu_out.loss.backward()
            optimizer.zero_grad()
        logger.info("[Triton] Training forward+backward warm-up done")

    for epoch in range(mt_config.num_train_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{mt_config.num_train_epochs}")
        logger.info(f"{'='*50}")

        # Sync LoRA weights to rollout backend at start of each epoch
        if _use_vllm_rollout:
            _lora_version[0] += 1
            _lora_path_new = os.path.join(_lora_base_dir, f"v{_lora_version[0]}")
            model.save_pretrained(_lora_path_new)
            _reload_vllm_lora(_lora_path_new, _lora_version[0])
        elif _num_rollout_gpus > 1:
            _train_state = model.state_dict()
            for gpu_i in range(1, _num_rollout_gpus):
                _rollout_models[gpu_i].load_state_dict(_train_state, strict=False)
            logger.info(f"[Multi-GPU] Synced weights to {_num_rollout_gpus} replicas")

        epoch_rewards = []
        epoch_trajectories = []

        # Shuffle tasks each epoch
        random.shuffle(all_tasks)

        # Work-stealing GPU scheduler: each GPU pulls tasks from a shared
        # queue instead of waiting for synchronized batches. This eliminates
        # GPU idle time when tasks have different durations.
        import queue as _queue_mod
        import threading as _threading_mod

        _task_queue = _queue_mod.Queue()
        for _ti, _t in enumerate(all_tasks):
            _task_queue.put((_ti, _t))

        _completed_tasks = []  # list of (task_idx, task, rollouts)
        _completed_lock = _threading_mod.Lock()
        _gradient_event = _threading_mod.Event()
        _tasks_since_grad = [0]  # mutable counter
        _grad_every_n = _num_rollout_gpus  # gradient step every 8 tasks
        def _gpu_worker(gpu_idx):
            """Each GPU continuously pulls tasks from the shared queue."""
            while True:
                try:
                    task_idx, task = _task_queue.get_nowait()
                except _queue_mod.Empty:
                    break
                rollouts = _process_single_task(task, gpu_idx)
                with _completed_lock:
                    _completed_tasks.append((task_idx, task, rollouts))
                    _tasks_since_grad[0] += 1
                    if _tasks_since_grad[0] >= _grad_every_n:
                        _gradient_event.set()
                _task_queue.task_done()

        # Launch persistent worker threads
        # With vLLM: can run more workers since vLLM handles batching internally
        _num_workers = _num_rollout_gpus * 2 if _use_vllm_rollout else _num_rollout_gpus
        _gpu_threads = []
        for _gi in range(_num_workers):
            _gt = _threading_mod.Thread(target=_gpu_worker, args=(_gi,), daemon=True)
            _gt.start()
            _gpu_threads.append(_gt)
        if _use_vllm_rollout:
            logger.info(f"[vLLM] Launched {_num_workers} worker threads → {len(_vllm_clients)} vLLM servers")

        # Main thread: process completed tasks and fire gradient steps
        _processed_count = 0
        while _processed_count < len(all_tasks):
            # Wait for enough tasks to complete (or all threads done)
            _gradient_event.wait(timeout=30.0)
            _gradient_event.clear()

            # Grab completed tasks
            with _completed_lock:
                _new_completed = _completed_tasks[_processed_count:]
                _tasks_since_grad[0] = 0

            if not _new_completed:
                # Check if all threads are still alive
                if all(not t.is_alive() for t in _gpu_threads):
                    break
                continue

            # Process each completed task: compute rewards & advantages
            batch_results = {}
            for task_idx, task, task_rollouts in _new_completed:
                batch_results[task_idx] = (task, task_rollouts)

            for task_idx in sorted(batch_results.keys()):
                task, task_rollouts = batch_results[task_idx]
                correct_answer = task.get("correct_answer", "")
                eval_criteria = task.get("evaluation_criteria", {})
                expected_actions = eval_criteria.get("actions", [])

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

            if (task_idx + 1) % 1 == 0 or task_idx == len(all_tasks) - 1:
                _gpu_tag = f" [batch={_num_rollout_gpus}x]" if _num_rollout_gpus > 1 else ""
                logger.info(
                    f"  Task {task_idx+1}/{len(all_tasks)}{_gpu_tag}: "
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

            # ── Async GRPO: gradient update after each batch of completed tasks ──
            # Work-stealing scheduler fires gradient step every _grad_every_n tasks.
            # GPU threads continue rolling out while gradient step runs on GPU0.
            _n_new = len(_new_completed)
            batch_trajectories = epoch_trajectories[-_n_new * mt_config.num_rollouts_per_task:]
            batch_positive = [t for t in batch_trajectories if t.get("advantage", 0) > 0]
            _grad_batch_num = (_processed_count + _n_new) // _grad_every_n

            if batch_positive:
                # Compute reference log-probs for this batch
                _batch_ref_log_probs = None
                if mt_config.peft_enabled and hasattr(model, "disable_adapter_layers"):
                    model.disable_adapter_layers()
                    _batch_ref_log_probs = _compute_ref_log_probs(
                        ref_model=model,
                        tokenizer=tokenizer,
                        trajectories=batch_trajectories,
                        device=device,
                        max_length=mt_config.max_trajectory_length,
                        use_dr_grpo=mt_config.use_dr_grpo,
                        processor=_processor,
                    )
                    model.enable_adapter_layers()

                # GRPO gradient update on this batch
                batch_loss = _grpo_policy_update(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    trajectories=batch_trajectories,
                    device=device,
                    beta=mt_config.beta,
                    max_length=mt_config.max_trajectory_length,
                    mini_batch_size=mt_config.grpo_mini_batch_size,
                    num_epochs=mt_config.trajectory_epochs,
                    use_dr_grpo=mt_config.use_dr_grpo,
                    ref_log_probs=_batch_ref_log_probs,
                    lr_scheduler=lr_scheduler,
                    processor=_processor,
                )
                logger.info(
                    f"  [Async] Batch {_grad_batch_num}: "
                    f"loss={batch_loss:.4f}, trajs={len(batch_trajectories)}, "
                    f"positive={len(batch_positive)}, mean_R={mean_reward:.3f}"
                )
                wb.log_step({
                    "batch/loss": batch_loss,
                    "batch/num_trajectories": len(batch_trajectories),
                    "batch/positive_trajs": len(batch_positive),
                }, step=global_step)

                # Periodic checkpoint saving (every 25 gradient batches)
                _save_every_n = getattr(mt_config, 'save_steps', 25)
                if _grad_batch_num > 0 and _grad_batch_num % _save_every_n == 0 and _grad_batch_num != getattr(train_multiturn, '_last_saved_batch', -1):
                    _ckpt_path = os.path.join(mt_config.output_dir, f"checkpoint-{_grad_batch_num}")
                    os.makedirs(_ckpt_path, exist_ok=True)
                    model.save_pretrained(_ckpt_path)
                    tokenizer.save_pretrained(_ckpt_path)
                    # Save optimizer state for resume
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "grad_batch_num": _grad_batch_num,
                        "global_step": global_step,
                        "epoch": epoch,
                        "processed_count": _processed_count,
                    }, os.path.join(_ckpt_path, "training_state.pt"))
                    train_multiturn._last_saved_batch = _grad_batch_num
                    logger.info(f"  [Checkpoint] Saved checkpoint-{_grad_batch_num}")
                    # Cleanup old checkpoints (keep latest save_total_limit)
                    _save_limit = getattr(mt_config, 'save_total_limit', 15)
                    import glob as _glob_mod
                    _all_ckpts = sorted(_glob_mod.glob(os.path.join(mt_config.output_dir, "checkpoint-*")),
                                        key=lambda p: int(p.split("-")[-1]))
                    if len(_all_ckpts) > _save_limit:
                        for _old in _all_ckpts[:-_save_limit]:
                            import shutil
                            shutil.rmtree(_old, ignore_errors=True)

                # KL divergence early warning
                if abs(batch_kl_avg := batch_loss) > 2.0:
                    logger.warning(f"  [KL Warning] |avg_loss|={abs(batch_loss):.4f} > 2.0 — policy may be diverging")

                # Sync updated weights to rollout backend
                if _use_vllm_rollout:
                    _lora_version[0] += 1
                    _lora_path_new = os.path.join(_lora_base_dir, f"v{_lora_version[0]}")
                    model.save_pretrained(_lora_path_new)
                    _reload_vllm_lora(_lora_path_new, _lora_version[0])
                    logger.info(f"  [Async] LoRA v{_lora_version[0]} synced to vLLM servers")
                elif _num_rollout_gpus > 1:
                    _train_state = model.state_dict()
                    for gpu_i in range(1, _num_rollout_gpus):
                        _rollout_models[gpu_i].load_state_dict(_train_state, strict=False)
                    logger.info(f"  [Async] Synced weights to {_num_rollout_gpus} replicas")
            else:
                logger.info(f"  [Async] Batch {_grad_batch_num}: no positive-advantage trajs, skipping update")

            _processed_count += len(_new_completed)

        # Wait for all GPU threads to finish
        for _gt in _gpu_threads:
            _gt.join(timeout=60)

        # --- Epoch statistics ---
        mean_epoch_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        positive_trajs = [t for t in epoch_trajectories if t.get("advantage", 0) > 0]
        logger.info(
            f"Epoch {epoch+1}: {len(epoch_trajectories)} total trajectories, "
            f"{len(positive_trajs)} positive advantage, "
            f"mean reward: {mean_epoch_reward:.4f}"
        )

        # W&B epoch logging
        wb.log_epoch(
            epoch=epoch + 1,
            mean_reward=mean_epoch_reward,
            num_trajectories=len(epoch_trajectories),
            positive_trajectories=len(positive_trajs),
            loss=0.0,
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

    # --- Cleanup vLLM servers ---
    if _use_vllm_rollout and _vllm_processes:
        logger.info(f"[vLLM] Shutting down {len(_vllm_processes)} inference servers...")
        for _proc in _vllm_processes:
            try:
                _proc.terminate()
                _proc.wait(timeout=10)
            except Exception:
                _proc.kill()
        logger.info("[vLLM] All servers stopped")

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
    max_completion_length: int = 1024,
    max_prompt_length: int = 4096,
    rollout_timeout: float = 600.0,
    vllm_client=None,
    processor=None,
) -> dict:
    """Run a single multi-turn rollout through the GYM environment.

    Returns a trajectory dict with:
    - messages: full conversation history
    - tool_calls: list of tool call records
    - final_response: agent's final text response
    - num_turns: number of turns taken
    - image_path: path to image file if this is a VQA task (for training)
    """
    import gymnasium as gym
    import time as _time_mod

    task_id = task["id"]
    _rollout_t0 = _time_mod.time()

    # Create environment (BUG-009: use constant, not string literal)
    from bioagents.utils.model_loader import GYM_ENV_ID
    env = gym.make(
        GYM_ENV_ID,
        domain=domain,
    )

    # Pass task data directly to avoid task_id lookup failures
    # in multi-domain combined datasets where task IDs may not
    # exist in the target domain's own task registry
    obs, info = env.reset(options={"task_data": task})
    # obs is a string (the initial observation text), info is a dict with metadata
    system_prompt = "You are a medical AI assistant. Use available tools to complete the task."
    if isinstance(obs, dict):
        # Legacy format
        observation_text = obs.get("ticket", str(obs))
        system_prompt = obs.get("system_prompt", system_prompt)
    else:
        # Current format: obs is a string containing the full initial observation
        observation_text = str(obs)

    # Build multimodal user content if image is available and processor is loaded
    _image_path = info.get("image_path") or task.get("_image_path")
    _has_vision = processor is not None and _image_path and os.path.isfile(_image_path)

    if _has_vision:
        # Multimodal message: image + text (matches eval format)
        user_content = [
            {"type": "image", "image": f"file://{_image_path}"},
            {"type": "text", "text": observation_text},
        ]
    else:
        user_content = observation_text

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # --- Extract tool definitions from GYM env for Qwen3.5 tool calling ---
    _env_tools_raw = info.get("tools", [])
    _tool_defs = []
    for td in _env_tools_raw:
        if hasattr(td, "model_dump"):
            _tool_defs.append(td.model_dump())
        elif isinstance(td, dict):
            _tool_defs.append(td)
    # Ensure OpenAI-compatible format: {"type": "function", "function": {...}}
    _openai_tools = []
    for td in _tool_defs:
        if "type" in td and "function" in td:
            _openai_tools.append(td)
        elif "name" in td:
            _openai_tools.append({
                "type": "function",
                "function": {
                    "name": td["name"],
                    "description": td.get("description", ""),
                    "parameters": td.get("parameters", {"type": "object", "properties": {}}),
                },
            })

    tool_calls = []
    final_response = ""

    for turn in range(max_turns):
        _turn_t0 = _time_mod.time()

        # Check rollout timeout
        _rollout_elapsed = _time_mod.time() - _rollout_t0
        if _rollout_elapsed > rollout_timeout:
            logger.warning(f"[{task_id}] Rollout timeout ({rollout_timeout:.0f}s) at turn {turn+1}")
            break

        if vllm_client is not None:
            # ── vLLM path: OpenAI-compatible API with continuous batching ──
            try:
                # Dynamic max_tokens: cap to remaining context to avoid 400 errors
                _vllm_max_tokens = max_completion_length
                # Estimate input tokens from all message content (content + tool_calls + name)
                _est_input = 0
                for _m in messages:
                    _mc = _m.get("content") or ""
                    _est_input += len(str(_mc)) // 3
                    if _m.get("tool_calls"):
                        _est_input += len(str(_m["tool_calls"])) // 3
                _est_input += len(messages) * 4  # per-message overhead tokens
                _remaining = 32768 - _est_input  # use 32k context
                if _remaining < _vllm_max_tokens:
                    _vllm_max_tokens = max(_remaining, 128)
                _completion = vllm_client.chat.completions.create(
                    model="current",  # LoRA adapter name
                    messages=messages,
                    tools=_openai_tools if _openai_tools else None,
                    temperature=max(temperature, 0.01),
                    top_p=0.95,
                    max_tokens=_vllm_max_tokens,
                )
                _choice = _completion.choices[0]
                response = _choice.message.content or ""
                _input_len = _completion.usage.prompt_tokens if _completion.usage else 0
                _gen_len = _completion.usage.completion_tokens if _completion.usage else len(response.split())

                # Handle vLLM-parsed tool calls (Qwen3.5 native tool calling)
                parsed_tool = None
                if _choice.message.tool_calls:
                    tc = _choice.message.tool_calls[0]
                    try:
                        _args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                    except (json.JSONDecodeError, TypeError):
                        _args = {"raw": tc.function.arguments}
                    parsed_tool = {"name": tc.function.name, "arguments": _args}
                    # Reconstruct response text for trajectory logging
                    response = json.dumps(parsed_tool, ensure_ascii=False)
                elif response:
                    # Try parsing text as tool call (fallback)
                    parsed_tool = _parse_tool_call_from_response(response)
            except Exception as _vllm_err:
                logger.warning(f"[{task_id}] vLLM generation error at turn {turn+1}: {_vllm_err}")
                response = '{"name": "think", "arguments": {"thought": "Let me analyze this carefully."}}'
                parsed_tool = _parse_tool_call_from_response(response)
                _input_len = 0
                _gen_len = 0
        else:
            # ── HuggingFace path: direct model.generate() ──
            _template_fn = processor if (_has_vision and processor is not None) else tokenizer
            try:
                prompt_text = _template_fn.apply_chat_template(
                    messages, tools=_openai_tools if _openai_tools else None,
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                prompt_text = _template_fn.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            if _has_vision and processor is not None:
                # Vision path: use processor to handle image + text together
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[prompt_text],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
            _input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_completion_length,
                    temperature=max(temperature, 0.01),
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            _gen_len = len(new_tokens)
            parsed_tool = _parse_tool_call_from_response(response) if response else None

        _turn_elapsed = _time_mod.time() - _turn_t0

        if not response:
            response = '{"name": "think", "arguments": {"thought": "Let me analyze this carefully."}}'
            parsed_tool = _parse_tool_call_from_response(response)

        _tool_name = parsed_tool.get("name", "?") if parsed_tool else "FINAL"
        logger.info(f"[{task_id}] turn {turn+1}/{max_turns}: in={_input_len} gen={_gen_len} t={_turn_elapsed:.1f}s tool={_tool_name}")

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

    # If loop exhausted max_turns without setting final_response, use last assistant msg
    if not final_response:
        for msg in reversed(messages):
            if msg["role"] == "assistant" and msg["content"]:
                final_response = msg["content"]
                break

    # Build full_text from assistant messages only (for text-only training path)
    _assistant_texts = []
    for m in messages:
        if m["role"] == "assistant":
            c = m["content"]
            _assistant_texts.append(c if isinstance(c, str) else str(c))

    return {
        "task_id": task_id,
        "messages": messages,
        "tool_calls": tool_calls,
        "final_response": final_response,
        "num_turns": len(tool_calls) + 1,
        "full_text": "\n".join(_assistant_texts),
        "image_path": _image_path if _has_vision else None,
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
    # Qwen3.5 XML: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    qwen35_match = re.search(
        r'<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>',
        text, re.DOTALL,
    )
    if qwen35_match:
        func_name = qwen35_match.group(1).strip()
        params_block = qwen35_match.group(2).strip()
        param_pairs = re.findall(
            r'<parameter=([^>]+)>(.*?)</parameter>', params_block, re.DOTALL
        )
        arguments = {}
        for key, val in param_pairs:
            val = val.strip()
            try:
                arguments[key.strip()] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                arguments[key.strip()] = val
        if func_name:
            return {"name": func_name, "arguments": arguments}
    return None


def _tokenize_trajectory(
    traj: dict,
    tokenizer,
    processor,
    max_length: int,
    device,
) -> Optional[dict]:
    """Tokenize a trajectory, handling vision inputs when available.

    For VQA trajectories with images, uses the processor to create inputs
    with pixel_values. For text-only trajectories, uses the tokenizer.

    Returns a dict with input_ids (and optionally pixel_values, image_grid_thw)
    on the target device, or None if the trajectory is empty.
    """
    full_text = traj.get("full_text", "")
    if not full_text:
        return None

    image_path = traj.get("image_path")
    if processor is not None and image_path and os.path.isfile(image_path):
        # Vision path: rebuild messages and process with images
        try:
            from qwen_vl_utils import process_vision_info
            messages = traj.get("messages", [])
            # Re-apply chat template with processor
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=False,
            )
            # Truncate if needed
            if inputs["input_ids"].shape[1] > max_length:
                inputs["input_ids"] = inputs["input_ids"][:, :max_length]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
            return {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            # Fall back to text-only on any error
            pass

    # Text-only path
    encoding = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return {k: v.to(device) for k, v in encoding.items()}


def _compute_ref_log_probs(
    ref_model,
    tokenizer,
    trajectories: list[dict],
    device,
    max_length: int = 4096,
    use_dr_grpo: bool = False,
    processor=None,
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

            # Vision-aware tokenization
            _img_path = traj.get("image_path")
            inputs = _tokenize_trajectory(
                traj, tokenizer, processor, max_length, device,
            )
            if inputs is None:
                continue
            input_ids = inputs["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            # Pass vision inputs if available
            _fwd_kwargs = {"input_ids": input_ids, "labels": input_ids}
            if "pixel_values" in inputs:
                _fwd_kwargs["pixel_values"] = inputs["pixel_values"]
            if "image_grid_thw" in inputs:
                _fwd_kwargs["image_grid_thw"] = inputs["image_grid_thw"]

            outputs = ref_model(**_fwd_kwargs)
            if use_dr_grpo:
                seq_len = max((input_ids != tokenizer.pad_token_id).sum().item(), 1)
                ref_log_probs[idx] = (-outputs.loss * seq_len).item()
            else:
                ref_log_probs[idx] = (-outputs.loss).item()
    return ref_log_probs


def _get_per_token_log_probs(
    model, input_ids: torch.Tensor, pixel_values=None, image_grid_thw=None,
) -> torch.Tensor:
    """Extract per-token log-probabilities from model logits.

    Given input_ids [1, seq_len], compute log p(y_t | y_{<t}, x) for each
    token position t. Uses the standard autoregressive left-shift:
      logits[:, t-1, :] predicts token at position t.

    Returns:
        Tensor of shape [seq_len - 1] with per-token log-probs.
    """
    import torch.nn.functional as F

    _fwd_kwargs = {"input_ids": input_ids}
    if pixel_values is not None:
        _fwd_kwargs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        _fwd_kwargs["image_grid_thw"] = image_grid_thw

    outputs = model(**_fwd_kwargs)
    logits = outputs.logits                        # [1, seq_len, vocab_size]
    shift_logits = logits[:, :-1, :]               # [1, seq_len-1, vocab_size]
    shift_labels = input_ids[:, 1:]                # [1, seq_len-1]

    # Use F.cross_entropy(reduction='none') instead of log_softmax + gather.
    # This avoids materializing a [seq_len, vocab_size] log-prob tensor,
    # saving significant GPU memory for large vocabularies (150K+).
    token_log_probs = -F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    )                                              # [seq_len-1]
    return token_log_probs                         # [seq_len-1]


def _compute_per_token_log_probs_batch(
    model,
    tokenizer,
    trajectories: list[dict],
    device,
    max_length: int = 4096,
    processor=None,
) -> dict[int, torch.Tensor]:
    """Pre-compute per-token log-probs for all trajectories under a model.

    Returns dict mapping trajectory index → CPU tensor [seq_len_i - 1]
    of per-token log-probs. Stored on CPU to save GPU memory.

    Used for both teacher and reference models in per-token OPD.
    """
    token_lps = {}
    model.eval()
    with torch.no_grad():
        for idx, traj in enumerate(trajectories):
            full_text = traj.get("full_text", "")
            if not full_text:
                continue
            inputs = _tokenize_trajectory(traj, tokenizer, processor, max_length, device)
            if inputs is None:
                continue
            input_ids = inputs["input_ids"]
            if input_ids.shape[1] < 2:
                continue
            _pv = inputs.get("pixel_values")
            _igt = inputs.get("image_grid_thw")
            per_token = _get_per_token_log_probs(model, input_ids, _pv, _igt)
            token_lps[idx] = per_token.cpu()  # store on CPU to save VRAM
    return token_lps


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
    teacher_token_log_probs: dict[int, torch.Tensor] | None = None,
    ref_token_log_probs: dict[int, torch.Tensor] | None = None,
    opd_alpha: float = 0.5,
    opd_lambda: float = 1.0,
    processor=None,
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

                # Skip trajectories with zero RL advantage — unless OPD is active,
                # where teacher signal can still provide useful gradients
                if abs(advantage) < 1e-6 and not (
                    teacher_token_log_probs is not None and idx in teacher_token_log_probs
                ):
                    continue

                # Build the full trajectory text
                full_text = traj.get("full_text", "")
                if not full_text:
                    continue

                # Tokenize (vision-aware)
                inputs = _tokenize_trajectory(traj, tokenizer, processor, max_length, device)
                if inputs is None:
                    continue
                input_ids = inputs["input_ids"]
                _pv = inputs.get("pixel_values")
                _igt = inputs.get("image_grid_thw")

                if input_ids.shape[1] < 2:
                    continue

                # ── Per-token OPD path ──
                # When OPD is active, compute per-token log-probs from logits
                # and apply per-token teacher advantage (GLM-5 style).
                # When OPD is inactive, use sequence-level outputs.loss (original path).
                has_opd = (
                    teacher_token_log_probs is not None
                    and ref_token_log_probs is not None
                    and idx in teacher_token_log_probs
                    and idx in ref_token_log_probs
                )

                if has_opd:
                    # Per-token student log-probs (with gradient)
                    student_token_lps = _get_per_token_log_probs(model, input_ids, _pv, _igt)
                    # [seq_len - 1], gradient flows through this

                    # Pre-computed teacher & ref per-token log-probs (no gradient)
                    teacher_t = teacher_token_log_probs[idx].to(device)
                    ref_t = ref_token_log_probs[idx].to(device)

                    # Align lengths (should match, but safety check)
                    min_len = min(len(student_token_lps), len(teacher_t), len(ref_t))
                    student_token_lps = student_token_lps[:min_len]
                    teacher_t = teacher_t[:min_len]
                    ref_t = ref_t[:min_len]

                    # Per-token OPD advantage: Â_opd(t) = λ * sg[log π_teacher(t) - log π_ref(t)]
                    opd_adv_per_token = opd_lambda * (teacher_t - ref_t)
                    # Clip per-token OPD advantage to prevent extreme values
                    opd_adv_per_token = opd_adv_per_token.clamp(-2.0, 2.0)

                    # Blend: α * RL_advantage (scalar, broadcast) + (1-α) * OPD_advantage (per-token)
                    effective_adv_per_token = opd_alpha * advantage + (1 - opd_alpha) * opd_adv_per_token

                    # Per-token policy loss
                    token_losses = -effective_adv_per_token * student_token_lps

                    # Aggregate: Dr.GRPO uses sum, standard uses mean
                    if use_dr_grpo:
                        policy_loss = token_losses.sum()
                    else:
                        policy_loss = token_losses.mean()

                    # Sequence-level log-prob for KL penalty (reuse per-token sum/mean)
                    if use_dr_grpo:
                        log_probs = student_token_lps.sum()
                    else:
                        log_probs = student_token_lps.mean()

                else:
                    # ── Original sequence-level path (no OPD) ──
                    _fwd_kwargs = {"input_ids": input_ids, "labels": input_ids}
                    if _pv is not None:
                        _fwd_kwargs["pixel_values"] = _pv
                    if _igt is not None:
                        _fwd_kwargs["image_grid_thw"] = _igt
                    outputs = model(**_fwd_kwargs)
                    if use_dr_grpo:
                        seq_len = max((input_ids != tokenizer.pad_token_id).sum().item(), 1)
                        log_probs = -outputs.loss * seq_len
                    else:
                        log_probs = -outputs.loss

                    policy_loss = -advantage * log_probs

                # KL divergence penalty: KL(π_θ || π_ref) ≈ log π_θ - log π_ref
                # Clipped to prevent explosion from off-policy divergence
                # (vLLM rollouts use base model, policy has LoRA drift).
                kl_penalty = torch.tensor(0.0, device=device)
                if ref_log_probs is not None and idx in ref_log_probs:
                    ref_lp = ref_log_probs[idx]
                    log_ratio = log_probs - ref_lp
                    # Skip sample if policy has drifted too far (importance ratio too extreme)
                    if abs(log_ratio.detach().item()) > 5.0:
                        continue
                    # Clip KL penalty for numerical stability
                    kl_penalty = log_ratio.clamp(-5.0, 5.0)
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
