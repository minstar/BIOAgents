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

    # PEFT / LoRA
    peft_enabled: bool = True
    peft_r: int = 16
    peft_lora_alpha: int = 32
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
    learning_rate: float = 5e-6
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

    # GRPO-specific
    num_generations: int = 4
    beta: float = 0.04
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

    # Environment
    max_turns: int = 10
    use_gym_env: bool = True

    # Logging
    wandb_project: str = "bioagents-grpo"
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
            ]:
                if key in t:
                    kwargs[key] = t[key]
        if "rewards" in raw:
            kwargs["reward_functions"] = raw["rewards"].get("functions", [])
            kwargs["bertscore_model"] = raw["rewards"].get("bertscore_model", cls.bertscore_model)
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

    Returns:
        List of callables matching TRL GRPOTrainer reward_funcs signature:
            fn(completions, **kwargs) -> list[float]
    """
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

    # --- GRPO Training Config (TRL 0.28+) ---
    os.makedirs(config.output_dir, exist_ok=True)

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
        # Logging
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.run_name,
        logging_dir=config.log_dir,
    )

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
# Multi-turn GRPO (environment-in-the-loop)
# ============================================================


def train_multiturn(config: BioAgentGRPOConfig):
    """Run multi-turn GRPO training with environment interaction.

    This variant performs rollouts through the BIOAgents GYM environment,
    collecting multi-turn trajectories and computing rewards based on
    the full interaction sequence.

    Uses the online GRPO approach:
    1. Generate G completions for each prompt
    2. For each completion, interact with the environment (multi-turn)
    3. Compute rewards from the full trajectory
    4. Update policy using GRPO objective
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("=" * 60)
    logger.info("BIOAgents Multi-Turn GRPO Trainer")
    logger.info("=" * 60)
    logger.info(f"Domain: {config.domain}")
    logger.info(f"Max turns per episode: {config.max_turns}")

    # --- Setup ---
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load environment
    from bioagents.gym.agent_env import BioAgentGymEnv, register_bioagent_gym

    register_bioagent_gym()

    # Load tasks
    tasks_path = Path(config.tasks_path)
    with open(tasks_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    logger.info(f"Loaded {len(all_tasks)} tasks for multi-turn training")

    # Build per-task prompt dataset
    records = []
    for task in all_tasks:
        records.append({
            "task_id": task["id"],
            "prompt": _build_prompt_from_task(task, config.domain),
            "solution": task.get("correct_answer", ""),
            "ticket": task.get("ticket", ""),
        })

    logger.info(f"Multi-turn dataset: {len(records)} episodes")
    logger.info("NOTE: Full multi-turn GRPO requires custom rollout loop.")
    logger.info("      For single-turn GRPO, use `train()` instead.")
    logger.info("      See AgentGym-RL/verl for multi-turn PPO reference.")

    # Save the prepared dataset for external training frameworks
    output_path = Path(config.output_dir) / "multiturn_prompts.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info(f"Multi-turn prompts saved to {output_path}")

    return records


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
        choices=["single_turn", "multi_turn"],
        help="Training mode: single_turn (TRL GRPO) or multi_turn (env-in-the-loop)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Build datasets and reward functions without training",
    )
    args = parser.parse_args()

    # Load config
    config = BioAgentGRPOConfig.from_yaml(args.config)
    logger.info(f"Loaded config from {args.config}")

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info(f"Model: {config.model_name_or_path}")
        logger.info(f"Domain: {config.domain}")

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
            logger.info(f"  {fn.__name__}: test_score={scores}")

        logger.info("✅ Dry run complete!")
        return

    if args.mode == "single_turn":
        train(config)
    elif args.mode == "multi_turn":
        train_multiturn(config)


if __name__ == "__main__":
    main()
main()
