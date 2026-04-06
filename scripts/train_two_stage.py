# -*- coding: utf-8 -*-
"""Two-Stage Training Pipeline: SFT Warmup -> GSPO/DAPO RL.

Implements the training approach proven by MedAgentGym (ICLR 2026)
and WebRL (ICLR 2025): warm up with supervised fine-tuning on
successful demonstrations first, then apply RL for further improvement.

Cold-start RL (directly applying GRPO/GSPO to a base model) risks
catastrophic collapse -- the B041 issue.

Usage:
    python scripts/train_two_stage.py \
        --sft-config configs/sft_medical_qa.yaml \
        --rl-config configs/gspo_dapo_lingshu7b.yaml \
        [--skip-sft]              # Skip SFT if warmup already done
        [--sft-checkpoint PATH]   # Use existing SFT checkpoint for RL stage
        [--rif-rft]               # Enable RIF-RFT sample filtering
        [--rif-rft-k 8]           # Number of rollouts for RIF-RFT filtering

Reference:
    - MedAgentGym (ICLR 2026): SFT -> DPO two-stage
    - WebRL (ICLR 2025): SFT warmup -> Adaptive RL
    - RIF-RFT: Rollout-based Instance Filtering
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioagents.training.grpo_trainer import BioAgentGRPOConfig, train, train_multiturn
from bioagents.training.sft_trainer import BioAgentSFTConfig


def run_sft_warmup(sft_config: BioAgentSFTConfig) -> str:
    """Run SFT warmup stage, merge LoRA, and return path to merged model."""
    from bioagents.training.sft_trainer import train as sft_train

    logger.info("=" * 60)
    logger.info("STAGE 1: SFT Warmup")
    logger.info("=" * 60)
    logger.info(f"Model: {sft_config.model_name_or_path}")
    logger.info(f"Output: {sft_config.output_dir}")

    sft_train(sft_config)

    # Find the LoRA adapter checkpoint
    final_path = os.path.join(sft_config.output_dir, "final")
    if not os.path.exists(final_path):
        checkpoints = sorted(Path(sft_config.output_dir).glob("checkpoint-*"))
        if checkpoints:
            final_path = str(checkpoints[-1])
        else:
            final_path = sft_config.output_dir

    # Merge LoRA adapter into base model for RL stage
    merged_path = os.path.join(sft_config.output_dir, "merged")
    if not os.path.exists(merged_path):
        merged_path = _merge_lora_checkpoint(
            base_model_path=sft_config.model_name_or_path,
            adapter_path=final_path,
            output_path=merged_path,
            torch_dtype=sft_config.torch_dtype,
        )

    logger.info(f"SFT warmup complete. Merged model: {merged_path}")
    return merged_path


def _merge_lora_checkpoint(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16",
) -> str:
    """Merge LoRA adapter into base model and save full model.

    This is necessary because TRL's GRPOTrainer needs a full model
    with proper config.json (including model_type), not just a LoRA adapter.
    """
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    logger.info("Merging LoRA adapter into base model...")
    logger.info(f"  Base: {base_model_path}")
    logger.info(f"  Adapter: {adapter_path}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    model_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # Check if it's a VL model
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    is_vl = model_type in ("qwen2_5_vl", "qwen2_vl")

    if is_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path, torch_dtype=model_dtype, trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=model_dtype, trust_remote_code=True,
        )

    # Load and merge LoRA
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    logger.info(f"  Merged model saved to: {output_path}")

    # Free memory
    del model, base_model
    torch.cuda.empty_cache()

    return output_path


def run_rif_rft_filtering(
    rl_config: BioAgentGRPOConfig,
    k: int = 8,
) -> str:
    """Run RIF-RFT: Rollout-based Instance Filtering.

    For each training sample, generate K rollouts and check if any
    produces a correct answer. Filter out "incompetent" samples where
    the model cannot produce any correct response.

    Returns path to filtered tasks file.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("=" * 60)
    logger.info(f"RIF-RFT Filtering (K={k} rollouts per sample)")
    logger.info("=" * 60)

    # Load tasks
    tasks_path = Path(rl_config.tasks_path)
    with open(tasks_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    logger.info(f"Total tasks before filtering: {len(all_tasks)}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        rl_config.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    model_dtype = dtype_map.get(rl_config.torch_dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        rl_config.model_name_or_path,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    # Generate K rollouts per task and check correctness
    competent_tasks = []
    incompetent_count = 0

    for i, task in enumerate(all_tasks):
        question = task.get("question", task.get("prompt", ""))
        answer = task.get("answer", task.get("solution", ""))

        if not question or not answer:
            competent_tasks.append(task)
            continue

        # Generate K responses
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        has_correct = False
        with torch.no_grad():
            for _ in range(k):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=rl_config.temperature,
                    top_p=rl_config.top_p,
                    do_sample=True,
                )
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                # Simple correctness check: does response contain the answer?
                answer_lower = answer.lower().strip()
                response_lower = response.lower().strip()
                if answer_lower in response_lower or response_lower.startswith(answer_lower):
                    has_correct = True
                    break

        if has_correct:
            competent_tasks.append(task)
        else:
            incompetent_count += 1

        if (i + 1) % 50 == 0:
            logger.info(
                f"RIF-RFT progress: {i+1}/{len(all_tasks)}, "
                f"kept={len(competent_tasks)}, filtered={incompetent_count}"
            )

    # Save filtered tasks
    filtered_path = tasks_path.parent / f"tasks_rif_filtered_k{k}.json"
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(competent_tasks, f, indent=2, ensure_ascii=False)

    filter_ratio = incompetent_count / len(all_tasks) * 100
    logger.info(f"RIF-RFT complete: {len(competent_tasks)}/{len(all_tasks)} kept ({filter_ratio:.1f}% filtered)")
    logger.info(f"Filtered tasks saved to: {filtered_path}")

    del model
    torch.cuda.empty_cache()

    return str(filtered_path)


def main():
    parser = argparse.ArgumentParser(description="Two-stage SFT -> RL training pipeline")
    parser.add_argument("--sft-config", type=str, help="Path to SFT config YAML")
    parser.add_argument("--rl-config", type=str, required=True, help="Path to GSPO/DAPO RL config YAML")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT warmup stage")
    parser.add_argument("--sft-checkpoint", type=str, help="Use existing SFT checkpoint for RL")
    parser.add_argument("--rif-rft", action="store_true", help="Enable RIF-RFT sample filtering")
    parser.add_argument("--rif-rft-k", type=int, default=8, help="Rollouts per sample for RIF-RFT")
    parser.add_argument("--gpu", type=str, default=None, help="GPU device(s) to use")
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # -- Stage 1: SFT Warmup --
    sft_checkpoint = args.sft_checkpoint

    if not args.skip_sft and not sft_checkpoint:
        if not args.sft_config:
            logger.error("--sft-config required unless --skip-sft or --sft-checkpoint is set")
            sys.exit(1)

        sft_config = BioAgentSFTConfig.from_yaml(args.sft_config)
        sft_checkpoint = run_sft_warmup(sft_config)

    # -- Stage 2: GSPO/DAPO RL --
    rl_config = BioAgentGRPOConfig.from_yaml(args.rl_config)

    # Use SFT checkpoint as starting point for RL
    if sft_checkpoint:
        # Check if it's a LoRA adapter (has adapter_config.json) and merge if needed
        adapter_config_path = os.path.join(sft_checkpoint, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            merged_path = os.path.join(os.path.dirname(sft_checkpoint), "merged")
            if not os.path.exists(os.path.join(merged_path, "config.json")):
                logger.info("SFT checkpoint is a LoRA adapter -- merging into base model...")
                sft_checkpoint = _merge_lora_checkpoint(
                    base_model_path=rl_config.model_name_or_path,
                    adapter_path=sft_checkpoint,
                    output_path=merged_path,
                    torch_dtype=rl_config.torch_dtype,
                )
            else:
                sft_checkpoint = merged_path
                logger.info(f"Using existing merged model: {merged_path}")

        logger.info(f"Using SFT checkpoint for RL: {sft_checkpoint}")
        rl_config.model_name_or_path = sft_checkpoint

    # -- Optional: RIF-RFT Filtering --
    if args.rif_rft:
        filtered_path = run_rif_rft_filtering(rl_config, k=args.rif_rft_k)
        rl_config.tasks_path = filtered_path

    # -- Run RL Training --
    logger.info("=" * 60)
    logger.info("STAGE 2: GSPO/DAPO RL Training")
    logger.info("=" * 60)
    logger.info(f"Model: {rl_config.model_name_or_path}")
    logger.info(f"Loss type: {rl_config.loss_type}")
    logger.info(f"Tasks: {rl_config.tasks_path}")

    if getattr(rl_config, "use_opd", False) or getattr(rl_config, "use_gym_env", False):
        logger.info("Using train_multiturn() (OPD=%s, GYM=%s)" % (
            getattr(rl_config, "use_opd", False),
            getattr(rl_config, "use_gym_env", False),
        ))
        train_multiturn(rl_config)
    else:
        train(rl_config)

    logger.info("=" * 60)
    logger.info("Two-stage training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
