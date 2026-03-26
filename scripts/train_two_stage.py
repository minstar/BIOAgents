"""Two-Stage Training Pipeline: SFT Warmup → GSPO/DAPO RL.

Implements the training approach proven by MedAgentGym (ICLR 2026)
and WebRL (ICLR 2025): warm up with supervised fine-tuning on
successful demonstrations first, then apply RL for further improvement.

Cold-start RL (directly applying GRPO/GSPO to a base model) risks
catastrophic collapse — the B041 issue.

Usage:
    python scripts/train_two_stage.py \
        --sft-config configs/sft_medical_qa.yaml \
        --rl-config configs/gspo_dapo_lingshu7b.yaml \
        [--skip-sft]              # Skip SFT if warmup already done
        [--sft-checkpoint PATH]   # Use existing SFT checkpoint for RL stage
        [--rif-rft]               # Enable RIF-RFT sample filtering
        [--rif-rft-k 8]           # Number of rollouts for RIF-RFT filtering

Reference:
    - MedAgentGym (ICLR 2026): SFT → DPO two-stage
    - WebRL (ICLR 2025): SFT warmup → Adaptive RL
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

from bioagents.training.grpo_trainer import BioAgentGRPOConfig, train
from bioagents.training.sft_trainer import BioAgentSFTConfig


def run_sft_warmup(sft_config: BioAgentSFTConfig) -> str:
    """Run SFT warmup stage and return path to best checkpoint."""
    from bioagents.training.sft_trainer import train as sft_train

    logger.info("=" * 60)
    logger.info("STAGE 1: SFT Warmup")
    logger.info("=" * 60)
    logger.info(f"Model: {sft_config.model_name_or_path}")
    logger.info(f"Output: {sft_config.output_dir}")

    sft_train(sft_config)

    # Find the final/best checkpoint
    final_path = os.path.join(sft_config.output_dir, "final")
    if os.path.exists(final_path):
        logger.info(f"SFT warmup complete. Checkpoint: {final_path}")
        return final_path

    # Fall back to last checkpoint
    checkpoints = sorted(Path(sft_config.output_dir).glob("checkpoint-*"))
    if checkpoints:
        best = str(checkpoints[-1])
        logger.info(f"SFT warmup complete. Using last checkpoint: {best}")
        return best

    logger.warning("No checkpoint found after SFT. Using output_dir directly.")
    return sft_config.output_dir


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
    parser = argparse.ArgumentParser(description="Two-stage SFT → RL training pipeline")
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

    # ── Stage 1: SFT Warmup ──
    sft_checkpoint = args.sft_checkpoint

    if not args.skip_sft and not sft_checkpoint:
        if not args.sft_config:
            logger.error("--sft-config required unless --skip-sft or --sft-checkpoint is set")
            sys.exit(1)

        sft_config = BioAgentSFTConfig.from_yaml(args.sft_config)
        sft_checkpoint = run_sft_warmup(sft_config)

    # ── Stage 2: GSPO/DAPO RL ──
    rl_config = BioAgentGRPOConfig.from_yaml(args.rl_config)

    # Use SFT checkpoint as starting point for RL
    if sft_checkpoint:
        logger.info(f"Using SFT checkpoint for RL: {sft_checkpoint}")
        rl_config.model_name_or_path = sft_checkpoint

    # ── Optional: RIF-RFT Filtering ──
    if args.rif_rft:
        filtered_path = run_rif_rft_filtering(rl_config, k=args.rif_rft_k)
        rl_config.tasks_path = filtered_path

    # ── Run RL Training ──
    logger.info("=" * 60)
    logger.info("STAGE 2: GSPO/DAPO RL Training")
    logger.info("=" * 60)
    logger.info(f"Model: {rl_config.model_name_or_path}")
    logger.info(f"Loss type: {rl_config.loss_type}")
    logger.info(f"Tasks: {rl_config.tasks_path}")

    train(rl_config)

    logger.info("=" * 60)
    logger.info("Two-stage training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
