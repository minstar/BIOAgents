#!/usr/bin/env python3
"""Compare agentic task performance: SFT-only vs RL checkpoint.

Runs AgentRunner on medical_qa test tasks to measure:
- Action score (tool-use quality)
- Final reward (composite 6D)
- QA accuracy
- Number of turns
- Process quality

Usage:
    # Evaluate registered models
    CUDA_VISIBLE_DEVICES=X python scripts/eval_agentic_sft_vs_rl.py [--num-tasks 50]

    # Evaluate custom model paths (e.g., v6 checkpoints)
    python scripts/eval_agentic_sft_vs_rl.py \
        --custom-models v6_step80=/path/to/step80/merged_hf \
                        v6_step90=/path/to/step90/merged_hf \
        --num-tasks 50 --domain medical_qa
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from bioagents.evaluation.agent_runner import AgentRunner, RunConfig
from loguru import logger


MODELS = {
    "sft_only": "checkpoints/sft_warmup_lingshu7b_v2_merged/merged",
    "rl_gspo_dapo_step600": "checkpoints/gspo_dapo_lingshu7b_v2_merged/checkpoint-600-merged",
    "rl_drgrpo_step350": "checkpoints/drgrpo_lingshu7b/checkpoint-350-merged",
}


def run_agentic_eval(model_name: str, model_path: str, task_ids: list, log_dir: str,
                     domain: str = "medical_qa", max_turns: int = 10,
                     max_new_tokens: int = 1024):
    """Run agentic evaluation for a single model."""
    abs_path = str(Path(model_path).resolve()) if Path(model_path).exists() else model_path

    config = RunConfig(
        model_name_or_path=abs_path,
        backend="transformers",
        domain=domain,
        task_ids=task_ids,
        task_split=None,
        max_turns=max_turns,
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        log_dir=log_dir,
        seed=42,
    )

    logger.info(f"Evaluating: {model_name} ({abs_path})")
    runner = AgentRunner(config)
    runner.load_model()
    results = runner.run_all_tasks()

    # Compute summary — qa_accuracy is stored in trajectory dict, not as attribute
    action_scores = [r.action_score for r in results if r.action_score is not None]
    rewards = [r.final_reward for r in results if r.final_reward is not None]
    qa_accs = [
        r.trajectory.get("qa_accuracy", 0.0)
        for r in results
        if r.trajectory and "qa_accuracy" in r.trajectory
    ]
    turns = [r.total_turns for r in results]
    completed = sum(1 for r in results if r.completed)

    summary = {
        "model_name": model_name,
        "model_path": model_path,
        "domain": domain,
        "num_tasks": len(results),
        "completed": completed,
        "avg_action_score": sum(action_scores) / max(len(action_scores), 1),
        "avg_reward": sum(rewards) / max(len(rewards), 1),
        "avg_qa_accuracy": sum(qa_accs) / max(len(qa_accs), 1),
        "avg_turns": sum(turns) / max(len(turns), 1),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of test tasks")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Registered model names to evaluate")
    parser.add_argument("--custom-models", nargs="+", default=None,
                        help="Custom model entries as name=path (e.g., v6_step80=/path/to/merged_hf)")
    parser.add_argument("--domain", type=str, default="medical_qa",
                        help="Domain to evaluate on")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max turns per task")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max new tokens per generation")
    parser.add_argument("--offset", type=int, default=0,
                        help="Offset into task list (for parallel splitting)")
    args = parser.parse_args()

    # Build model dict from registered + custom models
    eval_models = {}
    if args.models:
        for name in args.models:
            if name in MODELS:
                eval_models[name] = MODELS[name]
            else:
                logger.warning(f"Unknown registered model: {name}")
    if args.custom_models:
        for entry in args.custom_models:
            if "=" not in entry:
                logger.warning(f"Custom model must be name=path format: {entry}")
                continue
            name, path = entry.split("=", 1)
            eval_models[name] = path

    if not eval_models:
        # Default: registered models
        eval_models = {k: v for k, v in MODELS.items() if Path(v).exists()}
        if not eval_models:
            logger.error("No models found. Use --custom-models name=/path/to/model")
            return

    # Load test task IDs
    tasks_file = f"data/domains/{args.domain}/tasks.json"
    if not Path(tasks_file).exists():
        logger.error(f"Tasks file not found: {tasks_file}")
        return

    with open(tasks_file) as f:
        all_tasks = json.load(f)
        subset = all_tasks[args.offset:args.offset + args.num_tasks] if args.num_tasks > 0 else all_tasks[args.offset:]
        test_ids = [t["id"] for t in subset]

    logger.info(f"Evaluating {len(test_ids)} test tasks from {args.domain}")
    logger.info(f"Models: {list(eval_models.keys())}")

    log_dir = "logs/agentic_eval"
    os.makedirs(log_dir, exist_ok=True)

    all_results = {}
    for model_name, model_path in eval_models.items():
        if not Path(model_path).exists():
            logger.warning(f"Model path not found: {model_path}")
            continue

        summary = run_agentic_eval(
            model_name, model_path, test_ids, log_dir,
            domain=args.domain,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
        )
        all_results[model_name] = summary

        logger.info(f"\n{'='*60}")
        logger.info(f"Results for {model_name}:")
        logger.info(f"  Action Score: {summary['avg_action_score']:.3f}")
        logger.info(f"  Reward:       {summary['avg_reward']:.3f}")
        logger.info(f"  QA Accuracy:  {summary['avg_qa_accuracy']:.3f}")
        logger.info(f"  Avg Turns:    {summary['avg_turns']:.1f}")
        logger.info(f"  Completed:    {summary['completed']}/{summary['num_tasks']}")

        # Free GPU memory between models
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    # Save comparison
    outpath = f"results/agentic_eval/{args.domain}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON:")
    logger.info(f"{'Model':<30} {'Action':>8} {'Reward':>8} {'QA Acc':>8} {'Turns':>6}")
    logger.info("-" * 65)
    for name, s in all_results.items():
        logger.info(f"{name:<30} {s['avg_action_score']:>7.3f} {s['avg_reward']:>7.3f} "
                    f"{s['avg_qa_accuracy']:>7.3f} {s['avg_turns']:>5.1f}")

    logger.info(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
