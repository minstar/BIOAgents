#!/usr/bin/env python3
"""Compare agentic task performance: SFT-only vs RL checkpoint.

Runs AgentRunner on medical_qa test tasks to measure:
- Action score (tool-use quality)
- Final reward (composite 6D)
- QA accuracy
- Number of turns
- Process quality

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/eval_agentic_sft_vs_rl.py [--num-tasks 50]
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


def run_agentic_eval(model_name: str, model_path: str, task_ids: list, log_dir: str):
    """Run agentic evaluation for a single model."""
    abs_path = str(Path(model_path).resolve()) if Path(model_path).exists() else model_path

    config = RunConfig(
        model_name_or_path=abs_path,
        backend="transformers",
        domain="medical_qa",
        task_ids=task_ids,
        task_split=None,
        max_turns=10,
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=1024,
        log_dir=log_dir,
        seed=42,
    )

    logger.info(f"Evaluating: {model_name} ({abs_path})")
    runner = AgentRunner(config)
    runner.load_model()
    results = runner.run_all_tasks()

    # Compute summary
    action_scores = [r.action_score for r in results if r.action_score is not None]
    rewards = [r.final_reward for r in results if r.final_reward is not None]
    qa_accs = [r.qa_accuracy for r in results if r.qa_accuracy is not None]
    turns = [r.total_turns for r in results]
    completed = sum(1 for r in results if r.completed)

    summary = {
        "model_name": model_name,
        "model_path": model_path,
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
    parser.add_argument("--models", nargs="+", default=["sft_only", "rl_gspo_dapo_step600"],
                        help="Models to evaluate")
    args = parser.parse_args()

    # Load test task IDs from medical_qa domain (50 tasks total)
    with open("data/domains/medical_qa/tasks.json") as f:
        all_tasks = json.load(f)
        test_ids = [t["id"] for t in all_tasks[:args.num_tasks]]

    logger.info(f"Evaluating {len(test_ids)} test tasks")

    log_dir = "logs/agentic_eval"
    os.makedirs(log_dir, exist_ok=True)

    all_results = {}
    for model_name in args.models:
        if model_name not in MODELS:
            logger.warning(f"Unknown model: {model_name}")
            continue

        model_path = MODELS[model_name]
        if not Path(model_path).exists():
            logger.warning(f"Model path not found: {model_path}")
            continue

        summary = run_agentic_eval(model_name, model_path, test_ids, log_dir)
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
    outpath = f"results/agentic_eval/sft_vs_rl_{time.strftime('%Y%m%d_%H%M%S')}.json"
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
