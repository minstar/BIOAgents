#!/usr/bin/env python3
"""Run BIOAgents clinical diagnosis experiments with candidate models.

Usage:
    # Single model
    python scripts/run_experiment.py --model Qwen2.5-VL-7B-Instruct --backend vllm

    # All candidate models
    python scripts/run_experiment.py --all

    # Specific task
    python scripts/run_experiment.py --model Qwen2.5-VL-7B-Instruct --task dx_pneumonia_001
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bioagents.evaluation.agent_runner import AgentRunner, RunConfig
from loguru import logger

# Model paths
MODEL_REGISTRY = {
    "Qwen2.5-VL-7B-Instruct": "checkpoints/models/Qwen2.5-VL-7B-Instruct",
    "Lingshu-7B": "checkpoints/models/Lingshu-7B",
    "Qwen3-8B-Base": "/data/project/private/minstar/models/Qwen3-8B-Base",
}

PROJECT_ROOT = Path(__file__).parent.parent


def resolve_model_path(name: str) -> str:
    """Resolve model name to absolute path."""
    if name in MODEL_REGISTRY:
        path = PROJECT_ROOT / MODEL_REGISTRY[name]
    else:
        path = Path(name)
    
    if not path.exists():
        # Try HuggingFace hub ID
        return name
    return str(path.resolve())


def run_single_experiment(
    model_name: str,
    backend: str = "vllm",
    task_ids: list[str] = None,
    tp_size: int = 1,
):
    """Run experiment with a single model."""
    model_path = resolve_model_path(model_name)
    
    config = RunConfig(
        model_name_or_path=model_path,
        backend=backend,
        domain="clinical_diagnosis",
        task_ids=task_ids,
        max_turns=15,
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=1024,
        tensor_parallel_size=tp_size,
        log_dir=str(PROJECT_ROOT / "logs" / "runs"),
    )
    
    logger.info(f"Starting experiment: {model_name} ({backend})")
    logger.info(f"Model path: {model_path}")
    
    runner = AgentRunner(config)
    runner.load_model()
    results = runner.run_all_tasks()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="BIOAgents Experiment Runner")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct",
                        help="Model name (from registry) or path")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "transformers"],
                        help="Inference backend")
    parser.add_argument("--task", type=str, default=None,
                        help="Specific task ID (default: all tasks)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--all", action="store_true",
                        help="Run all candidate models")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models")
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable models:")
        for name, path in MODEL_REGISTRY.items():
            full_path = PROJECT_ROOT / path
            exists = "✓" if full_path.exists() else "✗ (not downloaded)"
            print(f"  {name:<30} {exists}  → {path}")
        return
    
    task_ids = [args.task] if args.task else None
    
    if args.all:
        all_results = {}
        for model_name in MODEL_REGISTRY:
            model_path = PROJECT_ROOT / MODEL_REGISTRY[model_name]
            if not model_path.exists():
                logger.warning(f"Skipping {model_name}: not found at {model_path}")
                continue
            try:
                results = run_single_experiment(
                    model_name, args.backend, task_ids, args.tp
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Failed for {model_name}: {e}")
                all_results[model_name] = str(e)
        
        # Print comparison
        print_comparison(all_results)
    else:
        run_single_experiment(args.model, args.backend, task_ids, args.tp)


def print_comparison(all_results: dict):
    """Print a comparison table of all models."""
    print("\n" + "=" * 90)
    print("  MODEL COMPARISON - Clinical Diagnosis Domain")
    print("=" * 90)
    print(f"  {'Model':<30} {'Tasks':>6} {'Avg Score':>10} {'Avg Reward':>11} {'Avg Turns':>10}")
    print("-" * 90)
    
    for model_name, results in all_results.items():
        if isinstance(results, str):
            print(f"  {model_name:<30} ERROR: {results[:40]}")
            continue
        
        n = len(results)
        avg_score = sum(r.action_score for r in results) / max(n, 1)
        avg_reward = sum(r.final_reward for r in results) / max(n, 1)
        avg_turns = sum(r.total_turns for r in results) / max(n, 1)
        print(f"  {model_name:<30} {n:>6} {avg_score:>10.3f} {avg_reward:>11.3f} {avg_turns:>10.1f}")
    
    print("=" * 90)


if __name__ == "__main__":
    main()
