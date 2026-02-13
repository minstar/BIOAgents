#!/usr/bin/env python3
"""Run all candidate models on clinical_diagnosis tasks.

Models:
  1. Lingshu-7B (medical specialized VL model) → GPU 0
  2. Qwen2.5-VL-7B-Instruct (general VL model) → GPU 1
  3. Qwen3-8B-Base (text-only baseline) → GPU 2

Each runs sequentially to avoid GPU OOM. Results are saved to logs/runs/.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioagents.evaluation.agent_runner import AgentRunner, RunConfig
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent

MODELS = [
    {
        "name": "Lingshu-7B",
        "path": str(PROJECT_ROOT / "checkpoints/models/Lingshu-7B"),
        "backend": "transformers",
        "gpu": "0,1",
        "notes": "Medical specialized VL model (Qwen2.5-VL based)",
    },
    {
        "name": "Qwen2.5-VL-7B-Instruct",
        "path": str(PROJECT_ROOT / "checkpoints/models/Qwen2.5-VL-7B-Instruct"),
        "backend": "transformers",
        "gpu": "2,3",
        "notes": "General purpose VL model",
    },
    {
        "name": "Qwen3-8B-Base",
        "path": "/data/project/private/minstar/models/Qwen3-8B-Base",
        "backend": "transformers",
        "gpu": "4,5",
        "notes": "Text-only base model (no instruction tuning)",
    },
    {
        "name": "Step3-VL-10B",
        "path": str(PROJECT_ROOT / "checkpoints/models/Step3-VL-10B"),
        "backend": "transformers",
        "gpu": "6,7",
        "notes": "Step3-VL-10B multimodal model (custom architecture)",
    },
]

ALL_TASK_IDS = [
    "dx_pneumonia_001",
    "dx_meningitis_001",
    "dx_appendicitis_001",
    "dx_drug_interaction_001",
    "dx_critical_vitals_001",
]


def run_model(model_info: dict, task_ids: list[str] = None):
    """Run a single model on all tasks."""
    name = model_info["name"]
    path = model_info["path"]
    
    # Check model exists
    if not Path(path).exists():
        logger.error(f"Model {name} not found at {path}")
        return None
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = model_info["gpu"]
    
    config = RunConfig(
        model_name_or_path=path,
        backend=model_info["backend"],
        domain="clinical_diagnosis",
        task_ids=task_ids or ALL_TASK_IDS,
        max_turns=15,
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=1024,
        log_dir=str(PROJECT_ROOT / "logs" / "runs"),
    )
    
    print(f"\n{'='*70}")
    print(f"  Model: {name}")
    print(f"  Path: {path}")
    print(f"  GPU: {model_info['gpu']}")
    print(f"  Notes: {model_info['notes']}")
    print(f"  Tasks: {len(config.task_ids)}")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")
    
    runner = AgentRunner(config)
    
    logger.info(f"Loading model: {name}")
    t0 = time.time()
    runner.load_model()
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")
    
    results = runner.run_all_tasks()
    
    # Clean up GPU memory
    import torch
    del runner.model
    if hasattr(runner, 'processor') and runner.processor is not None:
        del runner.processor
    if hasattr(runner, 'tokenizer'):
        del runner.tokenizer
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return {
        "name": name,
        "results": results,
        "load_time": load_time,
        "run_id": runner.run_id,
    }


def print_final_comparison(all_runs: list[dict]):
    """Print comparison table of all models."""
    print("\n\n" + "=" * 100)
    print("  FINAL COMPARISON - BIOAgents Clinical Diagnosis Baseline")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)
    
    # Header
    header = f"{'Model':<28}"
    for tid in ALL_TASK_IDS:
        short = tid.replace("dx_", "").replace("_001", "")
        header += f" {short:>12}"
    header += f" {'AVG':>8} {'Turns':>6}"
    print(header)
    print("-" * 100)
    
    comparison_data = []
    
    for run_info in all_runs:
        if run_info is None:
            continue
        name = run_info["name"]
        results = run_info["results"]
        
        row = f"{name:<28}"
        scores = []
        total_turns = []
        
        # Map results by task_id
        result_map = {r.task_id: r for r in results}
        
        for tid in ALL_TASK_IDS:
            r = result_map.get(tid)
            if r and r.completed:
                row += f" {r.action_score:>12.3f}"
                scores.append(r.action_score)
                total_turns.append(r.total_turns)
            else:
                row += f" {'ERR':>12}"
        
        avg_score = sum(scores) / max(len(scores), 1)
        avg_turns = sum(total_turns) / max(len(total_turns), 1)
        row += f" {avg_score:>8.3f} {avg_turns:>6.1f}"
        print(row)
        
        comparison_data.append({
            "model": name,
            "avg_action_score": avg_score,
            "avg_turns": avg_turns,
            "per_task": {
                r.task_id: {
                    "action_score": r.action_score,
                    "final_reward": r.final_reward,
                    "turns": r.total_turns,
                    "completed": r.completed,
                }
                for r in results
            },
            "load_time": run_info["load_time"],
            "run_id": run_info["run_id"],
        })
    
    print("=" * 100)
    
    # Save comparison
    comp_path = PROJECT_ROOT / "logs" / "runs" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comp_path, "w") as f:
        json.dump(comparison_data, f, indent=2, default=str)
    print(f"\nComparison saved to: {comp_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to run (default: all)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task IDs to run (default: all)")
    args = parser.parse_args()
    
    models_to_run = MODELS
    if args.models:
        models_to_run = [m for m in MODELS if m["name"] in args.models]
    
    task_ids = args.tasks
    
    print("=" * 70)
    print("  BIOAgents - Multi-Model Clinical Diagnosis Experiment")
    print(f"  Models: {[m['name'] for m in models_to_run]}")
    print(f"  Tasks: {task_ids or 'ALL'}")
    print(f"  Start: {datetime.now().isoformat()}")
    print("=" * 70)
    
    all_runs = []
    for model_info in models_to_run:
        try:
            result = run_model(model_info, task_ids)
            all_runs.append(result)
        except Exception as e:
            logger.error(f"Failed for {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_runs.append(None)
    
    print_final_comparison(all_runs)
    
    print(f"\nAll experiments completed at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
