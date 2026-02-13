"""Run experiments on the medical_qa domain with candidate models.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/run_medqa_experiment.py \
        --model Qwen3-8B-Base --domain medical_qa
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioagents.evaluation.agent_runner import AgentRunner, RunConfig
from loguru import logger


MODEL_PATHS = {
    "Qwen3-8B-Base": "/data/project/private/minstar/models/Qwen3-8B-Base",
    "Qwen2.5-VL-7B-Instruct": str(
        Path(__file__).parent.parent
        / "checkpoints"
        / "models"
        / "Qwen2.5-VL-7B-Instruct"
    ),
    "Lingshu-7B": str(
        Path(__file__).parent.parent / "checkpoints" / "models" / "Lingshu-7B"
    ),
}


def run_experiment(
    model_name: str,
    domain: str = "medical_qa",
    task_ids: list[str] = None,
    max_turns: int = 12,
    backend: str = "transformers",
):
    """Run a model on the specified domain."""
    model_path = MODEL_PATHS.get(model_name, model_name)

    config = RunConfig(
        model_name_or_path=model_path,
        backend=backend,
        domain=domain,
        task_ids=task_ids,
        max_turns=max_turns,
        temperature=0.1,
        max_new_tokens=1024,
        log_dir="logs/runs",
    )

    runner = AgentRunner(config)
    runner.load_model()
    results = runner.run_all_tasks()

    # Compute additional QA metrics
    if domain == "medical_qa":
        correct = 0
        answered = 0
        for r in results:
            qa_acc = r.trajectory.get("qa_accuracy", None)
            if qa_acc is not None:
                answered += 1
                if qa_acc > 0:
                    correct += 1

        logger.info(f"\n=== QA Accuracy: {correct}/{answered} = "
                     f"{correct/max(answered,1):.3f} ===")

        # Save QA summary
        qa_summary = {
            "model": model_name,
            "domain": domain,
            "total_tasks": len(results),
            "answered": answered,
            "correct": correct,
            "accuracy": correct / max(answered, 1),
            "avg_action_score": sum(r.action_score for r in results) / max(len(results), 1),
            "avg_turns": sum(r.total_turns for r in results) / max(len(results), 1),
            "per_task": [
                {
                    "task_id": r.task_id,
                    "action_score": r.action_score,
                    "qa_accuracy": r.trajectory.get("qa_accuracy", None),
                    "turns": r.total_turns,
                }
                for r in results
            ],
            "timestamp": datetime.now().isoformat(),
        }
        summary_path = runner.log_path / "qa_summary.json"
        with open(summary_path, "w") as f:
            json.dump(qa_summary, f, indent=2)
        logger.info(f"QA summary saved to {summary_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen3-8B-Base",
                        choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--domain", default="medical_qa")
    parser.add_argument("--task_ids", nargs="+", default=None,
                        help="Specific task IDs to run (None=all)")
    parser.add_argument("--max_turns", type=int, default=12)
    parser.add_argument("--backend", default="transformers")
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        domain=args.domain,
        task_ids=args.task_ids,
        max_turns=args.max_turns,
        backend=args.backend,
    )
