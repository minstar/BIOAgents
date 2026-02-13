#!/usr/bin/env python3
"""Multi-domain baseline evaluation for BIOAgents GYM.

Runs candidate models across ALL 5 medical domains to establish
pre-training baseline performance.

Domains:
  1. clinical_diagnosis  — 5 tasks  (all)
  2. medical_qa          — 15 tasks (test split)
  3. visual_diagnosis    — 8 tasks  (all)
  4. drug_interaction    — 5 tasks  (all)
  5. ehr_management      — 7 tasks  (test split)

Models:
  1. Qwen3-8B-Base              (text-only base model)
  2. Qwen2.5-VL-7B-Instruct    (general VLM, instruction-tuned)
  3. Lingshu-7B                 (medical specialized VLM)

Usage:
  # Run all models on all domains (sequential)
  python scripts/run_multidomain_baseline.py

  # Run specific model(s)
  python scripts/run_multidomain_baseline.py --models Qwen3-8B-Base

  # Run specific domain(s)
  python scripts/run_multidomain_baseline.py --domains clinical_diagnosis ehr_management

  # Run with specific GPU assignment
  CUDA_VISIBLE_DEVICES=0,1 python scripts/run_multidomain_baseline.py --models Qwen3-8B-Base
"""

import argparse
import gc
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

# ── Model Registry ──────────────────────────────────────────────────────────

MODELS = {
    "Qwen3-8B-Base": {
        "path": "/data/project/private/minstar/models/Qwen3-8B-Base",
        "backend": "transformers",
        "default_gpu": "0,1",
        "notes": "Text-only base model (no instruction tuning, strong tool-call via pretraining)",
    },
    "Qwen2.5-VL-7B-Instruct": {
        "path": str(PROJECT_ROOT / "checkpoints/models/Qwen2.5-VL-7B-Instruct"),
        "backend": "transformers",
        "default_gpu": "2,3",
        "notes": "General purpose VL model, instruction-tuned",
    },
    "Lingshu-7B": {
        "path": str(PROJECT_ROOT / "checkpoints/models/Lingshu-7B"),
        "backend": "transformers",
        "default_gpu": "4,5",
        "notes": "Medical specialized VL model (Qwen2.5-VL based)",
    },
    "Lingshu-7B-SFT": {
        "path": str(PROJECT_ROOT / "checkpoints/sft_multidomain_lingshu/merged"),
        "backend": "transformers",
        "default_gpu": "4,5",
        "notes": "Lingshu-7B + multi-domain tool-calling SFT (P1)",
    },
}

# ── Domain Registry ─────────────────────────────────────────────────────────

DOMAINS = {
    "clinical_diagnosis": {
        "task_split": None,          # use all 5 tasks
        "max_turns": 15,
        "description": "Patient assessment → differential diagnosis → treatment plan",
    },
    "medical_qa": {
        "task_split": "test",        # use 15 test tasks
        "max_turns": 12,
        "description": "Evidence-based medical QA (MedQA/MedMCQA/MMLU)",
    },
    "visual_diagnosis": {
        "task_split": None,          # use all 8 tasks
        "max_turns": 10,
        "description": "Medical image analysis and visual question answering",
    },
    "drug_interaction": {
        "task_split": None,          # use all 5 tasks
        "max_turns": 12,
        "description": "Drug-drug interaction detection and management",
    },
    "ehr_management": {
        "task_split": "test",        # use 7 test tasks
        "max_turns": 15,
        "description": "EHR chart review, clinical scoring, discharge planning",
    },
}


# ── Helper: GPU memory cleanup ──────────────────────────────────────────────

def cleanup_gpu():
    """Release GPU memory."""
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


# ── Core: Run one model × one domain ────────────────────────────────────────

def run_single(
    model_name: str,
    domain_name: str,
    gpu: str | None = None,
) -> dict | None:
    """Run a single model on a single domain and return structured result."""
    model_info = MODELS[model_name]
    domain_info = DOMAINS[domain_name]

    model_path = model_info["path"]
    if not Path(model_path).exists():
        logger.error(f"[SKIP] Model not found: {model_name} ({model_path})")
        return None

    # GPU assignment
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = model_info["default_gpu"]

    config = RunConfig(
        model_name_or_path=model_path,
        backend=model_info["backend"],
        domain=domain_name,
        task_split=domain_info["task_split"],
        max_turns=domain_info["max_turns"],
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=1024,
        log_dir=str(PROJECT_ROOT / "logs" / "baseline"),
    )

    logger.info(f"{'='*70}")
    logger.info(f"  Model : {model_name}")
    logger.info(f"  Domain: {domain_name} ({domain_info['description']})")
    logger.info(f"  Split : {domain_info['task_split'] or 'all'}")
    logger.info(f"  GPU   : {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    logger.info(f"  Time  : {datetime.now().isoformat()}")
    logger.info(f"{'='*70}")

    runner = AgentRunner(config)

    # Load model
    t0 = time.time()
    runner.load_model()
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    # Run all tasks
    results = runner.run_all_tasks()

    # ── Compute domain-specific metrics ──────────────────────────────────
    metrics = _compute_domain_metrics(domain_name, results)
    metrics["load_time_s"] = load_time
    metrics["run_id"] = runner.run_id

    # Save per-domain result
    result_file = runner.log_path / "domain_metrics.json"
    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Cleanup
    try:
        import torch
        del runner.model
        if hasattr(runner, "processor") and runner.processor is not None:
            del runner.processor
        if hasattr(runner, "tokenizer"):
            del runner.tokenizer
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    return {
        "model": model_name,
        "domain": domain_name,
        "metrics": metrics,
        "run_id": runner.run_id,
    }


def _compute_domain_metrics(domain_name: str, results: list) -> dict:
    """Compute per-domain metrics from task results."""
    n = len(results)
    completed = [r for r in results if r.completed]
    errors = [r for r in results if r.error]

    # Common metrics
    avg_action = sum(r.action_score for r in completed) / max(len(completed), 1)
    avg_reward = sum(r.final_reward for r in completed) / max(len(completed), 1)
    avg_turns = sum(r.total_turns for r in completed) / max(len(completed), 1)
    total_latency = sum(r.total_latency for r in results)

    metrics = {
        "domain": domain_name,
        "num_tasks": n,
        "num_completed": len(completed),
        "num_errors": len(errors),
        "avg_action_score": round(avg_action, 4),
        "avg_reward": round(avg_reward, 4),
        "avg_turns": round(avg_turns, 1),
        "total_latency_s": round(total_latency, 1),
    }

    # QA accuracy (medical_qa domain)
    if domain_name == "medical_qa":
        correct = 0
        answered = 0
        for r in completed:
            qa_acc = r.trajectory.get("qa_accuracy", None)
            if qa_acc is not None:
                answered += 1
                if qa_acc > 0:
                    correct += 1
        metrics["qa_answered"] = answered
        metrics["qa_correct"] = correct
        metrics["qa_accuracy"] = round(correct / max(answered, 1), 4)

    # Per-task breakdown
    metrics["per_task"] = {}
    for r in results:
        entry = {
            "action_score": round(r.action_score, 4),
            "final_reward": round(r.final_reward, 4),
            "turns": r.total_turns,
            "latency_s": round(r.total_latency, 1),
            "completed": r.completed,
        }
        if r.trajectory.get("qa_accuracy") is not None:
            entry["qa_accuracy"] = r.trajectory["qa_accuracy"]
        if r.trajectory.get("reward_details"):
            rd = r.trajectory["reward_details"]
            entry["reward_details"] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in rd.items()
            }
        if r.error:
            entry["error"] = r.error[:200]
        metrics["per_task"][r.task_id] = entry

    return metrics


# ── Results Aggregation & Reporting ──────────────────────────────────────────

def aggregate_results(all_runs: list[dict]) -> dict:
    """Aggregate all runs into a structured comparison."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "experiment": "BIOAgents Multi-Domain Baseline Evaluation",
        "timestamp": datetime.now().isoformat(),
        "models_tested": list({r["model"] for r in all_runs if r}),
        "domains_tested": list({r["domain"] for r in all_runs if r}),
        "results": [],
    }

    # Build model × domain matrix
    model_domain_results = {}
    for run in all_runs:
        if run is None:
            continue
        key = (run["model"], run["domain"])
        model_domain_results[key] = run["metrics"]

    # Model-level aggregation
    models = sorted({r["model"] for r in all_runs if r})
    domains = list(DOMAINS.keys())

    for model in models:
        model_entry = {
            "model": model,
            "domains": {},
            "overall": {},
        }
        all_action_scores = []
        all_rewards = []

        for domain in domains:
            m = model_domain_results.get((model, domain))
            if m:
                model_entry["domains"][domain] = {
                    "action_score": m["avg_action_score"],
                    "reward": m["avg_reward"],
                    "turns": m["avg_turns"],
                    "tasks": m["num_tasks"],
                    "completed": m["num_completed"],
                    "latency_s": m["total_latency_s"],
                }
                if "qa_accuracy" in m:
                    model_entry["domains"][domain]["qa_accuracy"] = m["qa_accuracy"]
                all_action_scores.append(m["avg_action_score"])
                all_rewards.append(m["avg_reward"])
            else:
                model_entry["domains"][domain] = None

        if all_action_scores:
            model_entry["overall"] = {
                "avg_action_score": round(
                    sum(all_action_scores) / len(all_action_scores), 4
                ),
                "avg_reward": round(
                    sum(all_rewards) / len(all_rewards), 4
                ),
                "domains_completed": len(all_action_scores),
            }

        report["results"].append(model_entry)

    return report


def print_comparison_table(report: dict):
    """Print a formatted comparison table."""
    DOMAIN_SHORT = {
        "clinical_diagnosis": "ClinDx",
        "medical_qa": "MedQA",
        "visual_diagnosis": "VisDx",
        "drug_interaction": "DrugInt",
        "ehr_management": "EHR",
    }

    domains = list(DOMAINS.keys())
    models = [r["model"] for r in report["results"]]

    print("\n")
    print("=" * 110)
    print("  BIOAgents Multi-Domain Baseline Evaluation")
    print(f"  Date: {report['timestamp'][:19]}")
    print("=" * 110)

    # ── Action Score Table ───────────────────────────────────────────────
    print("\n  📊 Action Score (tool-use accuracy)")
    print("-" * 110)
    header = f"  {'Model':<28}"
    for d in domains:
        header += f" {DOMAIN_SHORT[d]:>10}"
    header += f" {'OVERALL':>10}"
    print(header)
    print("-" * 110)

    for entry in report["results"]:
        row = f"  {entry['model']:<28}"
        for d in domains:
            dm = entry["domains"].get(d)
            if dm:
                row += f" {dm['action_score']:>10.3f}"
            else:
                row += f" {'—':>10}"
        ov = entry.get("overall", {})
        row += f" {ov.get('avg_action_score', 0):>10.3f}"
        print(row)
    print("-" * 110)

    # ── Reward Table ─────────────────────────────────────────────────────
    print("\n  🎯 Composite Reward")
    print("-" * 110)
    header = f"  {'Model':<28}"
    for d in domains:
        header += f" {DOMAIN_SHORT[d]:>10}"
    header += f" {'OVERALL':>10}"
    print(header)
    print("-" * 110)

    for entry in report["results"]:
        row = f"  {entry['model']:<28}"
        for d in domains:
            dm = entry["domains"].get(d)
            if dm:
                row += f" {dm['reward']:>10.3f}"
            else:
                row += f" {'—':>10}"
        ov = entry.get("overall", {})
        row += f" {ov.get('avg_reward', 0):>10.3f}"
        print(row)
    print("-" * 110)

    # ── Turns Table ──────────────────────────────────────────────────────
    print("\n  🔄 Avg Turns per Task")
    print("-" * 110)
    header = f"  {'Model':<28}"
    for d in domains:
        header += f" {DOMAIN_SHORT[d]:>10}"
    print(header)
    print("-" * 110)

    for entry in report["results"]:
        row = f"  {entry['model']:<28}"
        for d in domains:
            dm = entry["domains"].get(d)
            if dm:
                row += f" {dm['turns']:>10.1f}"
            else:
                row += f" {'—':>10}"
        print(row)
    print("-" * 110)

    # ── QA Accuracy (if available) ───────────────────────────────────────
    qa_available = any(
        entry["domains"].get("medical_qa", {})
        and "qa_accuracy" in (entry["domains"].get("medical_qa") or {})
        for entry in report["results"]
    )
    if qa_available:
        print("\n  📝 Medical QA Accuracy (exact match)")
        print("-" * 50)
        for entry in report["results"]:
            dm = entry["domains"].get("medical_qa")
            if dm and "qa_accuracy" in dm:
                print(f"  {entry['model']:<28} {dm['qa_accuracy']:>10.3f}")
        print("-" * 50)

    print("\n" + "=" * 110)


def save_report(report: dict, all_runs: list[dict]):
    """Save the full report and detailed results."""
    out_dir = PROJECT_ROOT / "logs" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Summary report
    summary_path = out_dir / f"baseline_report_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  📁 Summary report: {summary_path}")

    # Full detailed results
    detail_path = out_dir / f"baseline_detail_{ts}.json"
    detail = []
    for run in all_runs:
        if run is None:
            continue
        detail.append({
            "model": run["model"],
            "domain": run["domain"],
            "run_id": run["run_id"],
            "metrics": run["metrics"],
        })
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)
    print(f"  📁 Detailed results: {detail_path}")

    return summary_path, detail_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BIOAgents Multi-Domain Baseline Evaluation"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to evaluate (default: all). Choices: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--domains", nargs="+", default=None,
        help=f"Domains to evaluate (default: all). Choices: {list(DOMAINS.keys())}",
    )
    parser.add_argument(
        "--gpu", default=None,
        help="Override CUDA_VISIBLE_DEVICES for all runs",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without running",
    )
    args = parser.parse_args()

    # Resolve models and domains
    model_names = args.models or list(MODELS.keys())
    domain_names = args.domains or list(DOMAINS.keys())

    # Validate
    for m in model_names:
        if m not in MODELS:
            logger.error(f"Unknown model: {m}. Available: {list(MODELS.keys())}")
            sys.exit(1)
    for d in domain_names:
        if d not in DOMAINS:
            logger.error(f"Unknown domain: {d}. Available: {list(DOMAINS.keys())}")
            sys.exit(1)

    # Plan
    total_runs = len(model_names) * len(domain_names)
    print("\n" + "=" * 80)
    print("  🏥 BIOAgents Multi-Domain Baseline Evaluation")
    print("=" * 80)
    print(f"  Models  : {model_names}")
    print(f"  Domains : {domain_names}")
    print(f"  Total   : {total_runs} runs ({len(model_names)} models × {len(domain_names)} domains)")
    print(f"  GPU     : {args.gpu or 'per-model default'}")
    print(f"  Output  : logs/baseline/")
    print(f"  Start   : {datetime.now().isoformat()}")
    print("=" * 80)

    if args.dry_run:
        print("\n  [DRY RUN] Would execute:")
        for mi, m in enumerate(model_names):
            for di, d in enumerate(domain_names):
                gpu = args.gpu or MODELS[m]["default_gpu"]
                split = DOMAINS[d]["task_split"] or "all"
                print(f"    {mi*len(domain_names)+di+1:>3}. {m} × {d} (split={split}, gpu={gpu})")
        print("\n  Pass without --dry-run to execute.")
        return

    # Execute
    all_runs = []
    start = time.time()

    for mi, model_name in enumerate(model_names):
        for di, domain_name in enumerate(domain_names):
            run_idx = mi * len(domain_names) + di + 1
            logger.info(
                f"\n[{run_idx}/{total_runs}] {model_name} × {domain_name}"
            )
            try:
                result = run_single(model_name, domain_name, gpu=args.gpu)
                all_runs.append(result)
            except Exception as e:
                logger.error(f"FAILED: {model_name} × {domain_name}: {e}")
                import traceback
                traceback.print_exc()
                all_runs.append(None)
            finally:
                cleanup_gpu()

    elapsed = time.time() - start
    logger.info(f"\nAll {total_runs} runs completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Aggregate and report
    report = aggregate_results(all_runs)
    print_comparison_table(report)
    save_report(report, all_runs)

    print(f"\n  ✅ Evaluation complete! Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Finished at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
