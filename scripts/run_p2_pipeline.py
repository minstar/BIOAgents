#!/usr/bin/env python3
"""P2 Healthcare AI GYM Training Pipeline.

Orchestrates the complete P2 training sequence:
  P2-1: SFT data generation (701 examples) ✅ Already done
  P2-2: Aggressive SFT on Lingshu-7B (r=16 + FFN)
  P2-3: Qwen3-8B GRPO (turn efficiency)
  P2-4: Multi-domain GRPO (corrected rewards)

Usage:
    python scripts/run_p2_pipeline.py --stage all
    python scripts/run_p2_pipeline.py --stage sft
    python scripts/run_p2_pipeline.py --stage grpo
    python scripts/run_p2_pipeline.py --stage eval
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: str, description: str, cwd: str = None) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  CMD: {cmd}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(
        cmd, shell=True,
        cwd=cwd or str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )
    elapsed = time.time() - start
    status = "✅ SUCCESS" if result.returncode == 0 else "❌ FAILED"
    print(f"\n  {status} ({elapsed:.1f}s)")
    return result.returncode == 0


# ── Stage: SFT Training ──────────────────────────────────────────────


def stage_sft():
    """P2-2: Aggressive SFT training on Lingshu-7B."""
    print("\n" + "=" * 70)
    print("  P2-2: Aggressive SFT (r=16 + FFN) on Lingshu-7B")
    print("  Dataset: 701 examples, 3 epochs, LR=2e-5")
    print("=" * 70)

    config_path = "configs/sft_p2_aggressive_lingshu.yaml"

    # Dry run first
    ok = run_command(
        f"python -m bioagents.training.sft_trainer --config {config_path} --dry_run",
        "SFT Dry Run (validate dataset)",
    )
    if not ok:
        print("  ❌ Dry run failed. Aborting SFT.")
        return False

    # Full training
    ok = run_command(
        f"python -m bioagents.training.sft_trainer --config {config_path}",
        "SFT Full Training",
    )
    if not ok:
        print("  ❌ SFT training failed.")
        return False

    # Merge LoRA adapter
    print("\n  Merging LoRA adapter...")
    merge_script = f"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

base_path = "checkpoints/models/Lingshu-7B"
adapter_path = "checkpoints/sft_p2_aggressive_lingshu/final"
merged_path = "checkpoints/sft_p2_aggressive_lingshu/merged"

config = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
model_type = getattr(config, "model_type", "")
is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl")

if is_qwen_vl:
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
model.save_pretrained(merged_path)

tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
tokenizer.save_pretrained(merged_path)
print(f"Merged model saved to {{merged_path}}")
"""
    ok = run_command(
        f'python -c "{merge_script}"',
        "Merge LoRA Adapter",
    )
    return ok


# ── Stage: GRPO Training ─────────────────────────────────────────────


def stage_grpo_qwen3():
    """P2-3: Qwen3-8B GRPO training."""
    print("\n" + "=" * 70)
    print("  P2-3: Qwen3-8B GRPO (Turn Efficiency)")
    print("=" * 70)

    config_path = "configs/grpo_p2_qwen3_8b.yaml"

    # Dry run
    ok = run_command(
        f"python -m bioagents.training.grpo_trainer --config {config_path} --dry_run",
        "GRPO Dry Run (Qwen3-8B)",
    )
    if not ok:
        print("  ❌ GRPO dry run failed.")
        return False

    # Full training
    ok = run_command(
        f"python -m bioagents.training.grpo_trainer --config {config_path}",
        "GRPO Full Training (Qwen3-8B)",
    )
    return ok


def stage_grpo_multidomain():
    """P2-4: Multi-domain GRPO training."""
    print("\n" + "=" * 70)
    print("  P2-4: Multi-domain GRPO (Corrected Rewards)")
    print("=" * 70)

    # Check if SFT merged model exists
    merged_path = PROJECT_ROOT / "checkpoints" / "sft_p2_aggressive_lingshu" / "merged"
    if not merged_path.exists():
        print("  ⚠️ SFT merged model not found. Run --stage sft first.")
        print("  Falling back to base Lingshu-7B...")
        # Update config to use base model
        config_path = "configs/grpo_p2_multidomain.yaml"
    else:
        config_path = "configs/grpo_p2_multidomain.yaml"

    # Dry run
    ok = run_command(
        f"python -m bioagents.training.grpo_trainer --config {config_path} --dry_run",
        "Multi-domain GRPO Dry Run",
    )
    if not ok:
        return False

    # Full training
    ok = run_command(
        f"python -m bioagents.training.grpo_trainer --config {config_path}",
        "Multi-domain GRPO Full Training",
    )
    return ok


# ── Stage: Evaluation ─────────────────────────────────────────────────


def stage_eval():
    """P2-eval: Evaluate all P2 checkpoints across domains."""
    print("\n" + "=" * 70)
    print("  P2-eval: Multi-Domain Evaluation")
    print("=" * 70)

    eval_script = PROJECT_ROOT / "scripts" / "run_p2_eval.py"
    if not eval_script.exists():
        print("  Creating evaluation script...")
        _create_eval_script(eval_script)

    ok = run_command(
        f"python {eval_script}",
        "P2 Multi-domain Evaluation",
    )
    return ok


def _create_eval_script(path: Path):
    """Create the P2 evaluation script."""
    path.write_text('''#!/usr/bin/env python3
"""P2 evaluation: Run all P2 checkpoints across 5 domains."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioagents.evaluation.agent_runner import AgentRunner, RunConfig
from bioagents.gym.agent_env import BioAgentGymEnv

PROJECT_ROOT = Path(__file__).parent.parent

# Models to evaluate
MODELS = []

# P2-2 SFT checkpoint
sft_merged = PROJECT_ROOT / "checkpoints" / "sft_p2_aggressive_lingshu" / "merged"
if sft_merged.exists():
    MODELS.append(("P2-SFT-Lingshu", str(sft_merged)))

# P2-3 GRPO checkpoint
grpo_qwen3 = PROJECT_ROOT / "checkpoints" / "grpo_p2_qwen3_8b" / "final"
if grpo_qwen3.exists():
    MODELS.append(("P2-GRPO-Qwen3", str(grpo_qwen3)))

# P2-4 Multi-domain GRPO
grpo_multi = PROJECT_ROOT / "checkpoints" / "grpo_p2_multidomain" / "final"
if grpo_multi.exists():
    MODELS.append(("P2-GRPO-Multi", str(grpo_multi)))

# Original baselines
MODELS.append(("P1-SFT-Lingshu", str(PROJECT_ROOT / "checkpoints" / "sft_multidomain_lingshu" / "merged")))

DOMAINS = ["clinical_diagnosis", "visual_diagnosis", "drug_interaction", "ehr_management"]
# Note: medical_qa excluded from eval since GRPO trains on it

def main():
    if not MODELS:
        print("No P2 checkpoints found. Run training first.")
        return

    results = {}
    for model_name, model_path in MODELS:
        if not Path(model_path).exists():
            print(f"  Skipping {model_name}: {model_path} not found")
            continue

        results[model_name] = {}
        for domain in DOMAINS:
            print(f"\\n  Evaluating {model_name} on {domain}...")
            try:
                config = RunConfig(
                    model_name_or_path=model_path,
                    backend="transformers",
                    domain=domain,
                    max_turns=15,
                    temperature=0.1,
                    log_dir=str(PROJECT_ROOT / "logs" / "p2_eval"),
                )
                runner = AgentRunner(config)
                runner.load_model()
                task_results = runner.run_all_tasks()

                avg_action = sum(r.action_score for r in task_results) / max(len(task_results), 1)
                avg_reward = sum(r.final_reward for r in task_results) / max(len(task_results), 1)
                results[model_name][domain] = {
                    "action_score": round(avg_action, 3),
                    "reward": round(avg_reward, 3),
                    "tasks": len(task_results),
                }
            except Exception as e:
                print(f"    Error: {e}")
                results[model_name][domain] = {"error": str(e)}

    # Print summary
    print("\\n" + "=" * 80)
    print("  P2 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  {'Model':<25} {'ClinDx':>8} {'VisDx':>8} {'DrugInt':>8} {'EHR':>8} {'AVG':>8}")
    print("-" * 80)
    for model_name, domains in results.items():
        scores = []
        row = f"  {model_name:<25}"
        for d in DOMAINS:
            if d in domains and "action_score" in domains[d]:
                s = domains[d]["action_score"]
                row += f" {s:>8.3f}"
                scores.append(s)
            else:
                row += f" {'ERR':>8}"
        if scores:
            row += f" {sum(scores)/len(scores):>8.3f}"
        print(row)
    print("=" * 80)

    # Save
    output_path = PROJECT_ROOT / "logs" / "p2_eval" / f"p2_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_path}")

if __name__ == "__main__":
    main()
''')


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="P2 Healthcare AI GYM Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "sft", "grpo", "grpo_qwen3", "grpo_multi", "eval"],
        help="Which stage to run (default: all)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  Healthcare AI GYM — P2 Training Pipeline")
    print(f"  Stage: {args.stage}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}

    if args.stage in ("all", "sft"):
        results["sft"] = stage_sft()

    if args.stage in ("all", "grpo", "grpo_qwen3"):
        results["grpo_qwen3"] = stage_grpo_qwen3()

    if args.stage in ("all", "grpo", "grpo_multi"):
        results["grpo_multi"] = stage_grpo_multidomain()

    if args.stage in ("all", "eval"):
        results["eval"] = stage_eval()

    # Summary
    print("\n" + "=" * 70)
    print("  P2 Pipeline Summary")
    print("=" * 70)
    for stage, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {stage}")
    print("=" * 70)


if __name__ == "__main__":
    main()
