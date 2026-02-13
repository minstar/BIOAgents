#!/usr/bin/env python3
"""Aggregate ALL baseline results into a comprehensive P0-2 report.

Reads per-domain metrics from logs/baseline/ and produces:
1. Complete 3-model × 5-domain comparison matrix
2. Per-task breakdowns
3. Key findings and training priority analysis
4. Final JSON report

Usage:
    python scripts/aggregate_baseline_report.py
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

MODELS = ["Qwen3-8B-Base", "Qwen2.5-VL-7B-Instruct", "Lingshu-7B"]
DOMAINS = ["clinical_diagnosis", "medical_qa", "visual_diagnosis", "drug_interaction", "ehr_management"]
DOMAIN_SHORT = {
    "clinical_diagnosis": "ClinDx",
    "medical_qa": "MedQA",
    "visual_diagnosis": "VisDx",
    "drug_interaction": "DrugInt",
    "ehr_management": "EHR",
}

# ── Collect all baseline results ─────────────────────────────────────────────

def find_latest_run(model: str, domain: str) -> dict | None:
    """Find the latest domain_metrics.json for a model+domain pair."""
    baseline_dir = PROJECT_ROOT / "logs" / "baseline"
    pattern = f"{model}_{domain}_*"
    
    candidates = sorted(baseline_dir.glob(pattern), reverse=True)  # newest first
    for cand in candidates:
        metrics_file = cand / "domain_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                return json.load(f)
    return None


def collect_all_results() -> dict:
    """Collect all model × domain results into a matrix."""
    matrix = {}
    for model in MODELS:
        matrix[model] = {}
        for domain in DOMAINS:
            result = find_latest_run(model, domain)
            matrix[model][domain] = result
    return matrix


# ── Report generation ────────────────────────────────────────────────────────

def print_header():
    """Print the report header."""
    w = 120
    print("\n" + "=" * w)
    print("  BIOAgents P0-2: Multi-Domain Baseline Evaluation Report")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Models: {len(MODELS)} | Domains: {len(DOMAINS)} | Total Runs: {len(MODELS) * len(DOMAINS)}")
    print("=" * w)


def print_matrix_table(matrix: dict, metric: str, label: str, fmt: str = ".3f"):
    """Print a model × domain matrix for a given metric."""
    w = 120
    print(f"\n  {label}")
    print("-" * w)
    
    header = f"  {'Model':<28}"
    for d in DOMAINS:
        header += f"  {DOMAIN_SHORT[d]:>10}"
    header += f"  {'OVERALL':>10}  {'RANK':>5}"
    print(header)
    print("-" * w)
    
    model_avgs = []
    for model in MODELS:
        row = f"  {model:<28}"
        scores = []
        for domain in DOMAINS:
            m = matrix[model][domain]
            if m and metric in m:
                val = m[metric]
                row += f"  {val:>{10}{fmt}}"
                scores.append(val)
            elif m and "avg_" + metric in m:
                val = m["avg_" + metric]
                row += f"  {val:>{10}{fmt}}"
                scores.append(val)
            else:
                row += f"  {'—':>10}"
        
        avg = sum(scores) / len(scores) if scores else 0
        model_avgs.append((model, avg))
        row += f"  {avg:>{10}{fmt}}"
        print(row)
    
    # Print ranks
    model_avgs.sort(key=lambda x: x[1], reverse=True)
    rank_map = {m: i + 1 for i, (m, _) in enumerate(model_avgs)}
    
    # Reprint with ranks
    print("-" * w)
    
    return rank_map


def print_action_score_table(matrix: dict):
    """Print action score comparison."""
    w = 120
    print(f"\n  📊 Action Score (tool-use accuracy)")
    print("-" * w)
    
    header = f"  {'Model':<28}"
    for d in DOMAINS:
        header += f"  {DOMAIN_SHORT[d]:>10}"
    header += f"  {'OVERALL':>10}"
    print(header)
    print("-" * w)
    
    rankings = []
    for model in MODELS:
        row = f"  {model:<28}"
        scores = []
        for domain in DOMAINS:
            m = matrix[model][domain]
            if m:
                val = m.get("avg_action_score", 0)
                # Highlight best in domain
                row += f"  {val:>10.3f}"
                scores.append(val)
            else:
                row += f"  {'—':>10}"
        
        avg = sum(scores) / len(scores) if scores else 0
        rankings.append((model, avg, scores))
        row += f"  {avg:>10.3f}"
        print(row)
    
    print("-" * w)
    
    # Best per domain
    best_row = f"  {'Best →':>28}"
    for di, domain in enumerate(DOMAINS):
        best_model = max(MODELS, key=lambda m: (matrix[m][domain] or {}).get("avg_action_score", 0))
        best_val = (matrix[best_model][domain] or {}).get("avg_action_score", 0)
        best_row += f"  {best_model.split('-')[0][:10]:>10}"
    best_avg_model = max(rankings, key=lambda x: x[1])
    best_row += f"  {best_avg_model[0].split('-')[0][:10]:>10}"
    print(best_row)
    print("-" * w)
    
    return rankings


def print_reward_table(matrix: dict):
    """Print composite reward comparison."""
    w = 120
    print(f"\n  🎯 Composite Reward (accuracy 0.4 + format 0.2 + process 0.4)")
    print("-" * w)
    
    header = f"  {'Model':<28}"
    for d in DOMAINS:
        header += f"  {DOMAIN_SHORT[d]:>10}"
    header += f"  {'OVERALL':>10}"
    print(header)
    print("-" * w)
    
    rankings = []
    for model in MODELS:
        row = f"  {model:<28}"
        scores = []
        for domain in DOMAINS:
            m = matrix[model][domain]
            if m:
                val = m.get("avg_reward", 0)
                row += f"  {val:>10.3f}"
                scores.append(val)
            else:
                row += f"  {'—':>10}"
        
        avg = sum(scores) / len(scores) if scores else 0
        rankings.append((model, avg))
        row += f"  {avg:>10.3f}"
        print(row)
    
    print("-" * w)
    return rankings


def print_turns_table(matrix: dict):
    """Print average turns comparison."""
    w = 120
    print(f"\n  🔄 Avg Turns per Task")
    print("-" * w)
    
    header = f"  {'Model':<28}"
    for d in DOMAINS:
        header += f"  {DOMAIN_SHORT[d]:>10}"
    header += f"  {'OVERALL':>10}"
    print(header)
    print("-" * w)
    
    for model in MODELS:
        row = f"  {model:<28}"
        turn_vals = []
        for domain in DOMAINS:
            m = matrix[model][domain]
            if m:
                val = m.get("avg_turns", 0)
                row += f"  {val:>10.1f}"
                turn_vals.append(val)
            else:
                row += f"  {'—':>10}"
        
        avg = sum(turn_vals) / len(turn_vals) if turn_vals else 0
        row += f"  {avg:>10.1f}"
        print(row)
    
    print("-" * w)


def print_latency_table(matrix: dict):
    """Print total latency comparison."""
    w = 120
    print(f"\n  ⏱️  Total Latency per Domain (seconds)")
    print("-" * w)
    
    header = f"  {'Model':<28}"
    for d in DOMAINS:
        header += f"  {DOMAIN_SHORT[d]:>10}"
    header += f"  {'TOTAL':>10}"
    print(header)
    print("-" * w)
    
    for model in MODELS:
        row = f"  {model:<28}"
        total = 0
        for domain in DOMAINS:
            m = matrix[model][domain]
            if m:
                val = m.get("total_latency_s", 0)
                row += f"  {val:>10.1f}"
                total += val
            else:
                row += f"  {'—':>10}"
        row += f"  {total:>10.1f}"
        print(row)
    
    print("-" * w)


def print_qa_accuracy(matrix: dict):
    """Print QA accuracy for medical_qa domain."""
    w = 80
    print(f"\n  📝 Medical QA Accuracy (exact match)")
    print("-" * w)
    print(f"  {'Model':<28}  {'QA Acc':>10}  {'Answered':>10}  {'Correct':>10}")
    print("-" * w)
    
    for model in MODELS:
        m = matrix[model].get("medical_qa")
        if m and "qa_accuracy" in m:
            qa_acc = m["qa_accuracy"]
            qa_ans = m.get("qa_answered", "—")
            qa_cor = m.get("qa_correct", "—")
            print(f"  {model:<28}  {qa_acc:>10.3f}  {qa_ans:>10}  {qa_cor:>10}")
    
    print("-" * w)


def print_completion_matrix(matrix: dict):
    """Print completion status matrix."""
    w = 120
    print(f"\n  ✅ Completion Matrix (completed/total tasks)")
    print("-" * w)
    
    header = f"  {'Model':<28}"
    for d in DOMAINS:
        header += f"  {DOMAIN_SHORT[d]:>10}"
    header += f"  {'TOTAL':>10}"
    print(header)
    print("-" * w)
    
    for model in MODELS:
        row = f"  {model:<28}"
        total_completed = 0
        total_tasks = 0
        for domain in DOMAINS:
            m = matrix[model][domain]
            if m:
                c = m.get("num_completed", 0)
                t = m.get("num_tasks", 0)
                e = m.get("num_errors", 0)
                total_completed += c
                total_tasks += t
                status = f"{c}/{t}" + (f"({e}E)" if e > 0 else "")
                row += f"  {status:>10}"
            else:
                row += f"  {'MISSING':>10}"
        row += f"  {total_completed}/{total_tasks}"
        print(row)
    
    print("-" * w)


def print_per_task_breakdown(matrix: dict):
    """Print per-task breakdown for each model-domain."""
    w = 120
    print(f"\n  📋 Per-Task Breakdown (Action Score)")
    
    for domain in DOMAINS:
        print(f"\n  --- {DOMAIN_SHORT[domain]} ({domain}) ---")
        
        # Collect all task IDs
        task_ids = set()
        for model in MODELS:
            m = matrix[model][domain]
            if m and "per_task" in m:
                task_ids.update(m["per_task"].keys())
        
        if not task_ids:
            print("  No task data available")
            continue
        
        task_ids = sorted(task_ids)
        
        # Header
        header = f"  {'Task ID':<32}"
        for model in MODELS:
            short = model.split("-")[0][:12]
            header += f"  {short:>12}"
        print(header)
        print("  " + "-" * (32 + len(MODELS) * 14))
        
        for tid in task_ids:
            row = f"  {tid:<32}"
            for model in MODELS:
                m = matrix[model][domain]
                if m and "per_task" in m and tid in m["per_task"]:
                    task_data = m["per_task"][tid]
                    score = task_data.get("action_score", 0)
                    turns = task_data.get("turns", 0)
                    row += f"  {score:>6.3f}/{turns:<4}"
                else:
                    row += f"  {'—':>12}"
            print(row)


def compute_key_findings(matrix: dict, action_rankings: list, reward_rankings: list) -> list[str]:
    """Generate key findings from the results."""
    findings = []
    
    # 1. Overall ranking
    action_rankings.sort(key=lambda x: x[1], reverse=True)
    best_action = action_rankings[0]
    findings.append(
        f"🥇 Overall Action Score Ranking: "
        f"{action_rankings[0][0]} ({action_rankings[0][1]:.3f}) > "
        f"{action_rankings[1][0]} ({action_rankings[1][1]:.3f}) > "
        f"{action_rankings[2][0]} ({action_rankings[2][1]:.3f})"
    )
    
    reward_rankings.sort(key=lambda x: x[1], reverse=True)
    findings.append(
        f"🥇 Overall Reward Ranking: "
        f"{reward_rankings[0][0]} ({reward_rankings[0][1]:.3f}) > "
        f"{reward_rankings[1][0]} ({reward_rankings[1][1]:.3f}) > "
        f"{reward_rankings[2][0]} ({reward_rankings[2][1]:.3f})"
    )
    
    # 2. Best per domain
    for domain in DOMAINS:
        best_model = max(MODELS, key=lambda m: (matrix[m][domain] or {}).get("avg_action_score", 0))
        best_score = (matrix[best_model][domain] or {}).get("avg_action_score", 0)
        findings.append(f"  {DOMAIN_SHORT[domain]}: {best_model} ({best_score:.3f})")
    
    # 3. QA accuracy comparison
    qa_scores = {}
    for model in MODELS:
        m = matrix[model].get("medical_qa", {})
        if m:
            qa_scores[model] = m.get("qa_accuracy", 0)
    if qa_scores:
        best_qa = max(qa_scores.items(), key=lambda x: x[1])
        findings.append(f"📝 Best QA Accuracy: {best_qa[0]} ({best_qa[1]:.1%})")
    
    # 4. Tool-use efficiency
    for model in MODELS:
        avg_turns = []
        for domain in DOMAINS:
            m = matrix[model][domain]
            if m:
                avg_turns.append(m.get("avg_turns", 0))
        if avg_turns:
            findings.append(f"🔄 {model}: avg {sum(avg_turns)/len(avg_turns):.1f} turns/task")
    
    # 5. Training priorities
    findings.append("")
    findings.append("🎯 TRAINING PRIORITIES:")
    
    # Identify weakest areas per model
    for model in MODELS:
        scores = []
        for domain in DOMAINS:
            m = matrix[model][domain]
            if m:
                scores.append((domain, m.get("avg_action_score", 0)))
        scores.sort(key=lambda x: x[1])
        if scores:
            weakest = scores[0]
            findings.append(
                f"  {model}: Weakest in {DOMAIN_SHORT[weakest[0]]} ({weakest[1]:.3f}) "
                f"→ Focus SFT/GRPO on {DOMAIN_SHORT[weakest[0]]}"
            )
    
    return findings


def save_final_report(matrix: dict, action_rankings: list, reward_rankings: list):
    """Save the comprehensive JSON report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        "experiment": "BIOAgents P0-2: Multi-Domain Baseline Evaluation",
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "models": MODELS,
            "domains": DOMAINS,
            "total_runs": sum(
                1 for m in MODELS for d in DOMAINS if matrix[m][d] is not None
            ),
            "total_tasks_evaluated": sum(
                (matrix[m][d] or {}).get("num_tasks", 0)
                for m in MODELS for d in DOMAINS
            ),
        },
        "action_score_ranking": [
            {"model": m, "avg_action_score": round(s, 4)} for m, s, *_ in action_rankings
        ],
        "reward_ranking": [
            {"model": m, "avg_reward": round(s, 4)} for m, s in reward_rankings
        ],
        "results_matrix": {},
    }
    
    for model in MODELS:
        report["results_matrix"][model] = {}
        for domain in DOMAINS:
            m = matrix[model][domain]
            if m:
                report["results_matrix"][model][domain] = {
                    "action_score": m.get("avg_action_score"),
                    "reward": m.get("avg_reward"),
                    "turns": m.get("avg_turns"),
                    "tasks": m.get("num_tasks"),
                    "completed": m.get("num_completed"),
                    "errors": m.get("num_errors"),
                    "latency_s": m.get("total_latency_s"),
                }
                if "qa_accuracy" in m:
                    report["results_matrix"][model][domain]["qa_accuracy"] = m["qa_accuracy"]
    
    # Save
    out_dir = PROJECT_ROOT / "logs" / "baseline"
    out_path = out_dir / f"p02_baseline_report_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  📁 Final P0-2 report saved: {out_path}")
    
    return out_path


def main():
    # Collect
    print("\n  Collecting baseline results...")
    matrix = collect_all_results()
    
    # Verify completeness
    missing = []
    for model in MODELS:
        for domain in DOMAINS:
            if matrix[model][domain] is None:
                missing.append(f"{model} × {domain}")
    
    if missing:
        print(f"\n  ⚠️  Missing {len(missing)} runs:")
        for m in missing:
            print(f"    - {m}")
        print("  Run: python scripts/run_multidomain_baseline.py to fill gaps")
    else:
        print(f"  ✅ All {len(MODELS) * len(DOMAINS)} runs found!")
    
    # Print
    print_header()
    action_rankings = print_action_score_table(matrix)
    reward_rankings = print_reward_table(matrix)
    print_turns_table(matrix)
    print_latency_table(matrix)
    print_qa_accuracy(matrix)
    print_per_task_breakdown(matrix)
    
    # Key findings
    w = 120
    print("\n" + "=" * w)
    print("  KEY FINDINGS")
    print("=" * w)
    findings = compute_key_findings(matrix, action_rankings, reward_rankings)
    for f in findings:
        print(f"  {f}")
    print("=" * w)
    
    # Save
    save_final_report(matrix, action_rankings, reward_rankings)
    
    print(f"\n  ✅ P0-2 Baseline Evaluation Report Complete!")
    print(f"  Total: {len(MODELS)} models × {len(DOMAINS)} domains = {len(MODELS) * len(DOMAINS)} evaluations")


if __name__ == "__main__":
    main()
