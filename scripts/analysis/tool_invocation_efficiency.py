#!/usr/bin/env python3
"""Tool Invocation Efficiency (TIE) metric analyzer for BIOAgents training rollout logs.

Parses multi-turn RL rollout logs to compute:
  TIE = correct_outcomes / (total_tool_calls * avg_turns)  per domain

Also analyzes tool usage patterns, plan-execute-verify behavior, and tool diversity.

Usage:
    python tool_invocation_efficiency.py [--log LOG_PATH] [--output OUTPUT_PATH]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOG = (
    "/data/project/private/minstar/workspace/BIOAgents/logs/training_fast_vllm_v3.log"
)
DEFAULT_OUTPUT = (
    "/data/project/private/minstar/workspace/BIOAgents/logs/analysis/tie_metrics.json"
)

# A rollout is "completed" when its final tool call is one of these.
TERMINAL_TOOLS = frozenset({"FINAL", "submit_answer"})

# Tool categories for plan-execute-verify analysis
PLANNING_TOOLS = frozenset({"think", "analyze_answer_options"})
EXECUTION_TOOLS = frozenset({
    "search", "search_evidence", "search_guidelines",
    "browse", "browse_wiki_entry", "browse_article",
    "get_drug_info", "analyze_medical_image", "get_image_report",
    "calculate_image_metrics",
})
VERIFICATION_TOOLS = frozenset({"submit_answer", "FINAL"})

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    """A single turn within a rollout."""
    turn_num: int
    max_turns: int
    input_tokens: int
    gen_tokens: int
    time_s: float
    tool: str


@dataclass
class RolloutRecord:
    """One rollout attempt (1..G) for a given task."""
    task_id: str
    domain: str
    rollout_idx: int
    max_rollouts: int
    turns: List[TurnRecord] = field(default_factory=list)
    resp_len: Optional[int] = None
    completed: bool = False


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# [TASK_ID] turn X/10: in=N gen=M t=Ts tool=TOOL_NAME
RE_TURN = re.compile(
    r"_run_single_rollout:2039 - "
    r"\[(?P<task_id>[^\]]+)\] "
    r"turn (?P<turn>\d+)/(?P<max_turn>\d+): "
    r"in=(?P<in_tok>\d+) gen=(?P<gen_tok>\d+) "
    r"t=(?P<time>[\d.]+)s "
    r"tool=(?P<tool>\S+)"
)

# [GPUX] Task TASK_ID rollout Y/4 starting (domain=DOMAIN)
RE_START = re.compile(
    r"_process_single_task:1494 - "
    r"\[GPU\d+\] Task (?P<task_id>\S+) "
    r"rollout (?P<r_idx>\d+)/(?P<r_max>\d+) "
    r"starting \(domain=(?P<domain>\S+)\)"
)

# [GPUX] Task TASK_ID rollout Y/4 done — resp_len=N
RE_DONE = re.compile(
    r"_process_single_task:1527 - "
    r"\[GPU\d+\] Task (?P<task_id>\S+) "
    r"rollout (?P<r_idx>\d+)/(?P<r_max>\d+) "
    r"done — resp_len=(?P<resp_len>\d+)"
)

# Task X/3338 [batch=8x]: mean_R=R, best_R=B, worst_R=W
RE_BATCH = re.compile(
    r"train_multiturn:1699 - \s*"
    r"Task (?P<task_num>\d+)/(?P<total>\d+) "
    r"\[batch=(?P<bs>\d+)x\]: "
    r"mean_R=(?P<mean_r>[\d.]+), "
    r"best_R=(?P<best_r>[\d.]+), "
    r"worst_R=(?P<worst_r>[\d.]+)"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_log(log_path: str):
    """Parse the training log and return structured rollout data plus batch rewards."""

    # Active rollouts keyed by (task_id, rollout_idx)
    active = {}  # type: Dict[Tuple[str, int], RolloutRecord]
    # Completed rollouts
    completed_rollouts = []  # type: List[RolloutRecord]
    # Batch reward entries
    batch_rewards = []  # type: List[dict]

    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            # --- rollout start ---
            m = RE_START.search(line)
            if m:
                tid = m.group("task_id")
                r_idx = int(m.group("r_idx"))
                r_max = int(m.group("r_max"))
                domain = m.group("domain")
                key = (tid, r_idx)
                active[key] = RolloutRecord(
                    task_id=tid,
                    domain=domain,
                    rollout_idx=r_idx,
                    max_rollouts=r_max,
                )
                continue

            # --- turn entry ---
            m = RE_TURN.search(line)
            if m:
                tid = m.group("task_id")
                tool = m.group("tool")
                turn = TurnRecord(
                    turn_num=int(m.group("turn")),
                    max_turns=int(m.group("max_turn")),
                    input_tokens=int(m.group("in_tok")),
                    gen_tokens=int(m.group("gen_tok")),
                    time_s=float(m.group("time")),
                    tool=tool,
                )

                # Find the matching active rollout — try each rollout_idx
                matched_key = None
                for key, rec in active.items():
                    if key[0] == tid:
                        matched_key = key
                        break
                # If multiple rollouts of same task_id are active (shouldn't
                # happen normally), we pick the first. If none found, we create
                # a placeholder.
                if matched_key is None:
                    matched_key = (tid, 0)
                    active[matched_key] = RolloutRecord(
                        task_id=tid,
                        domain="unknown",
                        rollout_idx=0,
                        max_rollouts=4,
                    )
                active[matched_key].turns.append(turn)
                if tool in TERMINAL_TOOLS:
                    active[matched_key].completed = True
                continue

            # --- rollout done ---
            m = RE_DONE.search(line)
            if m:
                tid = m.group("task_id")
                r_idx = int(m.group("r_idx"))
                resp_len = int(m.group("resp_len"))
                key = (tid, r_idx)
                rec = active.pop(key, None)
                if rec is None:
                    # Possibly missed the start line — create minimal record
                    rec = RolloutRecord(
                        task_id=tid,
                        domain="unknown",
                        rollout_idx=r_idx,
                        max_rollouts=int(m.group("r_max")),
                    )
                rec.resp_len = resp_len
                # Mark completed if the last turn was a terminal tool
                if rec.turns and rec.turns[-1].tool in TERMINAL_TOOLS:
                    rec.completed = True
                completed_rollouts.append(rec)
                continue

            # --- batch reward ---
            m = RE_BATCH.search(line)
            if m:
                batch_rewards.append({
                    "task_num": int(m.group("task_num")),
                    "total_tasks": int(m.group("total")),
                    "batch_size": int(m.group("bs")),
                    "mean_R": float(m.group("mean_r")),
                    "best_R": float(m.group("best_r")),
                    "worst_R": float(m.group("worst_r")),
                })
                continue

    # Flush any still-active rollouts (run in progress)
    for rec in active.values():
        completed_rollouts.append(rec)

    return completed_rollouts, batch_rewards


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def compute_tie_metrics(rollouts: List[RolloutRecord]):
    """Compute TIE and related metrics per domain and overall."""

    domain_rollouts = defaultdict(list)  # type: Dict[str, List[RolloutRecord]]
    for r in rollouts:
        domain_rollouts[r.domain].append(r)

    domain_metrics = {}  # type: Dict[str, dict]

    for domain, recs in sorted(domain_rollouts.items()):
        n_rollouts = len(recs)
        n_completed = sum(1 for r in recs if r.completed)
        total_turns = sum(len(r.turns) for r in recs)
        total_tool_calls = total_turns  # each turn is one tool call
        total_gen_tokens = sum(t.gen_tokens for r in recs for t in r.turns)
        total_time = sum(t.time_s for r in recs for t in r.turns)
        avg_turns = total_turns / n_rollouts if n_rollouts else 0.0

        # TIE = completed_rollouts / (total_tool_calls * avg_turns)
        denominator = total_tool_calls * avg_turns
        tie = n_completed / denominator if denominator > 0 else 0.0

        # Tool frequency
        tool_counts = Counter()  # type: Counter
        for r in recs:
            for t in r.turns:
                tool_counts[t.tool] += 1

        # Unique tools
        unique_tools = set(tool_counts.keys())

        # Turn distribution
        turn_counts = [len(r.turns) for r in recs]
        avg_turn_count = sum(turn_counts) / len(turn_counts) if turn_counts else 0.0
        min_turns = min(turn_counts) if turn_counts else 0
        max_turns = max(turn_counts) if turn_counts else 0

        # Plan-Execute-Verify pattern analysis
        plan_count = sum(tool_counts.get(t, 0) for t in PLANNING_TOOLS)
        exec_count = sum(tool_counts.get(t, 0) for t in EXECUTION_TOOLS)
        verify_count = sum(tool_counts.get(t, 0) for t in VERIFICATION_TOOLS)

        # Per-rollout sequence pattern: does it follow plan -> execute -> verify?
        pev_conforming = 0
        for r in recs:
            if not r.turns:
                continue
            tools_seq = [t.tool for t in r.turns]
            # Check if any planning tool appears before execution/verify
            # and execution appears before verification
            first_plan = _first_index(tools_seq, PLANNING_TOOLS)
            first_exec = _first_index(tools_seq, EXECUTION_TOOLS)
            first_verify = _first_index(tools_seq, VERIFICATION_TOOLS)
            # Conforming: plan <= exec <= verify (any -1 means absent, skip)
            if first_verify >= 0:  # must have verification
                if first_exec >= 0 and first_exec < first_verify:
                    if first_plan < 0 or first_plan <= first_exec:
                        pev_conforming += 1

        domain_metrics[domain] = {
            "n_rollouts": n_rollouts,
            "n_completed": n_completed,
            "completion_rate": round(n_completed / n_rollouts, 4) if n_rollouts else 0,
            "total_tool_calls": total_tool_calls,
            "avg_turns": round(avg_turn_count, 2),
            "min_turns": min_turns,
            "max_turns": max_turns,
            "total_gen_tokens": total_gen_tokens,
            "total_time_s": round(total_time, 1),
            "tie": round(tie, 6),
            "tool_frequency": dict(tool_counts.most_common()),
            "tool_diversity": len(unique_tools),
            "plan_execute_verify": {
                "planning_calls": plan_count,
                "execution_calls": exec_count,
                "verification_calls": verify_count,
                "pev_conforming_rollouts": pev_conforming,
                "pev_conformance_rate": round(
                    pev_conforming / n_rollouts, 4
                ) if n_rollouts else 0,
            },
        }

    # Overall metrics
    all_rollouts = rollouts
    n_all = len(all_rollouts)
    n_all_completed = sum(1 for r in all_rollouts if r.completed)
    all_turns = sum(len(r.turns) for r in all_rollouts)
    avg_turns_all = all_turns / n_all if n_all else 0.0
    denom_all = all_turns * avg_turns_all
    tie_all = n_all_completed / denom_all if denom_all > 0 else 0.0

    all_tool_counts = Counter()  # type: Counter
    for r in all_rollouts:
        for t in r.turns:
            all_tool_counts[t.tool] += 1

    overall = {
        "n_rollouts": n_all,
        "n_completed": n_all_completed,
        "completion_rate": round(n_all_completed / n_all, 4) if n_all else 0,
        "total_tool_calls": all_turns,
        "avg_turns": round(avg_turns_all, 2),
        "tie": round(tie_all, 6),
        "tool_frequency": dict(all_tool_counts.most_common()),
        "tool_diversity": len(set(all_tool_counts.keys())),
        "n_domains": len(domain_metrics),
    }

    return {"overall": overall, "per_domain": domain_metrics}


def _first_index(seq: List[str], target_set: FrozenSet[str]) -> int:
    """Return the index of the first element in seq that is in target_set, or -1."""
    for i, item in enumerate(seq):
        if item in target_set:
            return i
    return -1


def analyze_batch_rewards(batch_rewards: List[dict]) -> dict:
    """Summarize batch reward progression."""
    if not batch_rewards:
        return {"n_batches": 0}

    mean_rs = [b["mean_R"] for b in batch_rewards]
    best_rs = [b["best_R"] for b in batch_rewards]
    worst_rs = [b["worst_R"] for b in batch_rewards]

    n = len(mean_rs)
    # Split into first half / second half to see trend
    mid = max(1, n // 2)
    first_half_mean = sum(mean_rs[:mid]) / mid
    second_half_mean = sum(mean_rs[mid:]) / max(1, n - mid)

    return {
        "n_batches": n,
        "mean_R_avg": round(sum(mean_rs) / n, 4),
        "mean_R_min": round(min(mean_rs), 4),
        "mean_R_max": round(max(mean_rs), 4),
        "best_R_avg": round(sum(best_rs) / n, 4),
        "worst_R_avg": round(sum(worst_rs) / n, 4),
        "first_half_mean_R": round(first_half_mean, 4),
        "second_half_mean_R": round(second_half_mean, 4),
        "reward_trend": "improving" if second_half_mean > first_half_mean + 0.01 else (
            "declining" if second_half_mean < first_half_mean - 0.01 else "stable"
        ),
        "latest_task_num": batch_rewards[-1]["task_num"],
        "total_tasks": batch_rewards[-1]["total_tasks"],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(tie_data: dict, reward_data: dict) -> None:
    """Print a human-readable summary table to stdout."""

    overall = tie_data["overall"]
    domains = tie_data["per_domain"]

    print("\n" + "=" * 90)
    print("  Tool Invocation Efficiency (TIE) Analysis")
    print("=" * 90)

    print(f"\n  Log status: {overall['n_rollouts']} rollouts parsed across "
          f"{overall['n_domains']} domains")
    print(f"  Completion rate: {overall['n_completed']}/{overall['n_rollouts']} "
          f"({overall['completion_rate']:.1%})")
    print(f"  Overall TIE: {overall['tie']:.6f}")
    print(f"  Avg turns/rollout: {overall['avg_turns']:.2f}")
    print(f"  Total tool calls: {overall['total_tool_calls']}")
    print(f"  Tool diversity: {overall['tool_diversity']} unique tools")

    # Per-domain table
    print("\n" + "-" * 90)
    header = (
        f"  {'Domain':<25s} {'Rollouts':>8s} {'Compl%':>7s} {'AvgTrn':>7s} "
        f"{'TIE':>10s} {'Tools':>6s} {'PEV%':>6s}"
    )
    print(header)
    print("-" * 90)

    for domain in sorted(domains.keys()):
        d = domains[domain]
        pev_rate = d["plan_execute_verify"]["pev_conformance_rate"]
        print(
            f"  {domain:<25s} {d['n_rollouts']:>8d} "
            f"{d['completion_rate']:>6.1%} {d['avg_turns']:>7.2f} "
            f"{d['tie']:>10.6f} {d['tool_diversity']:>6d} "
            f"{pev_rate:>5.1%}"
        )

    # Tool frequency (top 15)
    print("\n" + "-" * 90)
    print("  Top 15 Tool Calls (all domains)")
    print("-" * 90)
    tool_freq = overall["tool_frequency"]
    total_calls = sum(tool_freq.values())
    for i, (tool, count) in enumerate(
        sorted(tool_freq.items(), key=lambda x: -x[1])[:15]
    ):
        pct = count / total_calls * 100 if total_calls else 0
        bar = "#" * int(pct)
        print(f"  {tool:<30s} {count:>6d} ({pct:>5.1f}%) {bar}")

    # Reward summary
    if reward_data["n_batches"] > 0:
        print("\n" + "-" * 90)
        print("  Batch Reward Summary")
        print("-" * 90)
        print(f"  Batches processed: {reward_data['n_batches']}")
        print(f"  Mean reward (avg): {reward_data['mean_R_avg']:.4f}")
        print(f"  Mean reward range: [{reward_data['mean_R_min']:.4f}, "
              f"{reward_data['mean_R_max']:.4f}]")
        print(f"  First-half avg:    {reward_data['first_half_mean_R']:.4f}")
        print(f"  Second-half avg:   {reward_data['second_half_mean_R']:.4f}")
        print(f"  Trend:             {reward_data['reward_trend']}")
        print(f"  Progress:          task {reward_data['latest_task_num']}"
              f"/{reward_data['total_tasks']}")

    print("\n" + "=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Tool Invocation Efficiency from rollout logs."
    )
    parser.add_argument(
        "--log", default=DEFAULT_LOG, help="Path to training log file."
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Path to write JSON results."
    )
    args = parser.parse_args()

    log_path = args.log
    output_path = args.output

    if not os.path.isfile(log_path):
        print(f"ERROR: Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing: {log_path}")
    rollouts, batch_rewards = parse_log(log_path)

    if not rollouts:
        print("WARNING: No rollouts found in log. The run may not have started yet.",
              file=sys.stderr)
        # Write empty results
        result = {
            "status": "no_data",
            "log_path": log_path,
            "overall": {},
            "per_domain": {},
            "batch_rewards": {},
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Empty results saved to: {output_path}")
        return

    print(f"Found {len(rollouts)} rollouts, {len(batch_rewards)} batch reward entries")

    tie_data = compute_tie_metrics(rollouts)
    reward_data = analyze_batch_rewards(batch_rewards)

    # Print summary
    print_summary(tie_data, reward_data)

    # Save JSON
    result = {
        "status": "partial" if batch_rewards and batch_rewards[-1]["task_num"] < batch_rewards[-1]["total_tasks"] else "complete",
        "log_path": log_path,
        "overall": tie_data["overall"],
        "per_domain": tie_data["per_domain"],
        "batch_rewards": reward_data,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
