#!/usr/bin/env python3
"""Merge balanced tasks into agentic_rl_combined.

Reads tasks_balanced.json + split_tasks_balanced.json from each domain,
adds _source_domain and correct_answer fields, avoids ID collisions,
and appends to existing agentic_rl_combined.
"""
import json
from pathlib import Path

BASE = Path(__file__).parent.parent / "data" / "domains"

DOMAINS = [
    "drug_interaction",
    "triage_emergency",
    "ehr_management",
    "psychiatry",
    "obstetrics",
]


def main():
    ag_path = BASE / "agentic_rl_combined"
    ag_tasks = json.load(open(ag_path / "tasks.json"))
    ag_splits = json.load(open(ag_path / "split_tasks.json"))
    existing_ids = {t["id"] for t in ag_tasks}

    print(f"Before merge: {len(ag_tasks)} tasks")
    print(f"  Train: {len(ag_splits['train'])}, Test: {len(ag_splits['test'])}")

    new_tasks = []
    new_train = []
    new_test = []

    for domain in DOMAINS:
        task_file = BASE / domain / "tasks_balanced.json"
        split_file = BASE / domain / "split_tasks_balanced.json"
        if not task_file.exists():
            print(f"  SKIP {domain}: no tasks_balanced.json")
            continue

        tasks = json.load(open(task_file))
        splits = json.load(open(split_file)) if split_file.exists() else {"train": [], "test": []}
        train_set = set(splits.get("train", []))

        added = 0
        for t in tasks:
            t["_source_domain"] = domain
            if "correct_answer" not in t:
                t["correct_answer"] = ""

            # Avoid ID collisions
            if t["id"] in existing_ids:
                t["id"] = f"bal_{t['id']}"
            if t["id"] in existing_ids:
                continue
            existing_ids.add(t["id"])
            new_tasks.append(t)
            added += 1

            orig_id = t["id"].replace("bal_", "")
            if orig_id in train_set or t["id"] in train_set:
                new_train.append(t["id"])
            else:
                new_test.append(t["id"])

        print(f"  + {domain}: {added} tasks")

    merged_tasks = ag_tasks + new_tasks
    merged_splits = {
        "train": ag_splits["train"] + new_train,
        "test": ag_splits["test"] + new_test,
    }

    json.dump(merged_tasks, open(ag_path / "tasks.json", "w"), indent=2, ensure_ascii=False)
    json.dump(merged_splits, open(ag_path / "split_tasks.json", "w"), indent=2)

    # Domain stats
    src_counts = {}
    for t in merged_tasks:
        src = t.get("_source_domain", "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1

    print(f"\nAfter merge: {len(merged_tasks)} tasks")
    print(f"  Train: {len(merged_splits['train'])}, Test: {len(merged_splits['test'])}")
    for k, v in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
