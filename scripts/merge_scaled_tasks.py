#!/usr/bin/env python3
"""Merge scaled tasks into agentic_rl_combined and regenerate full pipeline.

Steps:
1. Load existing agentic_rl_combined tasks
2. Load scaled tasks from each domain
3. Assign _source_domain and correct_answer for reward compatibility
4. Merge (avoiding ID collisions)
5. Update split_tasks.json
6. Regenerate full_4modality_combined
"""
import json
import random
from pathlib import Path

random.seed(42)

BASE = Path(__file__).parent.parent / "data" / "domains"

# Domain mapping for _source_domain field
DOMAIN_MAP = {
    "clinical_diagnosis": "clinical_diagnosis",
    "drug_interaction": "drug_interaction",
    "ehr_management": "ehr_management",
    "triage_emergency": "triage_emergency",
    "visual_diagnosis": "multimodal_vqa",  # Map to existing VQA domain
    "radiology_report": "multimodal_vqa",  # Map to existing VQA domain
}


def load_scaled_tasks(domain: str) -> list[dict]:
    """Load scaled tasks and add required fields."""
    path = BASE / domain / "tasks_scaled.json"
    if not path.exists():
        return []

    tasks = json.load(open(path))
    split_path = BASE / domain / "split_tasks_scaled.json"
    splits = json.load(open(split_path)) if split_path.exists() else {"train": [], "test": []}

    for t in tasks:
        t["_source_domain"] = DOMAIN_MAP.get(domain, domain)
        # Ensure correct_answer exists for reward function
        if "correct_answer" not in t:
            t["correct_answer"] = ""  # Open-ended, keyword overlap scoring

    return tasks, splits


def main():
    # Load existing agentic_rl_combined
    ag_tasks = json.load(open(BASE / "agentic_rl_combined" / "tasks.json"))
    ag_splits = json.load(open(BASE / "agentic_rl_combined" / "split_tasks.json"))
    existing_ids = {t["id"] for t in ag_tasks}

    print(f"Existing agentic_rl_combined: {len(ag_tasks)} tasks")
    print(f"  Train: {len(ag_splits['train'])}, Test: {len(ag_splits['test'])}")

    # Load and merge scaled tasks
    domains_to_merge = [
        "clinical_diagnosis", "drug_interaction", "ehr_management",
        "triage_emergency", "psychiatry", "obstetrics",
    ]

    new_tasks = []
    new_train_ids = []
    new_test_ids = []

    for domain in domains_to_merge:
        tasks, splits = load_scaled_tasks(domain)
        if not tasks:
            continue

        # Prefix IDs to avoid collisions
        added = 0
        for t in tasks:
            if t["id"] in existing_ids:
                t["id"] = f"scaled_{t['id']}"
            if t["id"] in existing_ids:
                continue  # Skip if still colliding
            existing_ids.add(t["id"])
            new_tasks.append(t)
            added += 1

            # Assign to train/test based on scaled splits
            if t["id"].replace("scaled_", "") in splits.get("train", []) or t["id"] in splits.get("train", []):
                new_train_ids.append(t["id"])
            else:
                new_test_ids.append(t["id"])

        print(f"  + {domain}: {added} new tasks")

    # Merge
    merged_tasks = ag_tasks + new_tasks
    merged_splits = {
        "train": ag_splits["train"] + new_train_ids,
        "test": ag_splits["test"] + new_test_ids,
    }

    # Save updated agentic_rl_combined
    out_dir = BASE / "agentic_rl_combined"
    json.dump(merged_tasks, open(out_dir / "tasks.json", "w"), indent=2, ensure_ascii=False)
    json.dump(merged_splits, open(out_dir / "split_tasks.json", "w"), indent=2)

    # Print stats
    src_counts = {}
    for t in merged_tasks:
        src = t.get("_source_domain", "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1

    print(f"\nUpdated agentic_rl_combined: {len(merged_tasks)} tasks")
    print(f"  Train: {len(merged_splits['train'])}, Test: {len(merged_splits['test'])}")
    for k, v in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
