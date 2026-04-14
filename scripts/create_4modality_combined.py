#!/usr/bin/env python3
"""Create combined 4-modality dataset: TextQA + VQA + EHR/Agentic + all domains.

Merges multimodal_rl_combined (text_qa + VQA) with agentic_rl_combined
(clinical_dx, drug, EHR, triage, psych, OB) into a single training set.

This creates the dataset needed for RQ1 (new): "Can multi-modal agentic
GYM training improve ALL 4 modalities simultaneously?"
"""
import json
from pathlib import Path
import random

BASE = Path(__file__).parent.parent / "data" / "domains"
OUTPUT = BASE / "full_4modality_combined"

def main():
    # Load multimodal (TextQA + VQA)
    mm_tasks = json.load(open(BASE / "multimodal_rl_combined" / "tasks.json"))
    mm_splits = json.load(open(BASE / "multimodal_rl_combined" / "split_tasks.json"))

    # Load agentic (clinical_dx, drug, EHR, triage, psych, OB)
    ag_tasks = json.load(open(BASE / "agentic_rl_combined" / "tasks.json"))
    ag_splits = json.load(open(BASE / "agentic_rl_combined" / "split_tasks.json"))

    # Check for ID collisions
    mm_ids = {t["id"] for t in mm_tasks}
    ag_ids = {t["id"] for t in ag_tasks}
    overlap = mm_ids & ag_ids
    if overlap:
        print(f"WARNING: {len(overlap)} overlapping IDs, prefixing agentic tasks")
        for t in ag_tasks:
            if t["id"] in overlap:
                t["id"] = f"ag_{t['id']}"

    # Combine tasks
    combined = mm_tasks + ag_tasks

    # Combine splits
    combined_train = mm_splits.get("train", []) + ag_splits.get("train", [])
    combined_test = mm_splits.get("test", []) + ag_splits.get("test", [])

    # Fix any prefixed IDs in splits
    if overlap:
        ag_train_set = set(ag_splits.get("train", []))
        ag_test_set = set(ag_splits.get("test", []))
        combined_train = [
            f"ag_{x}" if x in overlap and x in ag_train_set else x
            for x in combined_train
        ]
        combined_test = [
            f"ag_{x}" if x in overlap and x in ag_test_set else x
            for x in combined_test
        ]

    combined_splits = {"train": combined_train, "test": combined_test}

    # Print stats
    src_counts = {}
    for t in combined:
        src = t.get("_source_domain", "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1

    print(f"Combined 4-modality dataset:")
    print(f"  Total tasks: {len(combined)}")
    print(f"  Train: {len(combined_train)}")
    print(f"  Test: {len(combined_test)}")
    print(f"  Source domains:")
    for k, v in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")

    # Save
    OUTPUT.mkdir(parents=True, exist_ok=True)
    json.dump(combined, open(OUTPUT / "tasks.json", "w"), indent=2)
    json.dump(combined_splits, open(OUTPUT / "split_tasks.json", "w"), indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
