#!/usr/bin/env python3
"""Merge all SFT datasets into a single combined file for warmup training.

Combines:
  1. opensource_medical_sft.jsonl (20K samples from 4 HF datasets)
  2. tool_trajectories_train.jsonl (tool-use trajectories from MedQA)
  3. medical_sft_combined.jsonl (clinical task trajectories)

Output: datasets/sft/sft_all_merged.jsonl

Usage:
    python scripts/merge_sft_data.py [--max-opensource 5000] [--output datasets/sft/sft_all_merged.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = PROJECT_ROOT / "datasets" / "sft"


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_sample(sample: dict) -> bool:
    """Check if a sample has valid messages structure."""
    messages = sample.get("messages", [])
    if not messages or len(messages) < 2:
        return False
    # Must have at least user + assistant
    roles = [m.get("role") for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return False
    # Assistant response shouldn't be empty
    for m in messages:
        if m.get("role") == "assistant" and not m.get("content", "").strip():
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Merge all SFT datasets")
    parser.add_argument(
        "--max-opensource", type=int, default=None,
        help="Max samples from opensource dataset (None = use all)",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(SFT_DIR / "sft_all_merged.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_samples = []
    stats = {}

    # 1. Opensource medical SFT (primary warmup data)
    opensource_path = SFT_DIR / "opensource_medical_sft.jsonl"
    if opensource_path.exists():
        data = load_jsonl(opensource_path)
        valid = [s for s in data if validate_sample(s)]
        if args.max_opensource and len(valid) > args.max_opensource:
            valid = rng.sample(valid, args.max_opensource)
        all_samples.extend(valid)
        stats["opensource_medical_sft"] = len(valid)
        print(f"  opensource_medical_sft: {len(valid)} samples (from {len(data)} total)")
    else:
        print(f"  [SKIP] {opensource_path} not found")

    # 2. Tool trajectories (from MedQA train split)
    traj_path = SFT_DIR / "tool_trajectories_train.jsonl"
    if traj_path.exists():
        data = load_jsonl(traj_path)
        valid = [s for s in data if validate_sample(s)]
        all_samples.extend(valid)
        stats["tool_trajectories"] = len(valid)
        print(f"  tool_trajectories: {len(valid)} samples (from {len(data)} total)")
    else:
        print(f"  [SKIP] {traj_path} not found")

    # 3. Clinical task trajectories
    combined_path = SFT_DIR / "medical_sft_combined.jsonl"
    if combined_path.exists():
        data = load_jsonl(combined_path)
        valid = [s for s in data if validate_sample(s)]
        all_samples.extend(valid)
        stats["medical_sft_combined"] = len(valid)
        print(f"  medical_sft_combined: {len(valid)} samples (from {len(data)} total)")
    else:
        print(f"  [SKIP] {combined_path} not found")

    # 4. Multi-domain SFT
    for fname in ["multidomain_sft.jsonl", "p2_multidomain_sft.jsonl"]:
        fpath = SFT_DIR / fname
        if fpath.exists():
            data = load_jsonl(fpath)
            valid = [s for s in data if validate_sample(s)]
            all_samples.extend(valid)
            stats[fname.replace(".jsonl", "")] = len(valid)
            print(f"  {fname}: {len(valid)} samples (from {len(data)} total)")

    # Shuffle
    rng.shuffle(all_samples)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Merged SFT Dataset Summary")
    print(f"{'=' * 50}")
    print(f"  Total samples: {len(all_samples)}")
    for source, count in sorted(stats.items()):
        print(f"  - {source}: {count}")
    print(f"  Output: {output_path}")

    # Save stats
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump({"total": len(all_samples), "per_source": stats}, f, indent=2)
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
