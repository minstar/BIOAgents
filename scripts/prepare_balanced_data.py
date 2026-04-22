#!/usr/bin/env python3
"""Create domain-balanced training data for v17.

Strategies:
  --strategy upsample_max  : Upsample all domains to match largest (892). Total ~7136.
  --strategy equal_400     : Cap large at 400, upsample small to 400. Total 3200.
  --strategy equal_300     : Cap large at 300, upsample small to 300. Total 2400.

Usage:
    python scripts/prepare_balanced_data.py --strategy equal_400
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def get_domain(extra_info):
    if isinstance(extra_info, str):
        return json.loads(extra_info).get("domain", "unknown")
    elif isinstance(extra_info, dict):
        return extra_info.get("domain", "unknown")
    return "unknown"


def balance_dataset(df: pd.DataFrame, target_per_domain: int) -> pd.DataFrame:
    """Balance dataset by up/downsampling each domain to target count."""
    df = df.copy()
    df["_domain"] = df["extra_info"].apply(get_domain)

    balanced_parts = []
    for domain in sorted(df["_domain"].unique()):
        domain_df = df[df["_domain"] == domain]
        n = len(domain_df)

        if n >= target_per_domain:
            # Downsample: take random subset
            sampled = domain_df.sample(n=target_per_domain, random_state=42)
        else:
            # Upsample: repeat + sample remainder
            repeats = target_per_domain // n
            remainder = target_per_domain % n
            parts = [domain_df] * repeats
            if remainder > 0:
                parts.append(domain_df.sample(n=remainder, random_state=42))
            sampled = pd.concat(parts, ignore_index=True)

        print(f"  {domain}: {n} -> {len(sampled)}")
        balanced_parts.append(sampled)

    result = pd.concat(balanced_parts, ignore_index=True)
    result = result.drop(columns=["_domain"])
    # Shuffle thoroughly
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["upsample_max", "equal_400", "equal_300"],
                        default="equal_400")
    parser.add_argument("--input", default="data/verl_parquet/full_4modality_vlm/train.parquet")
    parser.add_argument("--output-dir", default="data/verl_parquet/full_4modality_vlm_balanced")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    output_dir = project_root / args.output_dir

    print(f"Loading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Original: {len(df)} samples\n")

    # Domain distribution
    df_tmp = df.copy()
    df_tmp["_domain"] = df_tmp["extra_info"].apply(get_domain)
    print("Original distribution:")
    for domain, count in df_tmp["_domain"].value_counts().items():
        print(f"  {domain}: {count}")
    print()

    # Choose target
    if args.strategy == "upsample_max":
        target = df_tmp["_domain"].value_counts().max()
    elif args.strategy == "equal_400":
        target = 400
    elif args.strategy == "equal_300":
        target = 300

    print(f"Strategy: {args.strategy} (target={target}/domain)")
    balanced = balance_dataset(df, target)
    print(f"\nBalanced total: {len(balanced)} samples")
    print(f"Steps per epoch (batch=8): {len(balanced) // 8}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    balanced.to_parquet(train_path, index=False)
    print(f"Saved: {train_path}")

    # Copy val data as-is
    val_src = input_path.parent / "test.parquet"
    if val_src.exists():
        val_dst = output_dir / "test.parquet"
        import shutil
        shutil.copy2(val_src, val_dst)
        print(f"Copied val: {val_dst}")


if __name__ == "__main__":
    main()
