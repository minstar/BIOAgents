#!/usr/bin/env python3
"""Generate multimodal RL training data by combining text QA + VQA tasks.

Creates:
  data/domains/multimodal_rl_combined/tasks.json
  data/domains/multimodal_rl_combined/split_tasks.json

Combines:
  - Existing text QA tasks (from medical_qa_1000)
  - VQA-RAD training samples (500)
  - SLAKE training samples (500 from metadata, 9835 from full)
  - PathVQA training samples (300)

Usage:
    python scripts/generate_multimodal_rl_data.py
    python scripts/generate_multimodal_rl_data.py --use-full-slake  # Include 9835 SLAKE samples
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "domains" / "multimodal_rl_combined"

# Data sources
MEDICAL_QA_PATH = PROJECT_ROOT / "data" / "domains" / "medical_qa_1000" / "tasks.json"
VQA_RAD_PATH = PROJECT_ROOT / "datasets" / "medical_images" / "vqa_rad" / "metadata.json"
SLAKE_PATH = PROJECT_ROOT / "datasets" / "medical_images" / "slake" / "metadata.json"
SLAKE_FULL_PATH = PROJECT_ROOT / "evaluations" / "PMC-VQA" / "Slake1.0" / "train.json"
PATHVQA_PATH = PROJECT_ROOT / "datasets" / "medical_images" / "pathvqa" / "metadata.json"
SLAKE_IMAGES_DIR = PROJECT_ROOT / "datasets" / "medical_images" / "slake" / "images"


def vqa_to_gym_task(sample: dict, source: str) -> dict:
    """Convert a VQA sample to GYM task format."""
    question = sample.get("question", "")
    answer = sample.get("answer", "")
    image_path = sample.get("image_path")
    modality = sample.get("modality", "unknown")
    body_part = sample.get("body_part", "unknown")

    task_id = f"vqa_{source}_{hashlib.md5(question.encode()).hexdigest()[:12]}"

    # Build the ticket (prompt)
    if image_path:
        ticket = f"IMAGE: {image_path}\nQUESTION: {question}"
    else:
        ticket = f"[Visual question - {modality} / {body_part}]\nQUESTION: {question}"

    return {
        "id": task_id,
        "description": {
            "purpose": f"Answer a medical visual question from {source}",
            "difficulty": "medium",
            "source": source,
            "category": f"vqa_{modality}",
            "generated_from": "vqa_converter",
        },
        "ticket": ticket,
        "correct_answer": answer,
        "evaluation_criteria": {
            "accuracy": "Answer must match the ground truth",
            "format": "Use <answer>...</answer> tags",
        },
        "options": None,
        "raw_answer": answer,
        "_source_domain": "multimodal_vqa",
        "_image_path": image_path,
        "_modality": modality,
    }


def convert_slake_full(sample: dict) -> dict:
    """Convert a full SLAKE training sample to VQA metadata format."""
    img_name = sample.get("img_name", "")
    # Try to find image in SLAKE images directory
    image_path = None
    if img_name:
        candidate = SLAKE_IMAGES_DIR / img_name
        if candidate.exists():
            image_path = str(candidate)

    return {
        "question": sample["question"],
        "answer": str(sample["answer"]),
        "image_path": image_path,
        "modality": sample.get("modality", "unknown"),
        "body_part": sample.get("location", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-full-slake", action="store_true",
                        help="Use full 9835 SLAKE train samples")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Fraction of tasks for test split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    tasks = []

    # 1. Load existing text QA tasks
    if MEDICAL_QA_PATH.exists():
        with open(MEDICAL_QA_PATH) as f:
            text_qa = json.load(f)
        # Mark source domain
        for t in text_qa:
            t["_source_domain"] = "text_qa"
        tasks.extend(text_qa)
        print(f"Text QA: {len(text_qa)} tasks")

    # 2. VQA-RAD
    if VQA_RAD_PATH.exists():
        with open(VQA_RAD_PATH) as f:
            vqa_rad = json.load(f)
        vqa_rad_tasks = [vqa_to_gym_task(s, "vqa_rad") for s in vqa_rad]
        tasks.extend(vqa_rad_tasks)
        print(f"VQA-RAD: {len(vqa_rad_tasks)} tasks")

    # 3. SLAKE
    if args.use_full_slake and SLAKE_FULL_PATH.exists():
        with open(SLAKE_FULL_PATH) as f:
            slake_full = json.load(f)
        # Filter to English only
        slake_en = [s for s in slake_full if s.get("q_lang", "en") == "en"]
        slake_converted = [convert_slake_full(s) for s in slake_en]
        slake_tasks = [vqa_to_gym_task(s, "slake") for s in slake_converted]
        tasks.extend(slake_tasks)
        print(f"SLAKE (full, English): {len(slake_tasks)} tasks")
    elif SLAKE_PATH.exists():
        with open(SLAKE_PATH) as f:
            slake = json.load(f)
        slake_tasks = [vqa_to_gym_task(s, "slake") for s in slake]
        tasks.extend(slake_tasks)
        print(f"SLAKE: {len(slake_tasks)} tasks")

    # 4. PathVQA
    if PATHVQA_PATH.exists():
        with open(PATHVQA_PATH) as f:
            pathvqa = json.load(f)
        pathvqa_tasks = [vqa_to_gym_task(s, "pathvqa") for s in pathvqa]
        tasks.extend(pathvqa_tasks)
        print(f"PathVQA: {len(pathvqa_tasks)} tasks")

    print(f"\nTotal tasks: {len(tasks)}")

    # Shuffle
    random.shuffle(tasks)

    # Split
    n_test = int(len(tasks) * args.test_ratio)
    n_train = len(tasks) - n_test

    train_ids = [t["id"] for t in tasks[:n_train]]
    test_ids = [t["id"] for t in tasks[n_train:]]

    split_tasks = {"train": train_ids, "test": test_ids}

    # Count by source
    domain_counts = {}
    for t in tasks:
        d = t.get("_source_domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    print(f"Domain breakdown: {domain_counts}")
    print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)

    with open(OUTPUT_DIR / "split_tasks.json", "w") as f:
        json.dump(split_tasks, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
