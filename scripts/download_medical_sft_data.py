#!/usr/bin/env python3
from __future__ import annotations

"""Download and prepare contamination-free medical SFT datasets.

Downloads open-source medical instruction datasets from HuggingFace,
filters out any potential test set contamination, and converts to
unified SFT format for warmup training.

Datasets (Tier 1 - Contamination-free):
1. FreedomIntelligence/medical-o1-reasoning-SFT (~40K, EN+ZH)
2. bio-nlp-umass/bioinstruct (~25K, EN)
3. medalpaca/medical_meadow_wikidoc (~10K, EN)
4. medalpaca/medical_meadow_medical_flashcards (~33K, EN)

Usage:
    python scripts/download_medical_sft_data.py \
        --output datasets/sft/opensource_medical_sft.jsonl \
        --max-per-dataset 5000
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Known test set questions to filter (contamination detection)
def load_test_questions() -> set[str]:
    """Load test set question fingerprints for contamination check."""
    fingerprints = set()
    benchmark_dir = PROJECT_ROOT / "evaluations" / "self-biorag" / "data" / "benchmark"

    test_files = [
        "med_qa_test.jsonl",
        "mmlu_test.jsonl",
        "mmlu_anatomy_test.jsonl",
        "mmlu_clinical_knowledge_test.jsonl",
        "mmlu_college_biology_test.jsonl",
        "mmlu_college_medicine_test.jsonl",
        "mmlu_medical_genetics_test.jsonl",
        "mmlu_professional_medicine_test.jsonl",
    ]

    for fname in test_files:
        fpath = benchmark_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    q = record.get("question", record.get("input", ""))
                    if q:
                        # Use first 100 chars as fingerprint (normalized)
                        fp = _normalize_text(q[:100])
                        fingerprints.add(fp)
                except (json.JSONDecodeError, KeyError):
                    continue

    logger.info(f"Loaded {len(fingerprints)} test set fingerprints for contamination check")
    return fingerprints


def _normalize_text(text: str) -> str:
    """Normalize text for fingerprint matching."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text[:100]


def is_contaminated(text: str, test_fingerprints: set[str]) -> bool:
    """Check if text matches any test set question."""
    fp = _normalize_text(text[:100])
    return fp in test_fingerprints


def convert_to_sft_format(
    instruction: str,
    input_text: str = "",
    output_text: str = "",
    source: str = "unknown",
    dataset_id: str = "",
) -> dict:
    """Convert to unified SFT chat format."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a medical AI assistant with expertise in clinical "
                "medicine, diagnostics, and patient care. Provide accurate, "
                "evidence-based medical responses."
            ),
        },
    ]

    # Combine instruction and input
    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"

    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": output_text})

    return {
        "messages": messages,
        "metadata": {
            "source": source,
            "dataset": dataset_id,
        },
    }


def download_medical_o1_reasoning(
    max_samples: int,
    test_fps: set[str],
) -> list[dict]:
    """Download FreedomIntelligence/medical-o1-reasoning-SFT."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        return []

    logger.info("Downloading medical-o1-reasoning-SFT...")
    try:
        ds = load_dataset(
            "FreedomIntelligence/medical-o1-reasoning-SFT",
            "en",
            split="train",
        )
    except Exception as e:
        logger.warning(f"Failed to download medical-o1-reasoning-SFT: {e}")
        return []

    examples = []
    skipped = 0

    for row in ds:
        if len(examples) >= max_samples:
            break

        question = row.get("Question", row.get("question", ""))
        answer = row.get("Response", row.get("answer", ""))
        cot = row.get("Complex_CoT", row.get("complex_cot", ""))

        if not question or not answer:
            continue

        if is_contaminated(question, test_fps):
            skipped += 1
            continue

        # Use CoT reasoning as the output
        output = cot if cot else answer
        examples.append(convert_to_sft_format(
            instruction=question,
            output_text=output,
            source="medical-o1-reasoning-SFT",
            dataset_id="FreedomIntelligence/medical-o1-reasoning-SFT",
        ))

    logger.info(f"medical-o1-reasoning-SFT: {len(examples)} kept, {skipped} contaminated")
    return examples


def download_bioinstruct(
    max_samples: int,
    test_fps: set[str],
) -> list[dict]:
    """Download bio-nlp-umass/bioinstruct."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    logger.info("Downloading bioinstruct...")
    try:
        ds = load_dataset(
            "bio-nlp-umass/bioinstruct",
            split="train",
        )
    except Exception as e:
        logger.warning(f"Failed to download bioinstruct: {e}")
        return []

    examples = []
    skipped = 0

    for row in ds:
        if len(examples) >= max_samples:
            break

        instruction = row.get("instruction", "")
        input_text = row.get("input", "")
        output_text = row.get("output", "")

        if not instruction or not output_text:
            continue

        check_text = instruction + " " + input_text
        if is_contaminated(check_text, test_fps):
            skipped += 1
            continue

        examples.append(convert_to_sft_format(
            instruction=instruction,
            input_text=input_text,
            output_text=output_text,
            source="bioinstruct",
            dataset_id="bio-nlp-umass/bioinstruct",
        ))

    logger.info(f"bioinstruct: {len(examples)} kept, {skipped} contaminated")
    return examples


def download_medical_meadow_wikidoc(
    max_samples: int,
    test_fps: set[str],
) -> list[dict]:
    """Download medalpaca/medical_meadow_wikidoc."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    logger.info("Downloading medical_meadow_wikidoc...")
    try:
        ds = load_dataset(
            "medalpaca/medical_meadow_wikidoc",
            split="train",
        )
    except Exception as e:
        logger.warning(f"Failed to download medical_meadow_wikidoc: {e}")
        return []

    examples = []
    skipped = 0

    for row in ds:
        if len(examples) >= max_samples:
            break

        instruction = row.get("instruction", row.get("input", ""))
        output_text = row.get("output", "")

        if not instruction or not output_text:
            continue

        if is_contaminated(instruction, test_fps):
            skipped += 1
            continue

        examples.append(convert_to_sft_format(
            instruction=instruction,
            output_text=output_text,
            source="medical_meadow_wikidoc",
            dataset_id="medalpaca/medical_meadow_wikidoc",
        ))

    logger.info(f"medical_meadow_wikidoc: {len(examples)} kept, {skipped} contaminated")
    return examples


def download_medical_flashcards(
    max_samples: int,
    test_fps: set[str],
) -> list[dict]:
    """Download medalpaca/medical_meadow_medical_flashcards."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    logger.info("Downloading medical_meadow_medical_flashcards...")
    try:
        ds = load_dataset(
            "medalpaca/medical_meadow_medical_flashcards",
            split="train",
        )
    except Exception as e:
        logger.warning(f"Failed to download medical_meadow_medical_flashcards: {e}")
        return []

    examples = []
    skipped = 0

    for row in ds:
        if len(examples) >= max_samples:
            break

        instruction = row.get("instruction", row.get("input", ""))
        output_text = row.get("output", "")

        if not instruction or not output_text:
            continue

        if is_contaminated(instruction, test_fps):
            skipped += 1
            continue

        examples.append(convert_to_sft_format(
            instruction=instruction,
            output_text=output_text,
            source="medical_flashcards",
            dataset_id="medalpaca/medical_meadow_medical_flashcards",
        ))

    logger.info(f"medical_flashcards: {len(examples)} kept, {skipped} contaminated")
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare contamination-free medical SFT data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "datasets/sft/opensource_medical_sft.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=5000,
        help="Max samples per dataset",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["medical-o1", "bioinstruct", "wikidoc", "flashcards"],
        help="Which datasets to download",
    )
    args = parser.parse_args()

    # Load test set fingerprints for contamination check
    test_fps = load_test_questions()

    all_examples = []
    dataset_funcs = {
        "medical-o1": download_medical_o1_reasoning,
        "bioinstruct": download_bioinstruct,
        "wikidoc": download_medical_meadow_wikidoc,
        "flashcards": download_medical_flashcards,
    }

    for ds_name in args.datasets:
        if ds_name not in dataset_funcs:
            logger.warning(f"Unknown dataset: {ds_name}")
            continue

        examples = dataset_funcs[ds_name](args.max_per_dataset, test_fps)
        all_examples.extend(examples)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    source_counts = {}
    for ex in all_examples:
        src = ex["metadata"]["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    logger.info("=" * 60)
    logger.info("Download Complete!")
    logger.info(f"  Total examples: {len(all_examples)}")
    for src, count in sorted(source_counts.items()):
        logger.info(f"  - {src}: {count}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)

    # Save stats
    stats = {
        "total": len(all_examples),
        "per_source": source_counts,
        "contamination_check": "enabled",
        "test_fingerprints": len(test_fps),
    }
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
