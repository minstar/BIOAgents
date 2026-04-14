#!/usr/bin/env python3
"""Build a compact multi-domain evaluation set for tracking model performance during RL training.

Samples from each benchmark to create a small but representative tracking eval set.
Designed to run in ~5-10 min instead of hours for full benchmarks.

Domains sampled:
  - MedQA (4-option MCQ): 50 samples
  - MedMCQA (4-option MCQ): 50 samples
  - MMLU Clinical Knowledge (4-option MCQ): 30 samples
  - VQA-RAD (radiology VQA via HuggingFace): 30 samples
  - PathVQA (pathology VQA via local data): 30 samples
  - EHR management (agentic clinical tasks): 20 samples
  - MedLFQA (long-form QA): 20 samples

Output: data/eval/tracking_eval_set.json

Usage:
    python scripts/build_tracking_eval_set.py [--seed 42] [--output data/eval/tracking_eval_set.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _parse_options_from_input(input_text: str) -> List[str]:
    """Extract option labels (A/B/C/D/E) and their text from a question input string."""
    options = []
    pattern = re.compile(r"Option\s+([A-E]):\s*(.+?)(?=\nOption\s+[A-E]:|$)", re.DOTALL)
    for match in pattern.finditer(input_text):
        label = match.group(1)
        text = match.group(2).strip()
        options.append(f"{label}: {text}")
    return options


def _parse_question_from_input(input_text: str) -> str:
    """Extract the question text (before options) from a benchmark input string."""
    # Remove leading "QUESTION: " prefix
    text = input_text.strip()
    if text.startswith("QUESTION:"):
        text = text[len("QUESTION:"):].strip()
    # Cut before first option
    option_start = re.search(r"\nOption\s+[A-E]:", text)
    if option_start:
        text = text[:option_start.start()].strip()
    return text


# ──────────────────────────────────────────────────────────────
#  Domain samplers
# ──────────────────────────────────────────────────────────────

def sample_medqa(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Sample from MedQA (4-option) test set."""
    path = PROJECT_ROOT / "evaluations" / "self-biorag" / "data" / "benchmark" / "med_qa_test.jsonl"
    if not path.exists():
        print(f"  [WARN] MedQA not found at {path}", file=sys.stderr)
        return []

    raw = _load_jsonl(path)
    sampled = rng.sample(raw, min(n, len(raw)))
    results = []
    for idx, item in enumerate(sampled):
        input_text = item.get("instances", {}).get("input", "")
        question = _parse_question_from_input(input_text)
        options = _parse_options_from_input(input_text)
        answer = item.get("instances", {}).get("output", "").strip()

        results.append({
            "id": f"medqa_{idx:03d}",
            "question": question,
            "answer": answer,
            "options": options,
            "source": "medqa",
            "domain": "medical_qa",
        })
    print(f"  MedQA: {len(results)} samples")
    return results


def sample_medmcqa(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Sample from MedMCQA test set."""
    path = PROJECT_ROOT / "evaluations" / "self-biorag" / "data" / "benchmark" / "medmc_qa_test.jsonl"
    if not path.exists():
        print(f"  [WARN] MedMCQA not found at {path}", file=sys.stderr)
        return []

    raw = _load_jsonl(path)
    sampled = rng.sample(raw, min(n, len(raw)))
    results = []
    for idx, item in enumerate(sampled):
        input_text = item.get("instances", {}).get("input", "")
        question = _parse_question_from_input(input_text)
        options = _parse_options_from_input(input_text)
        answer = item.get("instances", {}).get("output", "").strip()

        results.append({
            "id": f"medmcqa_{idx:03d}",
            "question": question,
            "answer": answer,
            "options": options,
            "source": "medmcqa",
            "domain": "medical_qa",
        })
    print(f"  MedMCQA: {len(results)} samples")
    return results


def sample_mmlu_clinical(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Sample from MMLU Clinical Knowledge test set."""
    path = PROJECT_ROOT / "evaluations" / "self-biorag" / "data" / "benchmark" / "mmlu_clinical_knowledge_test.jsonl"
    if not path.exists():
        print(f"  [WARN] MMLU Clinical Knowledge not found at {path}", file=sys.stderr)
        return []

    raw = _load_jsonl(path)
    sampled = rng.sample(raw, min(n, len(raw)))
    results = []
    for idx, item in enumerate(sampled):
        input_text = item.get("instances", {}).get("input", "")
        question = _parse_question_from_input(input_text)
        options = _parse_options_from_input(input_text)
        answer = item.get("instances", {}).get("output", "").strip()

        results.append({
            "id": f"mmlu_ck_{idx:03d}",
            "question": question,
            "answer": answer,
            "options": options,
            "source": "mmlu_clinical_knowledge",
            "domain": "clinical_knowledge",
        })
    print(f"  MMLU Clinical Knowledge: {len(results)} samples")
    return results


def sample_vqa_rad(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Sample from VQA-RAD via HuggingFace datasets library."""
    try:
        from datasets import load_dataset
        ds = load_dataset("flaviagiammarino/vqa-rad", split="test")
    except Exception as e:
        print(f"  [WARN] VQA-RAD not available (HuggingFace load failed: {e})", file=sys.stderr)
        return []

    indices = list(range(len(ds)))
    sampled_indices = rng.sample(indices, min(n, len(indices)))
    results = []
    image_dir = PROJECT_ROOT / "datasets" / "vqa" / "vqa_rad" / "images"
    for idx, si in enumerate(sampled_indices):
        item = ds[si]
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        image_path = str(image_dir / f"vqarad_test_{si:05d}.png")
        if not Path(image_path).exists():
            image_path = None

        results.append({
            "id": f"vqa_rad_{idx:03d}",
            "question": question,
            "answer": answer,
            "options": None,
            "source": "vqa_rad",
            "domain": "radiology_vqa",
            "image_path": image_path,
        })
    print(f"  VQA-RAD: {len(results)} samples")
    return results


def sample_pathvqa(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Sample from PathVQA (local lxmert JSON format)."""
    path = PROJECT_ROOT / "evaluations" / "PathVQA" / "baselines" / "method1" / "saved" / "lxmert" / "pvqa_test.json"
    if not path.exists():
        # Fallback: try HuggingFace
        try:
            from datasets import load_dataset
            ds = load_dataset("flaviagiammarino/path-vqa", split="test")
            indices = list(range(len(ds)))
            sampled_indices = rng.sample(indices, min(n, len(indices)))
            results = []
            for idx, si in enumerate(sampled_indices):
                item = ds[si]
                results.append({
                    "id": f"pathvqa_{idx:03d}",
                    "question": str(item.get("question", "")).strip(),
                    "answer": str(item.get("answer", "")).strip(),
                    "options": None,
                    "source": "pathvqa",
                    "domain": "pathology_vqa",
                    "image_path": None,
                })
            print(f"  PathVQA: {len(results)} samples (from HuggingFace)")
            return results
        except Exception as e:
            print(f"  [WARN] PathVQA not available ({e})", file=sys.stderr)
            return []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Flatten: each entry can have multiple Q/A pairs
    qa_pairs = []
    for item in raw:
        img_id = item.get("img_id", "")
        questions = item.get("sentf", {}).get("pvqa", [])
        answers = item.get("labelf", {}).get("pvqa", [])
        for q, a_dict in zip(questions, answers):
            if isinstance(a_dict, dict):
                answer = list(a_dict.keys())[0] if a_dict else ""
            else:
                answer = str(a_dict)
            qa_pairs.append({
                "img_id": img_id,
                "question": q,
                "answer": answer,
            })

    if not qa_pairs:
        print("  [WARN] PathVQA: no Q/A pairs extracted", file=sys.stderr)
        return []

    sampled = rng.sample(qa_pairs, min(n, len(qa_pairs)))
    results = []
    image_dir = PROJECT_ROOT / "datasets" / "vqa" / "pathvqa" / "images"
    for idx, pair in enumerate(sampled):
        img_id = pair["img_id"]
        image_path = str(image_dir / f"pvqa_test_{img_id.replace('test_', '')}.png") if img_id else None
        if image_path and not Path(image_path).exists():
            image_path = None

        results.append({
            "id": f"pathvqa_{idx:03d}",
            "question": pair["question"],
            "answer": pair["answer"],
            "options": None,
            "source": "pathvqa",
            "domain": "pathology_vqa",
            "image_path": image_path,
        })
    print(f"  PathVQA: {len(results)} samples (from local)")
    return results


def sample_ehr(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Sample from EHR management domain tasks."""
    path = PROJECT_ROOT / "data" / "domains" / "ehr_management" / "tasks.json"
    if not path.exists():
        print(f"  [WARN] EHR tasks not found at {path}", file=sys.stderr)
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sampled = rng.sample(raw, min(n, len(raw)))
    results = []
    for idx, item in enumerate(sampled):
        question = item.get("ticket", "")
        expected_answer = item.get("expected_answer", "")
        rubric = item.get("rubric", {})

        results.append({
            "id": f"ehr_{idx:03d}",
            "question": question,
            "answer": expected_answer,
            "options": None,
            "source": "ehr_management",
            "domain": "ehr_management",
            "category": item.get("category", ""),
            "difficulty": item.get("difficulty", ""),
            "rubric": rubric,
        })
    print(f"  EHR: {len(results)} samples")
    return results


def sample_medlfqa(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Sample from MedLFQA (long-form QA) datasets.

    Draws evenly across available LFQA sub-datasets.
    """
    lfqa_dir = PROJECT_ROOT / "evaluations" / "OLAPH" / "MedLFQA"
    if not lfqa_dir.exists():
        print(f"  [WARN] MedLFQA dir not found at {lfqa_dir}", file=sys.stderr)
        return []

    lfqa_files = {
        "live_qa": lfqa_dir / "live_qa_test_MedLFQA.jsonl",
        "medication_qa": lfqa_dir / "medication_qa_test_MedLFQA.jsonl",
        "healthsearch_qa": lfqa_dir / "healthsearch_qa_test_MedLFQA.jsonl",
        "kqa_golden": lfqa_dir / "kqa_golden_test_MedLFQA.jsonl",
    }

    # Collect all available samples with sub-dataset tag
    all_lfqa = []
    for sub_name, fpath in lfqa_files.items():
        if fpath.exists():
            records = _load_jsonl(fpath)
            for r in records:
                r["_subdataset"] = sub_name
            all_lfqa.extend(records)

    if not all_lfqa:
        print("  [WARN] MedLFQA: no data files found", file=sys.stderr)
        return []

    sampled = rng.sample(all_lfqa, min(n, len(all_lfqa)))
    results = []
    for idx, item in enumerate(sampled):
        question = item.get("Question", "")
        answer = item.get("Free_form_answer", "")
        must_have = item.get("Must_have", [])
        nice_to_have = item.get("Nice_to_have", [])

        results.append({
            "id": f"medlfqa_{idx:03d}",
            "question": question,
            "answer": answer,
            "options": None,
            "source": f"medlfqa/{item.get('_subdataset', 'unknown')}",
            "domain": "long_form_qa",
            "must_have": must_have,
            "nice_to_have": nice_to_have,
        })
    print(f"  MedLFQA: {len(results)} samples")
    return results


def sample_mmlu_subtype(filename: str, subtype: str, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Sample from a specific MMLU subtype test set."""
    path = PROJECT_ROOT / "evaluations" / "self-biorag" / "data" / "benchmark" / filename
    if not path.exists():
        print(f"  [WARN] MMLU {subtype} not found at {path}", file=sys.stderr)
        return []

    raw = _load_jsonl(path)
    sampled = rng.sample(raw, min(n, len(raw)))
    results = []
    for idx, item in enumerate(sampled):
        # MMLU files use nested "instances" structure (same as medqa/medmcqa)
        input_text = item.get("instances", {}).get("input", "")
        question = _parse_question_from_input(input_text)
        options = _parse_options_from_input(input_text)
        answer = item.get("instances", {}).get("output", "").strip()

        results.append({
            "id": f"mmlu_{subtype}_{idx:03d}",
            "question": question,
            "answer": answer,
            "options": options if options else None,
            "source": f"mmlu_{subtype}",
            "domain": "medical_mcq",
        })
    print(f"  MMLU {subtype}: {len(results)} samples")
    return results


def sample_slake(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Sample from SLAKE VQA dataset."""
    slake_dir = PROJECT_ROOT / "datasets" / "vqa" / "slake"
    # Try local data first
    test_file = slake_dir / "test.json"
    if not test_file.exists():
        # Try HuggingFace
        try:
            from datasets import load_dataset
            ds = load_dataset("mdwiratathya/SLAKE-vqa-english", split="test")
            indices = rng.sample(range(len(ds)), min(n, len(ds)))
            results = []
            for idx, si in enumerate(indices):
                item = ds[si]
                results.append({
                    "id": f"slake_{idx:03d}",
                    "question": str(item.get("question", "")).strip(),
                    "answer": str(item.get("answer", "")).strip(),
                    "options": None,
                    "source": "slake",
                    "domain": "medical_vqa",
                })
            print(f"  SLAKE: {len(results)} samples (HuggingFace)")
            return results
        except Exception:
            print(f"  [WARN] SLAKE not available locally or from HuggingFace", file=sys.stderr)
            return []

    with open(test_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sampled = rng.sample(raw, min(n, len(raw)))
    results = []
    for idx, item in enumerate(sampled):
        results.append({
            "id": f"slake_{idx:03d}",
            "question": str(item.get("question", "")).strip(),
            "answer": str(item.get("answer", "")).strip(),
            "options": None,
            "source": "slake",
            "domain": "medical_vqa",
        })
    print(f"  SLAKE: {len(results)} samples")
    return results


def sample_domain_tasks(domain: str, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Sample from a GYM domain task set.

    These are agentic tasks with evaluation_criteria (actions + nl_assertions).
    We use the ticket as the question and nl_assertions as the expected answer.
    """
    path = PROJECT_ROOT / "data" / "domains" / domain / "tasks.json"
    if not path.exists():
        print(f"  [WARN] {domain} tasks not found", file=sys.stderr)
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sampled = rng.sample(raw, min(n, len(raw)))
    results = []
    for idx, item in enumerate(sampled):
        question = item.get("ticket", item.get("description", {}).get("purpose", ""))

        # For agentic tasks, build answer from evaluation criteria
        answer = item.get("correct_answer", item.get("expected_answer", ""))
        if not answer:
            eval_criteria = item.get("evaluation_criteria", {})
            nl_assertions = eval_criteria.get("nl_assertions", [])
            if nl_assertions:
                answer = "; ".join(nl_assertions)
            else:
                # Fallback: use description purpose
                answer = item.get("description", {}).get("purpose", "")

        options_dict = item.get("options", {})
        options = None
        if options_dict and isinstance(options_dict, dict):
            options = [f"{k}: {v}" for k, v in sorted(options_dict.items())]

        results.append({
            "id": f"{domain}_{idx:03d}",
            "question": question,
            "answer": answer,
            "options": options,
            "source": domain,
            "domain": domain,
            "task_type": "agentic",
        })
    print(f"  {domain}: {len(results)} samples")
    return results


# ──────────────────────────────────────────────────────────────
#  Main builder
# ──────────────────────────────────────────────────────────────

def build_tracking_eval_set(
    seed: int = 42,
    output_path: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build the tracking evaluation set and save to disk."""
    rng = random.Random(seed)

    print(f"Building tracking eval set (seed={seed})...")

    eval_set = {
        "medqa": sample_medqa(50, rng),
        "medmcqa": sample_medmcqa(50, rng),
        "mmlu_clinical": sample_mmlu_clinical(30, rng),
        "mmlu_anatomy": sample_mmlu_subtype("mmlu_anatomy_test.jsonl", "anatomy", 15, rng),
        "mmlu_college_medicine": sample_mmlu_subtype("mmlu_college_medicine_test.jsonl", "college_medicine", 15, rng),
        "mmlu_professional_medicine": sample_mmlu_subtype("mmlu_professional_medicine_test.jsonl", "professional_medicine", 20, rng),
        "mmlu_medical_genetics": sample_mmlu_subtype("mmlu_medical_genetics_test.jsonl", "medical_genetics", 15, rng),
        "mmlu_college_biology": sample_mmlu_subtype("mmlu_college_biology_test.jsonl", "college_biology", 15, rng),
        "vqa_rad": sample_vqa_rad(30, rng),
        "pathvqa": sample_pathvqa(30, rng),
        "slake": sample_slake(20, rng),
        "ehr": sample_ehr(15, rng),
        "clinical_diagnosis": sample_domain_tasks("clinical_diagnosis", 15, rng),
        "drug_interaction": sample_domain_tasks("drug_interaction", 10, rng),
        "triage_emergency": sample_domain_tasks("triage_emergency", 10, rng),
        "medlfqa": sample_medlfqa(20, rng),
    }

    # Summary
    total = sum(len(v) for v in eval_set.values())
    print(f"\nTotal samples: {total}")
    for domain, samples in eval_set.items():
        print(f"  {domain}: {len(samples)}")

    # Save
    if output_path is None:
        output_path = str(PROJECT_ROOT / "data" / "eval" / "tracking_eval_set.json")

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")

    return eval_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact tracking eval set for RL training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    build_tracking_eval_set(seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    main()
