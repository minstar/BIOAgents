"""Load and process MedQA, MedMCQA, and MMLU data into BIOAgents unified format.

Usage:
    python -m bioagents.data_pipeline.medqa_loader \
        --source medqa --split test --max_samples 100 \
        --output datasets/medical_qa/medqa_test_100.json
"""

import json
import re
import os
import argparse
from pathlib import Path
from typing import Optional

from loguru import logger


# ---- Raw data paths relative to project root ----
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_BENCHMARK_DIR = _PROJECT_ROOT / "evaluations" / "self-biorag" / "data" / "benchmark"
_INSTRUCTION_DIR = _PROJECT_ROOT / "databases" / "instruction"


def _parse_medqa_options(text: str) -> list[dict]:
    """Parse 'Option A: ...\nOption B: ...' into structured options list."""
    options = []
    # Match patterns like "Option A: text" or "\nOption A: text"
    pattern = r"Option\s+([A-E]):\s*(.*?)(?=\nOption\s+[A-E]:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    for label, opt_text in matches:
        options.append({"label": label.strip(), "text": opt_text.strip()})
    return options


def _extract_correct_label(question_text: str, answer_text: str, options: list[dict]) -> str:
    """Match the answer text to the correct option label."""
    answer_lower = answer_text.strip().lower()

    # Direct label match (e.g., answer is "A" or "B")
    if len(answer_lower) == 1 and answer_lower.upper() in "ABCDE":
        return answer_lower.upper()

    # Text match
    for opt in options:
        if opt["text"].strip().lower() == answer_lower:
            return opt["label"]

    # Fuzzy: check if answer text is a substring of any option
    for opt in options:
        if answer_lower in opt["text"].strip().lower():
            return opt["label"]

    # Fallback: return empty (unknown)
    return ""


def load_medqa_jsonl(
    split: str = "test",
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load MedQA data from JSONL files.

    Args:
        split: 'test' or 'train'
        max_samples: Maximum number of samples to load

    Returns:
        List of unified task dicts
    """
    if split == "test":
        filepath = _BENCHMARK_DIR / "med_qa_test.jsonl"
    elif split == "train":
        filepath = _BENCHMARK_DIR / "med_qa_train.json"
    else:
        filepath = _BENCHMARK_DIR / f"med_qa_{split}.jsonl"

    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return []

    records = []
    # Try JSONL format first
    with open(filepath, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
            if max_samples and len(records) >= max_samples:
                break

    logger.info(f"Loaded {len(records)} MedQA records from {filepath}")
    return _convert_medqa_records(records, source="MedQA")


def load_medmcqa_jsonl(
    split: str = "test",
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load MedMCQA data from JSONL files."""
    filepath = _BENCHMARK_DIR / f"medmc_qa_{split}.jsonl"
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return []

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
            if max_samples and len(records) >= max_samples:
                break

    logger.info(f"Loaded {len(records)} MedMCQA records from {filepath}")
    return _convert_medqa_records(records, source="MedMCQA")


def load_mmlu_jsonl(
    split: str = "test",
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load MMLU medical data from JSONL files."""
    filepath = _BENCHMARK_DIR / f"mmlu_{split}.jsonl"
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return []

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
            if max_samples and len(records) >= max_samples:
                break

    logger.info(f"Loaded {len(records)} MMLU records from {filepath}")
    return _convert_medqa_records(records, source="MMLU")


def _convert_medqa_records(records: list[dict], source: str) -> list[dict]:
    """Convert raw MedQA/MedMCQA/MMLU records into BIOAgents unified task format."""
    tasks = []

    for idx, record in enumerate(records):
        instances = record.get("instances", {})
        question_text = instances.get("input", "")
        answer_text = instances.get("output", "")

        if not question_text:
            continue

        # Parse options from question text
        options = _parse_medqa_options(question_text)
        if not options:
            continue

        # Extract correct answer label
        correct_label = _extract_correct_label(question_text, answer_text, options)
        if not correct_label:
            continue

        # Extract just the question (before options)
        q_match = re.search(r"QUESTION:\s*(.*?)(?=\nOption\s+[A-E]:)", question_text, re.DOTALL)
        if q_match:
            clean_question = q_match.group(1).strip()
        else:
            clean_question = question_text.split("Option")[0].strip()
            if clean_question.startswith("QUESTION:"):
                clean_question = clean_question[9:].strip()

        # Determine category from record name
        record_name = record.get("name", "")
        if "anatomy" in record_name.lower():
            category = "anatomy"
        elif "clinical_knowledge" in record_name.lower():
            category = "clinical_knowledge"
        elif "biology" in record_name.lower():
            category = "biology"
        elif "genetics" in record_name.lower():
            category = "genetics"
        elif "medicine" in record_name.lower():
            category = "medicine"
        else:
            category = "general"

        task_id = f"{source.lower()}_{idx:05d}"

        # Build the ticket (question with options)
        ticket_parts = [f"QUESTION: {clean_question}", ""]
        for opt in options:
            ticket_parts.append(f"Option {opt['label']}: {opt['text']}")

        task = {
            "id": task_id,
            "description": {
                "purpose": f"Answer a {source} medical question",
                "difficulty": "medium",
                "source": source,
                "category": category,
            },
            "ticket": "\n".join(ticket_parts),
            "correct_answer": correct_label,
            "options": options,
            "raw_question": clean_question,
            "raw_answer": answer_text,
            "evaluation_criteria": {
                "actions": [
                    {
                        "action_id": "submit",
                        "name": "submit_answer",
                        "arguments": {"answer": correct_label},
                        "compare_args": ["answer"],
                        "info": f"Submit the correct answer: {correct_label}",
                    }
                ],
                "nl_assertions": [
                    f"The agent selected option {correct_label} as the correct answer",
                    "The agent used evidence search tools before answering",
                ],
                "reward_basis": ["ACTION", "NL_ASSERTION"],
            },
        }

        tasks.append(task)

    logger.info(f"Converted {len(tasks)} {source} records to BIOAgents format")
    return tasks


def load_instruction_data(
    source: str = "MedInstruct-52k",
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load medical instruction data for SFT.

    Args:
        source: One of 'MedInstruct-52k', 'all_biomedical_instruction',
                'mol_instruction_qa', 'self_instruct_biomedical'
        max_samples: Maximum number of samples

    Returns:
        List of instruction dicts with 'instruction', 'input', 'output' keys
    """
    filepath = _INSTRUCTION_DIR / f"{source}.json"
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    logger.info(f"Loaded {len(data)} instruction records from {source}")
    return data


def generate_unified_dataset(
    sources: list[str] = None,
    split: str = "test",
    max_per_source: Optional[int] = None,
    output_path: Optional[str] = None,
) -> list[dict]:
    """Generate a unified medical QA dataset from multiple sources.

    Args:
        sources: List of sources to include. Default: ['medqa', 'medmcqa', 'mmlu']
        split: Data split ('test' or 'train')
        max_per_source: Max samples per source
        output_path: Optional path to save the output JSON

    Returns:
        Unified list of task dicts
    """
    if sources is None:
        sources = ["medqa", "medmcqa", "mmlu"]

    all_tasks = []
    loaders = {
        "medqa": load_medqa_jsonl,
        "medmcqa": load_medmcqa_jsonl,
        "mmlu": load_mmlu_jsonl,
    }

    for source in sources:
        loader = loaders.get(source)
        if loader is None:
            logger.warning(f"Unknown source: {source}")
            continue

        tasks = loader(split=split, max_samples=max_per_source)
        all_tasks.extend(tasks)
        logger.info(f"  {source}: {len(tasks)} tasks")

    logger.info(f"Total unified dataset: {len(all_tasks)} tasks")

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(all_tasks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved to {output_path}")

    return all_tasks


def get_dataset_stats(tasks: list[dict]) -> dict:
    """Get statistics about the processed dataset."""
    from collections import Counter

    sources = Counter(t["description"]["source"] for t in tasks)
    categories = Counter(t["description"]["category"] for t in tasks)

    # Check answer distribution
    answers = Counter(t["correct_answer"] for t in tasks)

    return {
        "total": len(tasks),
        "by_source": dict(sources),
        "by_category": dict(categories),
        "answer_distribution": dict(answers),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process medical QA datasets")
    parser.add_argument("--sources", nargs="+", default=["medqa", "medmcqa", "mmlu"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument(
        "--output", default="datasets/medical_qa/unified_test.json"
    )
    args = parser.parse_args()

    tasks = generate_unified_dataset(
        sources=args.sources,
        split=args.split,
        max_per_source=args.max_per_source,
        output_path=args.output,
    )

    stats = get_dataset_stats(tasks)
    print("\n=== Dataset Statistics ===")
    print(json.dumps(stats, indent=2))
