"""Convert MedQA train questions into GYM-compatible task format.

Usage:
    python scripts/expand_medqa_gym_tasks.py --max-tasks 1000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MEDQA_TRAIN_PATH = (
    PROJECT_ROOT
    / "evaluations"
    / "self-biorag"
    / "data"
    / "benchmark"
    / "med_qa_train_gpt4.jsonl"
)

EXISTING_TASKS_PATH = (
    PROJECT_ROOT / "data" / "domains" / "medical_qa_200" / "tasks.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "domains" / "medical_qa_1000"
OUTPUT_TASKS_PATH = OUTPUT_DIR / "tasks.json"
OUTPUT_SPLIT_PATH = OUTPUT_DIR / "split_tasks.json"

TRAIN_RATIO = 0.8
SEED = 42

# Simple keyword-to-category mapping for medical topics
CATEGORY_KEYWORDS: List[Tuple[List[str], str]] = [
    (["heart", "cardiac", "myocard", "coronary", "atrial", "ventricul", "arrhythmia", "murmur", "endocard"], "cardiology"),
    (["lung", "pulmonary", "respir", "pneumo", "bronch", "asthma", "copd", "pleural"], "pulmonology"),
    (["kidney", "renal", "nephro", "glomerul", "dialysis", "ureter", "bladder", "urin"], "nephrology"),
    (["liver", "hepat", "cirrho", "biliru", "jaundice", "gallbladder", "bile", "cholest"], "gastroenterology"),
    (["brain", "neuro", "seizure", "stroke", "cerebr", "meningit", "dementia", "alzheimer"], "neurology"),
    (["skin", "dermat", "rash", "eczema", "psoriasis", "melanoma", "pruritus"], "dermatology"),
    (["bone", "fracture", "joint", "arthritis", "osteo", "orthoped", "musculoskeletal"], "orthopedics"),
    (["pregnan", "obstet", "gestation", "fetus", "fetal", "prenatal", "labor", "delivery", "postpartum"], "obstetrics"),
    (["child", "pediatr", "infant", "neonat", "newborn", "adolescent"], "pediatrics"),
    (["cancer", "tumor", "malignan", "oncol", "carcinoma", "lymphoma", "leukemia", "metasta"], "oncology"),
    (["thyroid", "adrenal", "pituitary", "endocrin", "diabet", "insulin", "cortisol", "hormone"], "endocrinology"),
    (["blood", "anemia", "hemato", "platelet", "coagul", "thrombo", "bleeding", "hemophilia"], "hematology"),
    (["infect", "bacter", "virus", "viral", "antibiotic", "hiv", "aids", "sepsis", "fever"], "infectious_disease"),
    (["immun", "autoimmun", "allerg", "lupus", "rheumat"], "immunology"),
    (["psychiatr", "depress", "anxiety", "schizophren", "bipolar", "psycho", "mental"], "psychiatry"),
    (["eye", "ophthalm", "retina", "vision", "glaucoma", "cataract"], "ophthalmology"),
    (["ear", "hearing", "otitis", "sinus", "pharyn", "laryn", "tonsil"], "otolaryngology"),
    (["surgery", "surgical", "incision", "operative", "resect", "appendect"], "surgery"),
    (["pharmaco", "drug", "medication", "dosage", "adverse effect", "side effect", "toxicity"], "pharmacology"),
    (["anatomy", "histolog", "cell", "tissue", "patholog", "microscop", "biopsy"], "pathology"),
    (["genetic", "chromosom", "mutation", "heredit", "autosomal", "x-linked", "gene"], "genetics"),
    (["epidemiol", "prevalence", "incidence", "screening", "public health", "statistic", "risk factor"], "epidemiology"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, data: Any) -> None:
    """Save data to a JSON file with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def extract_existing_questions(tasks: List[Dict[str, Any]]) -> Set[str]:
    """Extract raw question texts from existing tasks for deduplication."""
    questions: Set[str] = set()
    for task in tasks:
        raw_q = task.get("raw_question", "")
        if raw_q:
            # Normalise whitespace for robust matching
            questions.add(_normalise(raw_q))
    return questions


def _normalise(text: str) -> str:
    """Collapse whitespace for comparison."""
    return " ".join(text.split()).strip().lower()


def parse_answer_letter(answer_str: str) -> str:
    """Extract the single answer letter from MedQA answer field.

    Handles formats like '(D)', '(D) Nitrofurantoin', 'D', etc.
    """
    match = re.match(r"\(?([A-D])\)?", answer_str.strip())
    if match:
        return match.group(1)
    # Fallback: look for a letter anywhere
    match = re.search(r"\(([A-D])\)", answer_str)
    if match:
        return match.group(1)
    return "A"  # safe fallback


def parse_options(question_text: str) -> Dict[str, str]:
    """Parse options from the question text or instances input.

    Looks for patterns like 'Option A: ...' or '(A) ...'.
    """
    options: Dict[str, str] = {}

    # Try 'Option X: ...' format (from instances.input)
    for match in re.finditer(r"Option ([A-D]):\s*(.+?)(?=\nOption [A-D]:|$)", question_text, re.DOTALL):
        options[match.group(1)] = match.group(2).strip()

    if options:
        return options

    # Try '(X) ...' format (from question field)
    for match in re.finditer(r"\(([A-D])\)\s*(.+?)(?=\s*\([A-D]\)|$)", question_text, re.DOTALL):
        options[match.group(1)] = match.group(2).strip()

    return options


def extract_raw_question(question_text: str) -> str:
    """Extract the question stem without answer options."""
    # Remove '(A) ...' style options
    cleaned = re.split(r"\s*\(A\)\s", question_text, maxsplit=1)[0].strip()
    return cleaned


def classify_category(question_text: str) -> str:
    """Assign a medical category based on keyword matching."""
    text_lower = question_text.lower()
    for keywords, category in CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in text_lower:
                return category
    return "general_medicine"


def classify_difficulty(question_text: str, explanation: str) -> str:
    """Heuristic difficulty classification."""
    text_len = len(question_text)
    expl_len = len(explanation) if explanation else 0

    if text_len > 800 or expl_len > 600:
        return "hard"
    if text_len > 400 or expl_len > 300:
        return "medium"
    return "easy"


def convert_medqa_to_gym_task(
    record: Dict[str, Any],
    task_index: int,
) -> Dict[str, Any]:
    """Convert a single MedQA record to GYM task format."""
    question_text: str = record.get("question", "")
    answer_str: str = record.get("answer", "")
    explanation: str = record.get("explanation", "")
    instances: Dict[str, Any] = record.get("instances", {})
    instances_input: str = instances.get("input", "")
    instances_output: str = instances.get("output", "")

    # Parse answer letter
    correct_letter = parse_answer_letter(answer_str)

    # Parse options – prefer instances.input format (cleaner)
    options = parse_options(instances_input) if instances_input else parse_options(question_text)

    # Raw question (stem only)
    raw_question = extract_raw_question(question_text) if question_text else ""

    # Build the ticket (formatted question with options)
    if instances_input:
        ticket = f"QUESTION: {instances_input}"
    else:
        ticket = f"QUESTION: {question_text}"

    # Category and difficulty
    category = classify_category(question_text)
    difficulty = classify_difficulty(question_text, explanation)

    # Raw answer text
    raw_answer = instances_output if instances_output else options.get(correct_letter, answer_str)

    # Task ID
    task_id = f"medqa_{task_index:05d}"

    # Build evaluation criteria
    evaluation_criteria = _build_evaluation_criteria(
        correct_letter=correct_letter,
        category=category,
        raw_answer=raw_answer,
    )

    # Build description
    description = {
        "purpose": f"Answer a MedQA medical question on {category.replace('_', ' ')}",
        "difficulty": difficulty,
        "source": "MedQA",
        "category": category,
        "key_challenges": [
            f"Requires {category.replace('_', ' ')} knowledge",
            "Evidence-based reasoning needed" if difficulty != "easy" else "Basic recall question",
        ],
    }

    task: Dict[str, Any] = {
        "id": task_id,
        "description": description,
        "ticket": ticket,
        "correct_answer": correct_letter,
        "options": options,
        "raw_question": raw_question,
        "raw_answer": raw_answer,
        "evaluation_criteria": evaluation_criteria,
    }
    return task


def _build_evaluation_criteria(
    correct_letter: str,
    category: str,
    raw_answer: str,
) -> Dict[str, Any]:
    """Build GYM evaluation_criteria with accuracy-based assertions."""
    category_label = category.replace("_", " ")
    return {
        "actions": [
            {
                "action_id": "search_evidence",
                "name": "search_pubmed",
                "arguments": {},
                "info": f"Search for evidence relevant to {category_label}",
            },
            {
                "action_id": "submit",
                "name": "submit_answer",
                "arguments": {"answer": correct_letter},
                "compare_args": ["answer"],
                "info": f"Submit the correct answer: {correct_letter}",
            },
        ],
        "nl_assertions": [
            f"The agent selected option {correct_letter} as the correct answer",
            "The agent used evidence search tools before answering",
            f"The agent demonstrated {category_label} reasoning",
        ],
        "reward_basis": ["ACTION", "NL_ASSERTION"],
    }


def create_train_test_split(
    task_ids: List[str],
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> Dict[str, List[str]]:
    """Create an 80/20 train/test split of task IDs."""
    rng = random.Random(seed)
    shuffled = list(task_ids)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return {
        "train": shuffled[:split_idx],
        "test": shuffled[split_idx:],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Expand MedQA train data into GYM-compatible task format.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=1000,
        help="Maximum number of tasks to generate (default: 1000).",
    )
    parser.add_argument(
        "--medqa-path",
        type=str,
        default=str(MEDQA_TRAIN_PATH),
        help="Path to MedQA train JSONL file.",
    )
    parser.add_argument(
        "--existing-tasks-path",
        type=str,
        default=str(EXISTING_TASKS_PATH),
        help="Path to existing tasks.json for deduplication.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for generated files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for train/test split (default: 42).",
    )
    args = parser.parse_args(argv)

    medqa_path = Path(args.medqa_path)
    existing_path = Path(args.existing_tasks_path)
    output_dir = Path(args.output_dir)

    # 1. Load existing tasks for dedup
    existing_questions: Set[str] = set()
    if existing_path.exists():
        existing_tasks = load_json(existing_path)
        existing_questions = extract_existing_questions(existing_tasks)
        print(f"Loaded {len(existing_tasks)} existing tasks for deduplication.")
    else:
        print(f"Warning: existing tasks file not found at {existing_path}")

    # 2. Load MedQA train data
    if not medqa_path.exists():
        print(f"Error: MedQA train file not found at {medqa_path}", file=sys.stderr)
        sys.exit(1)

    medqa_records = load_jsonl(medqa_path)
    print(f"Loaded {len(medqa_records)} MedQA train records.")

    # 3. Filter and convert
    gym_tasks: List[Dict[str, Any]] = []
    skipped_dup = 0
    skipped_invalid = 0
    task_index = 0

    for record in medqa_records:
        if len(gym_tasks) >= args.max_tasks:
            break

        question_text = record.get("question", "")
        if not question_text:
            skipped_invalid += 1
            continue

        # Dedup by normalised question text
        raw_q = extract_raw_question(question_text)
        if _normalise(raw_q) in existing_questions:
            skipped_dup += 1
            continue

        # Parse options and validate
        instances_input = record.get("instances", {}).get("input", "")
        options = parse_options(instances_input) if instances_input else parse_options(question_text)
        if len(options) < 2:
            skipped_invalid += 1
            continue

        task = convert_medqa_to_gym_task(record, task_index)
        gym_tasks.append(task)
        task_index += 1

    print(f"Generated {len(gym_tasks)} GYM tasks.")
    print(f"Skipped {skipped_dup} duplicates, {skipped_invalid} invalid records.")

    if not gym_tasks:
        print("No tasks generated. Exiting.", file=sys.stderr)
        sys.exit(1)

    # 4. Save tasks.json
    save_json(output_dir / "tasks.json", gym_tasks)
    print(f"Saved tasks.json to {output_dir / 'tasks.json'}")

    # 5. Create and save split_tasks.json
    task_ids = [t["id"] for t in gym_tasks]
    split = create_train_test_split(task_ids, seed=args.seed)
    save_json(output_dir / "split_tasks.json", split)
    print(
        f"Saved split_tasks.json: {len(split['train'])} train, "
        f"{len(split['test'])} test"
    )


if __name__ == "__main__":
    main()
