"""Unified VQA Data Loader for BIOAgents Healthcare AI GYM.

Loads and unifies 6 Visual Medical QA datasets into a common format:
1. VQA-RAD     — Radiology VQA (HuggingFace: flaviagiammarino/vqa-rad)
2. SLAKE       — Multilingual Medical VQA (local: evaluations/PMC-VQA/Slake1.0)
3. PathVQA     — Pathology VQA (local: evaluations/PathVQA + HuggingFace)
4. PMC-VQA     — Medical Paper Image VQA (HuggingFace: RadGenome/PMC-VQA)
5. VQA-Med-2021 — Medical VQA Challenge (local: evaluations/VQA-Med-2021)
6. Quilt-VQA   — Histopathology VQA (local: evaluations/quilt-llava)

Unified output format:
{
    "id": str,
    "dataset": str,           # source dataset name
    "image_path": str | None, # local image path (if available)
    "image_id": str,          # image identifier within dataset
    "question": str,
    "answer": str,
    "answer_type": str,       # "yes_no", "choice", "open_ended", "number"
    "options": list | None,   # MC options if applicable
    "modality": str,          # imaging modality
    "category": str,          # medical category
    "difficulty": str,        # "easy", "medium", "hard"
    "metadata": dict,         # dataset-specific extra info
}

Usage:
    from bioagents.data_pipeline.vqa_loader import (
        load_all_vqa_datasets,
        load_vqa_rad,
        load_slake,
        load_pathvqa,
        load_pmc_vqa,
        load_vqa_med_2021,
        load_quilt_vqa,
        get_vqa_stats,
    )

    # Load all available VQA datasets
    all_data = load_all_vqa_datasets(max_per_dataset=500)
    stats = get_vqa_stats(all_data)
"""

import csv
import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Optional

from loguru import logger

# ── Project root paths ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_EVAL_DIR = _PROJECT_ROOT / "evaluations"
_DATA_DIR = _PROJECT_ROOT / "datasets" / "vqa"


# ══════════════════════════════════════════════════════════════
#  1. VQA-RAD Loader
# ══════════════════════════════════════════════════════════════

def load_vqa_rad(
    max_samples: Optional[int] = None,
    split: str = "test",
) -> list[dict]:
    """Load VQA-RAD (Radiology VQA) dataset.

    Primary source: HuggingFace (flaviagiammarino/vqa-rad)
    Fallback: local data if previously downloaded.

    Args:
        max_samples: Max number of samples to load.
        split: "train" or "test".

    Returns:
        List of unified VQA dicts.
    """
    records = []

    # Try HuggingFace datasets first
    try:
        from datasets import load_dataset
        ds = load_dataset("flaviagiammarino/vqa-rad", split=split, trust_remote_code=True)
        logger.info(f"[VQA-RAD] Loaded {len(ds)} samples from HuggingFace ({split})")

        for idx, item in enumerate(ds):
            if max_samples and idx >= max_samples:
                break

            # Save image locally for reference
            image_dir = _DATA_DIR / "vqa_rad" / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"vqarad_{split}_{idx:05d}.png"

            if hasattr(item.get("image", None), "save") and not image_path.exists():
                try:
                    item["image"].save(str(image_path))
                except Exception:
                    image_path = None

            answer = str(item.get("answer", "")).strip()
            question = str(item.get("question", "")).strip()

            # Determine answer type
            answer_type = _classify_answer_type(answer, question)

            records.append({
                "id": f"vqa_rad_{split}_{idx:05d}",
                "dataset": "VQA-RAD",
                "image_path": str(image_path) if image_path and image_path.exists() else None,
                "image_id": f"vqarad_{idx}",
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "options": None,
                "modality": "radiology",
                "category": _categorize_radiology_question(question),
                "difficulty": "medium",
                "metadata": {
                    "source": "HuggingFace",
                    "split": split,
                },
            })

        return records

    except Exception as e:
        logger.warning(f"[VQA-RAD] HuggingFace loading failed: {e}")

    # Fallback: local data
    local_paths = [
        _EVAL_DIR / "PMC-VQA" / "Slake1.0",  # Sometimes VQA-RAD is co-located
    ]
    logger.warning("[VQA-RAD] No local fallback found. Use HuggingFace to download.")
    return records


# ══════════════════════════════════════════════════════════════
#  2. SLAKE Loader
# ══════════════════════════════════════════════════════════════

def load_slake(
    max_samples: Optional[int] = None,
    split: str = "test",
) -> list[dict]:
    """Load SLAKE (Semantically-Labeled Knowledge-Enhanced) Medical VQA dataset.

    Primary source: local data at evaluations/PMC-VQA/Slake1.0/
    Fallback: HuggingFace (BoKelvin/SLAKE)

    Args:
        max_samples: Max number of samples.
        split: "train", "test", or "validate".

    Returns:
        List of unified VQA dicts.
    """
    records = []

    # Try HuggingFace first (includes images)
    try:
        from datasets import load_dataset
        ds = load_dataset("BoKelvin/SLAKE", split=split, trust_remote_code=True)
        logger.info(f"[SLAKE] Loaded {len(ds)} samples from HuggingFace ({split})")

        image_dir = _DATA_DIR / "slake" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(ds):
            if max_samples and idx >= max_samples:
                break

            answer = str(item.get("answer", "")).strip()
            question = str(item.get("question", "")).strip()
            q_lang = item.get("q_lang", "en")

            # Filter to English only
            if q_lang and q_lang != "en":
                continue

            answer_type = _classify_answer_type(answer, question)

            # Save image locally
            image_path = image_dir / f"slake_{split}_{idx:05d}.png"
            if hasattr(item.get("image", None), "save") and not image_path.exists():
                try:
                    item["image"].save(str(image_path))
                except Exception:
                    image_path = None

            # Modality from content_type
            content_type = str(item.get("content_type", "")).lower()
            modality = _slake_content_to_modality(content_type, item) if content_type else "radiology"

            records.append({
                "id": f"slake_{split}_{idx:05d}",
                "dataset": "SLAKE",
                "image_path": str(image_path) if image_path and image_path.exists() else None,
                "image_id": item.get("img_id", f"slake_hf_{idx}"),
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "options": None,
                "modality": modality,
                "category": item.get("content_type", "general"),
                "difficulty": "medium",
                "metadata": {"source": "HuggingFace", "split": split, "q_lang": q_lang},
            })

        return records

    except Exception as e:
        logger.warning(f"[SLAKE] HuggingFace loading failed: {e}")

    # Fallback: local SLAKE data
    slake_dir = _EVAL_DIR / "PMC-VQA" / "Slake1.0"
    slake_json = slake_dir / f"{split}.json"

    if slake_json.exists():
        with open(slake_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"[SLAKE] Loaded {len(data)} samples from local ({split})")

        for idx, item in enumerate(data):
            if max_samples and idx >= max_samples:
                break

            # SLAKE format: img_id, img_name, question, answer, answer_type, etc.
            img_name = item.get("img_name", "")
            image_path = slake_dir / "imgs" / img_name if img_name else None

            answer = str(item.get("answer", "")).strip()
            question = str(item.get("question", "")).strip()
            q_lang = item.get("q_lang", "en")

            # Filter to English only by default
            if q_lang != "en":
                continue

            answer_type_raw = item.get("answer_type", "OPEN")
            answer_type = "yes_no" if answer_type_raw == "CLOSED" else "open_ended"

            # Modality from content_type
            content_type = item.get("content_type", "").lower()
            modality = _slake_content_to_modality(content_type, item)

            records.append({
                "id": f"slake_{split}_{idx:05d}",
                "dataset": "SLAKE",
                "image_path": str(image_path) if image_path and image_path.exists() else None,
                "image_id": item.get("img_id", f"slake_{idx}"),
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "options": None,
                "modality": modality,
                "category": item.get("content_type", "general"),
                "difficulty": "medium",
                "metadata": {
                    "source": "local",
                    "q_lang": q_lang,
                    "content_type": content_type,
                    "img_name": img_name,
                },
            })

        return records

    logger.warning("[SLAKE] No data found.")
    return records


# ══════════════════════════════════════════════════════════════
#  3. PathVQA Loader
# ══════════════════════════════════════════════════════════════

def load_pathvqa(
    max_samples: Optional[int] = None,
    split: str = "test",
) -> list[dict]:
    """Load PathVQA (Pathology VQA) dataset.

    Primary source: local data at evaluations/PathVQA/
    Fallback: HuggingFace (flaviagiammarino/path-vqa)

    Args:
        max_samples: Max number of samples.
        split: "train", "test", or "val".

    Returns:
        List of unified VQA dicts.
    """
    records = []

    # Try HuggingFace first (includes images)
    try:
        from datasets import load_dataset
        ds = load_dataset("flaviagiammarino/path-vqa", split=split, trust_remote_code=True)
        logger.info(f"[PathVQA] Loaded {len(ds)} samples from HuggingFace ({split})")

        for idx, item in enumerate(ds):
            if max_samples and idx >= max_samples:
                break

            answer = str(item.get("answer", "")).strip()
            question = str(item.get("question", "")).strip()
            answer_type = _classify_answer_type(answer, question)

            # Save image locally
            image_dir = _DATA_DIR / "pathvqa" / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"pvqa_{split}_{idx:05d}.png"

            if hasattr(item.get("image", None), "save") and not image_path.exists():
                try:
                    item["image"].save(str(image_path))
                except Exception:
                    image_path = None

            records.append({
                "id": f"pathvqa_{split}_{idx:05d}",
                "dataset": "PathVQA",
                "image_path": str(image_path) if image_path and image_path.exists() else None,
                "image_id": f"pvqa_hf_{idx}",
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "options": None,
                "modality": "pathology",
                "category": "histopathology",
                "difficulty": "medium",
                "metadata": {"source": "HuggingFace", "split": split},
            })

        return records

    except Exception as e:
        logger.warning(f"[PathVQA] Loading failed: {e}")
        return records


# ══════════════════════════════════════════════════════════════
#  4. PMC-VQA Loader
# ══════════════════════════════════════════════════════════════

def load_pmc_vqa(
    max_samples: Optional[int] = None,
    split: str = "test",
) -> list[dict]:
    """Load PMC-VQA (PubMedCentral Visual QA) dataset.

    Primary source: HuggingFace (RadGenome/PMC-VQA)
    Also attempts local CSV at evaluations/PMC-VQA/

    Args:
        max_samples: Max number of samples.
        split: "test" or "train".

    Returns:
        List of unified VQA dicts.
    """
    records = []

    # Try HuggingFace first (may include images)
    try:
        from datasets import load_dataset
        ds = load_dataset("RadGenome/PMC-VQA", split=split, trust_remote_code=True)
        logger.info(f"[PMC-VQA] Loaded {len(ds)} samples from HuggingFace ({split})")

        image_dir = _DATA_DIR / "pmc_vqa" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(ds):
            if max_samples and idx >= max_samples:
                break

            question = str(item.get("Question", "")).strip()
            answer = str(item.get("Answer", "")).strip()

            options = []
            for key in ["Choice A", "Choice B", "Choice C", "Choice D"]:
                opt = str(item.get(key, "")).strip()
                if opt:
                    options.append(opt)

            # Save image locally if available
            image_path = image_dir / f"pmc_{split}_{idx:05d}.png"
            if hasattr(item.get("image", None), "save") and not image_path.exists():
                try:
                    item["image"].save(str(image_path))
                except Exception:
                    image_path = None

            records.append({
                "id": f"pmc_vqa_{split}_{idx:05d}",
                "dataset": "PMC-VQA",
                "image_path": str(image_path) if image_path and image_path.exists() else None,
                "image_id": str(item.get("Figure_path", f"pmc_hf_{idx}")),
                "question": question,
                "answer": answer,
                "answer_type": "choice" if options else "open_ended",
                "options": options if options else None,
                "modality": "mixed",
                "category": "medical_literature",
                "difficulty": "hard",
                "metadata": {"source": "HuggingFace", "split": split},
            })

        return records

    except Exception as e:
        logger.warning(f"[PMC-VQA] HuggingFace loading failed: {e}")

    # Fallback: local CSV
    local_csv = _EVAL_DIR / "PMC-VQA" / f"{split}.csv"
    if local_csv.exists():
        try:
            with open(local_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)

            logger.info(f"[PMC-VQA] Loaded {len(data)} samples from local CSV ({split})")

            for idx, item in enumerate(data):
                if max_samples and idx >= max_samples:
                    break

                question = str(item.get("Question", "")).strip()
                answer = str(item.get("Answer", "")).strip()
                img_id = str(item.get("Figure_path", f"pmc_{idx}"))

                options = []
                for opt_key in ["Choice A", "Choice B", "Choice C", "Choice D"]:
                    opt_text = item.get(opt_key, "").strip()
                    if opt_text:
                        options.append(opt_text)

                answer_type = "choice" if options else "open_ended"

                records.append({
                    "id": f"pmc_vqa_{split}_{idx:05d}",
                    "dataset": "PMC-VQA",
                    "image_path": None,
                    "image_id": img_id,
                    "question": question,
                    "answer": answer,
                    "answer_type": answer_type,
                    "options": options if options else None,
                    "modality": "mixed",
                    "category": "medical_literature",
                    "difficulty": "hard",
                    "metadata": {"source": "local", "split": split},
                })

            return records
        except Exception as e:
            logger.warning(f"[PMC-VQA] Local CSV parse error: {e}")

    logger.warning("[PMC-VQA] No data found.")
    return records


# ══════════════════════════════════════════════════════════════
#  5. VQA-Med-2021 Loader
# ══════════════════════════════════════════════════════════════

def load_vqa_med_2021(
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load VQA-Med-2021 challenge dataset.

    Source: local data at evaluations/VQA-Med-2021/

    Returns:
        List of unified VQA dicts.
    """
    records = []

    eval_dir = _EVAL_DIR / "VQA-Med-2021" / "EvaluationCode" / "data" / "resources"

    # Load questions
    q_path = eval_dir / "Task1-VQAnswering2021-Test-ReferenceQuestions_mscoco_format_vqa.json"
    a_path = eval_dir / "Task1-VQAnswering2021-Test-ReferenceAnswers_mscoco_format_vqa.json"

    if not q_path.exists():
        logger.warning(f"[VQA-Med-2021] Question file not found: {q_path}")
        return records

    with open(q_path, "r", encoding="utf-8") as f:
        q_data = json.load(f)

    # Load answers if available
    a_map = {}
    if a_path.exists():
        with open(a_path, "r", encoding="utf-8") as f:
            a_data = json.load(f)
        # Build answer map: question_id -> answer
        for ann in a_data.get("annotations", []):
            qid = ann.get("question_id")
            answers = ann.get("answers", [])
            if answers:
                a_map[qid] = answers[0].get("answer", "") if isinstance(answers[0], dict) else str(answers[0])

    # Also try the text-format answers
    a_txt_path = eval_dir / "Task1-VQAnswering2021-Test-ReferenceAnswers-4Evaluator.txt"
    if a_txt_path.exists() and not a_map:
        with open(a_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "|" in line:
                    parts = line.split("|", 1)
                    if len(parts) == 2:
                        qid_str, answer = parts
                        try:
                            a_map[int(qid_str.strip())] = answer.strip()
                        except ValueError:
                            a_map[qid_str.strip()] = answer.strip()

    # Process questions
    questions = q_data.get("questions", [])
    logger.info(f"[VQA-Med-2021] Loaded {len(questions)} questions, {len(a_map)} answers")

    # Image directory — check if zip needs extraction
    img_zip = _EVAL_DIR / "VQA-Med-2021" / "Task1-VQA-2021-TestSet-w-GroundTruth.zip"
    img_dir = _EVAL_DIR / "VQA-Med-2021" / "VQA-Med-2021-TestSet-Images"

    if img_zip.exists() and not img_dir.exists():
        logger.info("[VQA-Med-2021] Extracting test images from zip...")
        try:
            with zipfile.ZipFile(img_zip, "r") as zf:
                zf.extractall(_EVAL_DIR / "VQA-Med-2021")
        except Exception as e:
            logger.warning(f"[VQA-Med-2021] Zip extraction failed: {e}")

    for idx, q_item in enumerate(questions):
        if max_samples and idx >= max_samples:
            break

        qid = q_item.get("question_id", idx)
        question = str(q_item.get("question", "")).strip()
        image_id = str(q_item.get("image_id", f"vqamed_{idx}"))

        answer = str(a_map.get(qid, a_map.get(str(qid), ""))).strip()
        answer_type = _classify_answer_type(answer, question)

        # Find image
        image_path = None
        if img_dir.exists():
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = img_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_path = str(candidate)
                    break

        records.append({
            "id": f"vqamed2021_{idx:05d}",
            "dataset": "VQA-Med-2021",
            "image_path": image_path,
            "image_id": image_id,
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
            "options": None,
            "modality": "radiology",
            "category": "medical_vqa_challenge",
            "difficulty": "hard",
            "metadata": {
                "source": "local",
                "question_id": qid,
            },
        })

    return records


# ══════════════════════════════════════════════════════════════
#  6. Quilt-VQA Loader
# ══════════════════════════════════════════════════════════════

def load_quilt_vqa(
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load Quilt-VQA (Histopathology VQA) dataset.

    Primary source: local data at evaluations/quilt-llava/
    Fallback: HuggingFace (wisdomik/Quilt_VQA)

    Returns:
        List of unified VQA dicts.
    """
    records = []

    # Try HuggingFace first (may include images)
    try:
        from datasets import load_dataset
        ds = load_dataset("wisdomik/Quilt_VQA", split="test", trust_remote_code=True)
        logger.info(f"[Quilt-VQA] Loaded {len(ds)} samples from HuggingFace")

        image_dir = _DATA_DIR / "quilt_vqa" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(ds):
            if max_samples and idx >= max_samples:
                break

            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            answer_type = _classify_answer_type(answer, question)

            # Save image locally if available
            image_path = image_dir / f"quilt_{idx:05d}.png"
            if hasattr(item.get("image", None), "save") and not image_path.exists():
                try:
                    item["image"].save(str(image_path))
                except Exception:
                    image_path = None

            records.append({
                "id": f"quilt_vqa_{idx:05d}",
                "dataset": "Quilt-VQA",
                "image_path": str(image_path) if image_path and image_path.exists() else None,
                "image_id": f"quilt_hf_{idx}",
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "options": None,
                "modality": "pathology",
                "category": "histopathology",
                "difficulty": "hard",
                "metadata": {"source": "HuggingFace"},
            })

        return records

    except Exception as e:
        logger.warning(f"[Quilt-VQA] HuggingFace loading failed: {e}")

    # Fallback: local JSONL
    local_jsonl = _EVAL_DIR / "quilt-llava" / "playground" / "data" / "quilt_gpt" / "quilt_gpt_questions.jsonl"

    if local_jsonl.exists():
        items = []
        with open(local_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        logger.info(f"[Quilt-VQA] Loaded {len(items)} samples from local JSONL")

        answers_jsonl = _EVAL_DIR / "quilt-llava" / "playground" / "data" / "quilt_gpt" / "quilt_gpt_answers.jsonl"
        answer_map = {}
        if answers_jsonl.exists():
            with open(answers_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            a_item = json.loads(line)
                            qid = a_item.get("question_id", a_item.get("id"))
                            answer_map[qid] = a_item.get("text", a_item.get("answer", ""))
                        except json.JSONDecodeError:
                            continue

        for idx, item in enumerate(items):
            if max_samples and idx >= max_samples:
                break

            qid = item.get("question_id", item.get("id", idx))
            question = str(item.get("text", item.get("question", ""))).strip()
            image_id = str(item.get("image", f"quilt_{idx}"))
            answer = str(answer_map.get(qid, "")).strip()
            answer_type = _classify_answer_type(answer, question)

            records.append({
                "id": f"quilt_vqa_{idx:05d}",
                "dataset": "Quilt-VQA",
                "image_path": None,
                "image_id": image_id,
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "options": None,
                "modality": "pathology",
                "category": "histopathology",
                "difficulty": "hard",
                "metadata": {"source": "local", "question_id": qid},
            })

        return records

    logger.warning("[Quilt-VQA] No data found.")
    return records


# ══════════════════════════════════════════════════════════════
#  Unified Interface
# ══════════════════════════════════════════════════════════════

# Registry of all VQA dataset loaders
VQA_DATASET_REGISTRY = {
    "vqa_rad": {
        "loader": load_vqa_rad,
        "name": "VQA-RAD",
        "description": "Radiology Visual QA",
        "modality": "radiology",
        "source": "HuggingFace: flaviagiammarino/vqa-rad",
    },
    "slake": {
        "loader": load_slake,
        "name": "SLAKE",
        "description": "Semantically-Labeled Knowledge-Enhanced Medical VQA",
        "modality": "mixed",
        "source": "Local + HuggingFace: BoKelvin/SLAKE",
    },
    "pathvqa": {
        "loader": load_pathvqa,
        "name": "PathVQA",
        "description": "Pathology Visual QA",
        "modality": "pathology",
        "source": "Local + HuggingFace: flaviagiammarino/path-vqa",
    },
    "pmc_vqa": {
        "loader": load_pmc_vqa,
        "name": "PMC-VQA",
        "description": "PubMedCentral Visual QA",
        "modality": "mixed",
        "source": "HuggingFace: RadGenome/PMC-VQA",
    },
    "vqa_med_2021": {
        "loader": load_vqa_med_2021,
        "name": "VQA-Med-2021",
        "description": "Medical VQA Challenge 2021",
        "modality": "radiology",
        "source": "Local: evaluations/VQA-Med-2021/",
    },
    "quilt_vqa": {
        "loader": load_quilt_vqa,
        "name": "Quilt-VQA",
        "description": "Histopathology VQA",
        "modality": "pathology",
        "source": "Local + HuggingFace: wisdomik/Quilt_VQA",
    },
}


def load_all_vqa_datasets(
    datasets: Optional[list[str]] = None,
    max_per_dataset: Optional[int] = None,
    split: str = "test",
) -> list[dict]:
    """Load all (or selected) VQA datasets into unified format.

    Args:
        datasets: List of dataset keys to load. None = all available.
        max_per_dataset: Max samples per dataset.
        split: Data split ("train", "test", "val").

    Returns:
        Unified list of VQA records.
    """
    if datasets is None:
        datasets = list(VQA_DATASET_REGISTRY.keys())

    all_records = []
    for ds_key in datasets:
        if ds_key not in VQA_DATASET_REGISTRY:
            logger.warning(f"Unknown VQA dataset: {ds_key}")
            continue

        info = VQA_DATASET_REGISTRY[ds_key]
        loader = info["loader"]
        logger.info(f"\n{'─'*50}")
        logger.info(f"Loading {info['name']} ({ds_key})...")

        try:
            # VQA-Med-2021 and Quilt-VQA don't have split parameter
            if ds_key in ("vqa_med_2021", "quilt_vqa"):
                records = loader(max_samples=max_per_dataset)
            else:
                records = loader(max_samples=max_per_dataset, split=split)

            all_records.extend(records)
            logger.info(f"  -> {len(records)} samples loaded")

        except Exception as e:
            logger.error(f"  Error loading {ds_key}: {e}")

    logger.info(f"\n{'='*50}")
    logger.info(f"Total VQA samples: {len(all_records)} from {len(datasets)} datasets")
    return all_records


def get_vqa_stats(records: list[dict]) -> dict:
    """Get comprehensive statistics for VQA records.

    Args:
        records: List of unified VQA dicts.

    Returns:
        Statistics dict.
    """
    if not records:
        return {"total": 0}

    by_dataset = Counter(r["dataset"] for r in records)
    by_modality = Counter(r["modality"] for r in records)
    by_answer_type = Counter(r["answer_type"] for r in records)
    by_category = Counter(r["category"] for r in records)
    by_difficulty = Counter(r["difficulty"] for r in records)
    has_image = sum(1 for r in records if r["image_path"])

    return {
        "total": len(records),
        "by_dataset": dict(by_dataset),
        "by_modality": dict(by_modality),
        "by_answer_type": dict(by_answer_type),
        "by_category": dict(by_category),
        "by_difficulty": dict(by_difficulty),
        "with_images": has_image,
        "without_images": len(records) - has_image,
    }


def save_unified_vqa(
    records: list[dict],
    output_path: Optional[str] = None,
) -> str:
    """Save unified VQA records to JSON.

    Args:
        records: List of unified VQA dicts.
        output_path: Optional output path. Default: datasets/vqa/unified_vqa.json

    Returns:
        Path to saved file.
    """
    if output_path is None:
        output_path = str(_DATA_DIR / "unified_vqa.json")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(records)} VQA records to {out}")
    return str(out)


# ══════════════════════════════════════════════════════════════
#  Helper Functions
# ══════════════════════════════════════════════════════════════

def _classify_answer_type(answer: str, question: str = "") -> str:
    """Classify the answer type based on answer content and question text."""
    answer_lower = answer.lower().strip()
    question_lower = question.lower().strip()

    # Yes/No
    if answer_lower in ("yes", "no", "true", "false"):
        return "yes_no"

    # Number
    if answer_lower.replace(".", "").replace("-", "").isdigit():
        return "number"

    # Question-based heuristics
    if question_lower.startswith(("is ", "are ", "does ", "do ", "was ", "were ", "can ", "could ")):
        if answer_lower in ("yes", "no"):
            return "yes_no"

    if "how many" in question_lower or "how much" in question_lower:
        return "number"

    return "open_ended"


def _categorize_radiology_question(question: str) -> str:
    """Categorize a radiology question by anatomical region or finding type."""
    q = question.lower()

    if any(w in q for w in ["chest", "lung", "pulmonary", "cardiac", "heart"]):
        return "chest"
    if any(w in q for w in ["brain", "head", "cerebral", "intracranial"]):
        return "neuroradiology"
    if any(w in q for w in ["abdomen", "liver", "kidney", "spleen", "bowel"]):
        return "abdominal"
    if any(w in q for w in ["bone", "fracture", "joint", "spine"]):
        return "musculoskeletal"
    if any(w in q for w in ["breast", "mammogra"]):
        return "breast"

    return "general"


def _slake_content_to_modality(content_type: str, item: dict = None) -> str:
    """Map SLAKE content_type to standard modality."""
    ct = content_type.lower()

    if any(w in ct for w in ["xray", "x-ray", "radiograph"]):
        return "radiology"
    if any(w in ct for w in ["ct", "computed"]):
        return "ct"
    if any(w in ct for w in ["mri", "magnetic"]):
        return "mri"
    if any(w in ct for w in ["pathol", "histol", "biopsy"]):
        return "pathology"

    # Check image name if available
    if item:
        img_name = str(item.get("img_name", "")).lower()
        if "xr" in img_name or "chest" in img_name:
            return "radiology"
        if "ct" in img_name:
            return "ct"
        if "mri" in img_name or "mr_" in img_name:
            return "mri"

    return "radiology"  # default for SLAKE


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and unify VQA datasets")
    parser.add_argument(
        "--datasets", nargs="+",
        default=None,
        choices=list(VQA_DATASET_REGISTRY.keys()) + [None],
        help="Datasets to load (default: all available)",
    )
    parser.add_argument("--max-per-dataset", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--stats-only", action="store_true", help="Only print stats")
    args = parser.parse_args()

    records = load_all_vqa_datasets(
        datasets=args.datasets,
        max_per_dataset=args.max_per_dataset,
        split=args.split,
    )

    stats = get_vqa_stats(records)
    print("\n=== VQA Dataset Statistics ===")
    print(json.dumps(stats, indent=2))

    if not args.stats_only and records:
        path = save_unified_vqa(records, args.output)
        print(f"\nSaved to: {path}")
