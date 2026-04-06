#!/usr/bin/env python3
"""Run tracking evaluation on the compact multi-domain eval set.

Loads a model, runs inference on the tracking eval set, and reports
per-domain accuracy. Designed to complete in ~5-10 min on a single GPU.

Supports:
  - Qwen2.5-VL models (text-only mode for non-VQA, vision mode for VQA)
  - Qwen3 causal LMs
  - Any HuggingFace AutoModelForCausalLM

Usage:
    # Default: Qwen2.5-VL-7B-Instruct
    python scripts/run_tracking_eval.py --model qwen2vl

    # Qwen3 base model
    python scripts/run_tracking_eval.py --model qwen3

    # Custom model path
    python scripts/run_tracking_eval.py --model-path /path/to/model --model-name my_model

    # Specific domains only
    python scripts/run_tracking_eval.py --model qwen2vl --domains medqa mmlu_clinical

    # Custom eval set
    python scripts/run_tracking_eval.py --model qwen2vl --eval-set data/eval/tracking_eval_set.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────
#  Model registry
# ──────────────────────────────────────────────────────────────

MODELS = {
    "qwen3": {
        "name": "Qwen3-8B-Base",
        "path": "/data/project/private/minstar/models/Qwen3-8B-Base",
        "type": "causal",
        "supports_vision": False,
    },
    "qwen2vl": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "path": str(PROJECT_ROOT / "checkpoints" / "models" / "Qwen2.5-VL-7B-Instruct"),
        "type": "vlm",
        "supports_vision": True,
    },
    "lingshu": {
        "name": "Lingshu-7B",
        "path": str(PROJECT_ROOT / "checkpoints" / "models" / "Lingshu-7B"),
        "type": "vlm",
        "supports_vision": True,
    },
}


# ──────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────

def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.lower().strip()
    # Remove trailing periods
    text = text.rstrip(".")
    # Remove common prefixes
    for prefix in ["the answer is ", "answer: ", "option "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def mcq_match(pred: str, gold: str, options: Optional[List[str]] = None) -> bool:
    """Check if prediction matches gold answer for MCQ.

    Handles: exact match, option letter match, substring match.
    """
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # Exact match
    if pred_norm == gold_norm:
        return True

    # Check if gold answer text appears in prediction
    if gold_norm and gold_norm in pred_norm:
        return True

    # Try matching option letter (A/B/C/D/E)
    if options:
        gold_letter = None
        for opt in options:
            opt_parts = opt.split(":", 1)
            if len(opt_parts) == 2:
                letter = opt_parts[0].strip().upper()
                opt_text = normalize_answer(opt_parts[1])
                if opt_text == gold_norm or gold_norm in opt_text:
                    gold_letter = letter
                    break

        if gold_letter:
            # Check if prediction starts with or contains the correct letter
            pred_upper = pred.strip().upper()
            if pred_upper.startswith(gold_letter):
                return True
            # Check for pattern like "(A)" or "A."
            letter_patterns = [
                f"({gold_letter})",
                f"{gold_letter}.",
                f"{gold_letter}:",
                f"answer is {gold_letter}",
            ]
            pred_lower = pred.lower()
            for pat in letter_patterns:
                if pat.lower() in pred_lower:
                    return True

    return False


def open_ended_match(pred: str, gold: str) -> bool:
    """Check if prediction matches gold for open-ended (VQA) questions."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    # For yes/no questions
    if gold_norm in ("yes", "no"):
        return gold_norm in pred_norm.split()[:5]

    # Substring containment (both directions)
    if gold_norm and len(gold_norm) > 2:
        if gold_norm in pred_norm or pred_norm in gold_norm:
            return True

    return False


def compute_rouge_l(pred: str, ref: str) -> float:
    """Compute ROUGE-L F1 between prediction and reference."""
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


def compute_must_have_score(pred: str, must_have: List[str]) -> float:
    """Score based on how many must_have items are mentioned."""
    if not must_have:
        return 1.0
    pred_lower = pred.lower()
    covered = 0
    for item in must_have:
        words = [w for w in item.lower().split() if len(w) > 3]
        if words and sum(1 for w in words if w in pred_lower) / len(words) >= 0.5:
            covered += 1
        elif not words and item.lower() in pred_lower:
            covered += 1
    return covered / len(must_have)


# ──────────────────────────────────────────────────────────────
#  Prompt construction
# ──────────────────────────────────────────────────────────────

def build_mcq_prompt(sample: Dict[str, Any], is_chat: bool) -> str:
    """Build prompt for MCQ tasks."""
    question = sample["question"]
    options = sample.get("options", [])
    options_str = "\n".join(options) if options else ""

    prompt_body = (
        f"Answer the following medical question by selecting the best option.\n\n"
        f"Question: {question}\n"
    )
    if options_str:
        prompt_body += f"\n{options_str}\n"
    prompt_body += "\nAnswer with ONLY the correct option text. Do not explain."

    if is_chat:
        return prompt_body
    return prompt_body


def build_vqa_prompt(sample: Dict[str, Any], is_chat: bool) -> str:
    """Build prompt for VQA tasks (text-only mode)."""
    question = sample["question"]
    prompt_body = (
        f"Answer the following medical visual question as concisely as possible.\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt_body


def build_ehr_prompt(sample: Dict[str, Any], is_chat: bool) -> str:
    """Build prompt for EHR management tasks."""
    question = sample["question"]
    prompt_body = (
        f"You are a clinical AI assistant. Analyze the following clinical scenario "
        f"and provide a thorough assessment.\n\n"
        f"Task: {question}\n\n"
        f"Provide a concise clinical assessment:"
    )
    return prompt_body


def build_lfqa_prompt(sample: Dict[str, Any], is_chat: bool) -> str:
    """Build prompt for long-form QA tasks."""
    question = sample["question"]
    prompt_body = (
        f"Answer the following medical question thoroughly but concisely.\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt_body


def build_agentic_prompt(sample: Dict[str, Any], is_chat: bool) -> str:
    """Build prompt for agentic clinical tasks (clinical_diagnosis, drug_interaction, triage)."""
    question = sample["question"]
    prompt_body = (
        f"You are a clinical AI assistant. Analyze the following clinical scenario "
        f"and provide a thorough step-by-step assessment with your reasoning.\n\n"
        f"Scenario: {question}\n\n"
        f"Provide your clinical assessment:"
    )
    return prompt_body


DOMAIN_PROMPT_BUILDERS = {
    "medqa": build_mcq_prompt,
    "medmcqa": build_mcq_prompt,
    "mmlu_clinical": build_mcq_prompt,
    "mmlu_anatomy": build_mcq_prompt,
    "mmlu_college_medicine": build_mcq_prompt,
    "mmlu_professional_medicine": build_mcq_prompt,
    "mmlu_medical_genetics": build_mcq_prompt,
    "mmlu_college_biology": build_mcq_prompt,
    "vqa_rad": build_vqa_prompt,
    "pathvqa": build_vqa_prompt,
    "slake": build_vqa_prompt,
    "ehr": build_ehr_prompt,
    "clinical_diagnosis": build_agentic_prompt,
    "drug_interaction": build_agentic_prompt,
    "triage_emergency": build_agentic_prompt,
    "medlfqa": build_lfqa_prompt,
}

MCQ_DOMAINS = {
    "medqa", "medmcqa", "mmlu_clinical",
    "mmlu_anatomy", "mmlu_college_medicine", "mmlu_professional_medicine",
    "mmlu_medical_genetics", "mmlu_college_biology",
}
VQA_DOMAINS = {"vqa_rad", "pathvqa", "slake"}
LFQA_DOMAINS = {"medlfqa", "ehr"}
AGENTIC_DOMAINS = {"clinical_diagnosis", "drug_interaction", "triage_emergency"}


# ──────────────────────────────────────────────────────────────
#  Inference engine
# ──────────────────────────────────────────────────────────────

def load_model(model_key: Optional[str] = None, model_path: Optional[str] = None):
    """Load model and tokenizer. Returns (model, tokenizer, model_info)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if model_key and model_key in MODELS:
        info = MODELS[model_key]
        path = info["path"]
        name = info["name"]
        is_vlm = info.get("supports_vision", False)
    elif model_path:
        path = model_path
        name = Path(model_path).name
        is_vlm = False
        info = {"name": name, "path": path, "type": "causal", "supports_vision": False}
    else:
        raise ValueError("Provide --model (registry key) or --model-path")

    print(f"Loading model: {name} from {path}")

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl")

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }

    if is_qwen_vl:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **load_kwargs)
            info["supports_vision"] = True
            info["type"] = "vlm"
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Detect if chat model
    is_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    info["is_chat"] = is_chat

    print(f"Model loaded. Chat mode: {is_chat}, VLM: {info.get('supports_vision', False)}")
    return model, tokenizer, info


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 8,
    is_chat: bool = False,
) -> List[str]:
    """Run batch inference with left-padding."""
    tokenizer.padding_side = "left"
    all_responses = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        if is_chat:
            # Format as chat messages
            formatted = []
            for p in batch_prompts:
                messages = [{"role": "user", "content": p}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted.append(text)
            batch_prompts = formatted

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            generated = output[input_len:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            all_responses.append(response)

    return all_responses


# ──────────────────────────────────────────────────────────────
#  Evaluation logic
# ──────────────────────────────────────────────────────────────

def evaluate_domain(
    domain: str,
    samples: List[Dict[str, Any]],
    model,
    tokenizer,
    model_info: Dict[str, Any],
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Evaluate a single domain and return metrics."""
    if not samples:
        return {"n": 0, "accuracy": 0.0, "skipped": True}

    is_chat = model_info.get("is_chat", False)
    prompt_builder = DOMAIN_PROMPT_BUILDERS.get(domain, build_mcq_prompt)

    # Determine max_new_tokens per domain
    if domain in LFQA_DOMAINS:
        max_new_tokens = 512
    elif domain in VQA_DOMAINS:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    # Build prompts
    prompts = [prompt_builder(s, is_chat) for s in samples]

    # Run inference
    t0 = time.time()
    predictions = batch_generate(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        is_chat=is_chat,
    )
    elapsed = time.time() - t0

    # Score
    results_detail = []
    correct = 0

    for sample, pred in zip(samples, predictions):
        gold = sample.get("answer", "")
        options = sample.get("options")

        if domain in MCQ_DOMAINS:
            is_correct = mcq_match(pred, gold, options)
            score = 1.0 if is_correct else 0.0
        elif domain in VQA_DOMAINS:
            is_correct = open_ended_match(pred, gold)
            score = 1.0 if is_correct else 0.0
        elif domain == "medlfqa":
            must_have = sample.get("must_have", [])
            rouge = compute_rouge_l(pred, gold)
            mh_score = compute_must_have_score(pred, must_have)
            score = 0.5 * rouge + 0.5 * mh_score
            is_correct = score >= 0.3
        elif domain == "ehr":
            rouge = compute_rouge_l(pred, gold)
            rubric = sample.get("rubric", {})
            must_mention = rubric.get("must_mention", [])
            mh_score = compute_must_have_score(pred, must_mention) if must_mention else 0.0
            score = 0.5 * rouge + 0.5 * mh_score if must_mention else rouge
            is_correct = score >= 0.3
        elif domain in AGENTIC_DOMAINS:
            # Agentic tasks: score by nl_assertion coverage
            assertions = [a.strip() for a in gold.split(";") if a.strip()]
            mh_score = compute_must_have_score(pred, assertions) if assertions else 0.0
            rouge = compute_rouge_l(pred, gold)
            score = 0.6 * mh_score + 0.4 * rouge
            is_correct = score >= 0.3
        else:
            is_correct = normalize_answer(pred) == normalize_answer(gold)
            score = 1.0 if is_correct else 0.0

        if is_correct:
            correct += 1

        results_detail.append({
            "id": sample.get("id", ""),
            "question": sample.get("question", "")[:100],
            "gold": gold[:200] if domain in LFQA_DOMAINS else gold,
            "pred": pred[:200] if domain in LFQA_DOMAINS else pred,
            "score": round(score, 4),
            "correct": is_correct,
        })

    accuracy = correct / len(samples) if samples else 0.0

    return {
        "n": len(samples),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "elapsed_sec": round(elapsed, 1),
        "details": results_detail,
    }


def run_tracking_eval(
    model_key: Optional[str] = None,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    eval_set_path: Optional[str] = None,
    domains: Optional[List[str]] = None,
    batch_size: int = 8,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full tracking evaluation."""

    # Load eval set
    if eval_set_path is None:
        eval_set_path = str(PROJECT_ROOT / "data" / "eval" / "tracking_eval_set.json")

    if not Path(eval_set_path).exists():
        print(f"ERROR: Eval set not found at {eval_set_path}", file=sys.stderr)
        print("Run `python scripts/build_tracking_eval_set.py` first.", file=sys.stderr)
        sys.exit(1)

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"Loaded eval set from {eval_set_path}")
    total_samples = sum(len(v) for v in eval_set.values())
    print(f"Total samples: {total_samples}")

    # Filter domains if specified
    if domains:
        eval_set = {k: v for k, v in eval_set.items() if k in domains}
        print(f"Filtered to domains: {list(eval_set.keys())}")

    # Load model
    model, tokenizer, model_info = load_model(model_key=model_key, model_path=model_path)
    if model_name:
        model_info["name"] = model_name

    # Run evaluation per domain
    print(f"\n{'=' * 60}")
    print(f"  Tracking Evaluation: {model_info['name']}")
    print(f"{'=' * 60}\n")

    all_results = {}
    total_correct = 0
    total_n = 0
    total_time = 0.0

    for domain, samples in eval_set.items():
        print(f"Evaluating {domain} ({len(samples)} samples)...", end=" ", flush=True)
        result = evaluate_domain(
            domain, samples, model, tokenizer, model_info,
            batch_size=batch_size,
        )
        all_results[domain] = result
        total_correct += result.get("correct", 0)
        total_n += result["n"]
        total_time += result.get("elapsed_sec", 0)

        acc_str = f"{result['accuracy'] * 100:.1f}%"
        print(f"accuracy={acc_str}  ({result.get('correct', 0)}/{result['n']})  "
              f"[{result.get('elapsed_sec', 0):.1f}s]")

    overall_accuracy = total_correct / total_n if total_n > 0 else 0.0

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: {model_info['name']}")
    print(f"{'=' * 60}")
    print(f"{'Domain':<20} {'N':>5} {'Correct':>8} {'Accuracy':>10} {'Time':>8}")
    print(f"{'-' * 51}")
    for domain, result in all_results.items():
        acc_str = f"{result['accuracy'] * 100:.1f}%"
        print(f"{domain:<20} {result['n']:>5} {result.get('correct', 0):>8} {acc_str:>10} "
              f"{result.get('elapsed_sec', 0):>7.1f}s")
    print(f"{'-' * 51}")
    print(f"{'OVERALL':<20} {total_n:>5} {total_correct:>8} "
          f"{overall_accuracy * 100:.1f}%{' ':>3} {total_time:>7.1f}s")
    print()

    # Build report
    report = {
        "model_name": model_info["name"],
        "model_path": model_info.get("path", ""),
        "timestamp": datetime.now().isoformat(),
        "eval_set_path": eval_set_path,
        "total_samples": total_n,
        "overall_accuracy": round(overall_accuracy, 4),
        "total_time_sec": round(total_time, 1),
        "per_domain": {},
    }
    for domain, result in all_results.items():
        report["per_domain"][domain] = {
            "n": result["n"],
            "accuracy": result["accuracy"],
            "correct": result.get("correct", 0),
            "elapsed_sec": result.get("elapsed_sec", 0),
        }

    # Save report
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "eval" / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_info["name"])
    report_path = Path(output_dir) / f"tracking_eval_{safe_name}_{timestamp}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {report_path}")

    # Also save detailed results (with predictions)
    detail_path = Path(output_dir) / f"tracking_eval_{safe_name}_{timestamp}_detail.json"
    detail_report = dict(report)
    detail_report["per_domain_detail"] = {
        domain: result.get("details", []) for domain, result in all_results.items()
    }
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail_report, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to: {detail_path}")

    return report


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tracking evaluation on compact multi-domain eval set"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODELS.keys()),
        help="Model registry key (qwen2vl, qwen3, lingshu)",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Custom model path (overrides --model)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Custom model name for reporting",
    )
    parser.add_argument(
        "--eval-set", type=str, default=None,
        help="Path to tracking eval set JSON",
    )
    parser.add_argument(
        "--domains", nargs="+", default=None,
        help="Specific domains to evaluate (e.g. medqa mmlu_clinical)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results",
    )
    args = parser.parse_args()

    if args.model is None and args.model_path is None:
        parser.error("Provide --model or --model-path")

    run_tracking_eval(
        model_key=args.model,
        model_path=args.model_path,
        model_name=args.model_name,
        eval_set_path=args.eval_set,
        domains=args.domains,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
