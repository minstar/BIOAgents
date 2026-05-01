"""Text-only benchmark evaluation (no tools, no agent loop).

Evaluates the base model on standard QA benchmarks without any tool access.
This provides the pure model capability baseline for the ablation:
  Base (text-only) → Base+AR (tools) → RL models (tools+RL)

Usage:
    python eval_benchmark_textonly.py \
        --model_path checkpoints/models/Qwen3.5-9B \
        --benchmarks medqa mmlu medmcqa \
        --output-dir results/benchmarks_textonly/base_qwen35_9b
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reuse benchmark file paths from multiturn eval (same source data)
BENCHMARK_FILES = {
    "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
    "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
    "mmlu": "evaluations/self-biorag/data/benchmark/mmlu_test.jsonl",
    "mmlu_anatomy": "evaluations/self-biorag/data/benchmark/mmlu_anatomy_test.jsonl",
    "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_clinical_knowledge_test.jsonl",
    "mmlu_professional": "evaluations/self-biorag/data/benchmark/mmlu_professional_medicine_test.jsonl",
    "mmlu_genetics": "evaluations/self-biorag/data/benchmark/mmlu_medical_genetics_test.jsonl",
    "mmlu_biology": "evaluations/self-biorag/data/benchmark/mmlu_college_biology_test.jsonl",
    "mmlu_college_med": "evaluations/self-biorag/data/benchmark/mmlu_college_medicine_test.jsonl",
    "kqa_golden": "evaluations/OLAPH/MedLFQA/kqa_golden_test_MedLFQA.jsonl",
    "live_qa": "evaluations/OLAPH/MedLFQA/live_qa_test_MedLFQA.jsonl",
    "medication_qa": "evaluations/OLAPH/MedLFQA/medication_qa_test_MedLFQA.jsonl",
    "healthsearch_qa": "evaluations/OLAPH/MedLFQA/healthsearch_qa_test_MedLFQA.jsonl",
    "kqa_silver": "evaluations/OLAPH/MedLFQA/kqa_silver_wogold_test_MedLFQA.jsonl",
    # VQA benchmarks not supported in text-only mode
}

MEDLFQA_BENCHMARKS = {"kqa_golden", "live_qa", "medication_qa", "healthsearch_qa", "kqa_silver"}

SYSTEM_PROMPT = (
    "You are a medical AI assistant. Answer the following medical question. "
    "For multiple choice questions, reason through the options and end with "
    "'Answer: X' where X is the letter of your chosen answer. "
    "For open-ended questions, provide a concise evidence-based answer."
)


def load_benchmark(name: str) -> list[dict]:
    """Load benchmark from JSONL."""
    filepath = PROJECT_ROOT / BENCHMARK_FILES[name]
    if not filepath.exists():
        logger.error(f"Benchmark file not found: {filepath}")
        return []

    tasks = []
    with open(filepath) as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)

            if name in MEDLFQA_BENCHMARKS:
                question = item.get("Question", "")
                answer = item.get("Free_form_answer", "").strip()
                must_have = item.get("Must_have", [])
                nice_to_have = item.get("Nice_to_have", [])
                has_options = False
            else:
                instances = item.get("instances", {})
                question = instances.get("input", "")
                answer = instances.get("output", "").strip()
                must_have = []
                nice_to_have = []
                has_options = bool(re.search(r"Option [A-E]:", question))

            # Extract options and map full-text answer to a letter
            gt_letter = ""
            if has_options:
                for letter in "ABCDE":
                    pat = rf"Option {letter}:\s*(.+?)(?=Option [A-E]:|$)"
                    m = re.search(pat, question, re.DOTALL)
                    if m:
                        opt_text = re.sub(r'\s*Option:\s*$', '', m.group(1).strip(), flags=re.DOTALL).strip()
                        if opt_text.lower() == answer.lower():
                            gt_letter = letter
                            break

            tasks.append({
                "id": f"{name}_{idx}",
                "question": question,
                "answer": answer,
                "gt_letter": gt_letter,
                "has_options": has_options,
                "must_have": must_have,
                "nice_to_have": nice_to_have,
            })

    logger.info(f"Loaded {len(tasks)} tasks from {name}")
    return tasks


def extract_answer_letter(text: str):
    """Extract answer letter from model response."""
    matches = re.findall(r"Answer:\s*([A-E])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"the answer is\s*([A-E])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    match = re.search(r"\(?([A-E])\)?\s*$", text.strip())
    if match:
        return match.group(1).upper()
    return None


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Simple ROUGE-L (longest common subsequence)."""
    pred_words = prediction.lower().split()
    ref_words = reference.lower().split()
    if not pred_words or not ref_words:
        return 0.0

    m, n = len(ref_words), len(pred_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == pred_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0
    recall = lcs / m if m else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_hallucination(prediction: str, must_have: list) -> float:
    """Check hallucination rate (proportion of must_have items NOT in prediction)."""
    if not must_have:
        return 0.0
    pred_lower = prediction.lower()
    missed = sum(1 for item in must_have if item.lower() not in pred_lower)
    return missed / len(must_have) * 100


def compute_completeness(prediction: str, must_have: list) -> float:
    """Check completeness (proportion of must_have items found in prediction)."""
    if not must_have:
        return 100.0
    pred_lower = prediction.lower()
    found = sum(1 for item in must_have if item.lower() in pred_lower)
    return found / len(must_have) * 100


def save_partial(benchmark_name, results, correct, total, output_dir, is_lfqa=False):
    """Save partial results for resumability."""
    partial = {
        "benchmark": benchmark_name,
        "correct": correct,
        "total": total,
        "results": results,
    }
    if is_lfqa:
        rouge_scores = [r["rouge_l"] for r in results if "rouge_l" in r]
        partial["avg_rouge_l"] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    else:
        partial["accuracy"] = correct / max(total, 1)
    path = output_dir / f"{benchmark_name}_partial.json"
    with open(path, "w") as f:
        json.dump(partial, f, indent=2, ensure_ascii=False)


def run_benchmark(
    benchmark_name: str,
    tasks: list[dict],
    model,
    tokenizer,
    output_dir: Path,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    resume_from: int = 0,
):
    """Run text-only evaluation on a benchmark."""
    is_lfqa = benchmark_name in MEDLFQA_BENCHMARKS
    results = []
    correct = 0
    total = 0

    # Load partial results for resuming
    partial_path = output_dir / f"{benchmark_name}_partial.json"
    if resume_from == 0 and partial_path.exists():
        try:
            prev = json.load(open(partial_path))
            results = prev.get("results", [])
            correct = prev.get("correct", 0)
            total = len(results)
            resume_from = total
            logger.info(f"Resuming from {resume_from} (partial file)")
        except Exception:
            pass

    t_start = time.time()

    for i, task in enumerate(tasks):
        if i < resume_from:
            continue

        question = task["question"]
        gt_answer = task["answer"]

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        # Tokenize
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                top_p=0.95,
            )

        # Decode only the generated tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        # Score
        total += 1
        result = {
            "id": task["id"],
            "question": question[:200],
            "gt_answer": gt_answer,
            "prediction": response[:500],
        }

        if is_lfqa:
            rouge = compute_rouge_l(response, gt_answer)
            hall = compute_hallucination(response, task.get("must_have", []))
            comp = compute_completeness(response, task.get("must_have", []))
            result["rouge_l"] = rouge
            result["hallucination"] = hall
            result["completeness"] = comp
            result["correct"] = rouge > 0.2  # threshold for "correct"
            if result["correct"]:
                correct += 1
        else:
            predicted = extract_answer_letter(response)
            gt_letter = task.get("gt_letter", "")
            if not gt_letter:
                # Fallback: try to extract letter from gt_answer directly
                gt_upper = gt_answer.strip().upper()
                gt_match = re.search(r"(?:ANSWER:\s*)?[\(]([A-E])[\)]", gt_upper, re.IGNORECASE)
                if gt_match:
                    gt_letter = gt_match.group(1).upper()
                elif len(gt_upper) == 1 and gt_upper in "ABCDE":
                    gt_letter = gt_upper

            is_correct = predicted == gt_letter if (predicted and gt_letter) else False
            result["predicted"] = predicted
            result["gt_letter"] = gt_letter
            result["correct"] = is_correct
            if is_correct:
                correct += 1

        results.append(result)

        # Log progress every 10 samples
        if total % 10 == 0:
            elapsed = time.time() - t_start
            rate = total / (elapsed / 60) if elapsed > 0 else 0
            remaining = len(tasks) - i - 1
            eta = remaining / rate if rate > 0 else 0

            if is_lfqa:
                avg_rouge = sum(r["rouge_l"] for r in results) / len(results)
                avg_hall = sum(r.get("hallucination", 0) for r in results) / len(results)
                avg_comp = sum(r.get("completeness", 0) for r in results) / len(results)
                logger.info(
                    f"  [{benchmark_name}] {total}/{len(tasks)} "
                    f"rouge_l={avg_rouge:.3f} hall={avg_hall:.1f}% comp={avg_comp:.1f}% "
                    f"rate={rate:.1f}/min ETA={eta:.0f}min"
                )
            else:
                acc = correct / total
                logger.info(
                    f"  [{benchmark_name}] {total}/{len(tasks)} "
                    f"acc={acc:.3f} rate={rate:.1f}/min ETA={eta:.0f}min"
                )

            save_partial(benchmark_name, results, correct, total, output_dir, is_lfqa)

    # Final save
    save_partial(benchmark_name, results, correct, total, output_dir, is_lfqa)

    # Save completed results
    final_result = {
        "benchmark": benchmark_name,
        "total": total,
        "correct": correct,
        "results": results,
        "model_type": "text_only",
    }
    if is_lfqa:
        rouge_scores = [r["rouge_l"] for r in results]
        final_result["avg_rouge_l"] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    else:
        final_result["accuracy"] = correct / max(total, 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = output_dir / f"{benchmark_name}_textonly_{timestamp}.json"
    with open(final_path, "w") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    if is_lfqa:
        logger.info(
            f"{benchmark_name}: ROUGE-L={final_result['avg_rouge_l']:.3f} "
            f"({total} samples)"
        )
    else:
        logger.info(
            f"{benchmark_name}: accuracy={final_result['accuracy']:.1%} "
            f"({correct}/{total})"
        )

    return final_result


def main():
    parser = argparse.ArgumentParser(description="Text-only benchmark evaluation (no tools)")
    parser.add_argument("--model_path", required=True, help="Path to HF model")
    parser.add_argument("--benchmarks", nargs="+", default=["medqa"],
                        choices=list(BENCHMARK_FILES.keys()),
                        help="Benchmarks to evaluate")
    parser.add_argument("--output-dir", default="results/benchmarks_textonly",
                        help="Output directory")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max new tokens (default 1024)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per benchmark (0=all)")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume from sample index")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    logger.info("Model loaded successfully")

    for bench_name in args.benchmarks:
        if bench_name not in BENCHMARK_FILES:
            logger.warning(f"Unknown benchmark: {bench_name}, skipping")
            continue

        tasks = load_benchmark(bench_name)
        if not tasks:
            continue

        if args.max_samples > 0:
            tasks = tasks[:args.max_samples]

        logger.info(f"Evaluating {bench_name}: {len(tasks)} samples (text-only, no tools)")

        run_benchmark(
            bench_name, tasks, model, tokenizer,
            output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            resume_from=args.resume_from,
        )


if __name__ == "__main__":
    main()
