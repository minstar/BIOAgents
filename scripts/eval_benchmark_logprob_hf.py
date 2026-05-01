#!/usr/bin/env python3
"""
Log-probability based evaluation using HF transformers directly.

For models that can't run on SGLang (e.g., qwen3_next architecture),
this script loads the model via HF and computes logprobs locally.
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
}

SYSTEM_PROMPT = (
    "You are a medical AI assistant. Answer the following multiple-choice "
    "question by selecting the correct option letter (A, B, C, D, or E). "
    "Respond with only the letter of the correct answer."
)

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
# Qwen3.5 token IDs for A-E
OPTION_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}


def load_benchmark(benchmark_name: str, max_samples: int | None = None) -> list[dict]:
    """Load benchmark data from JSONL."""
    import re

    rel_path = BENCHMARK_FILES.get(benchmark_name)
    if rel_path is None:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: {list(BENCHMARK_FILES.keys())}"
        )

    file_path = PROJECT_ROOT / rel_path
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")

    samples = []
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            instances = item.get("instances", {})
            question = instances.get("input", "")
            answer = instances.get("output", "").strip()
            has_options = bool(re.search(r"Option [A-E]:", question))

            gt_letter = ""
            if has_options:
                for letter in OPTION_LETTERS:
                    pat = rf"Option {letter}:\s*(.+?)(?=Option [A-E]:|$)"
                    m = re.search(pat, question, re.DOTALL)
                    if m:
                        opt_text = re.sub(
                            r"\s*Option:\s*$", "", m.group(1).strip(), flags=re.DOTALL
                        ).strip()
                        if opt_text.lower() == answer.lower():
                            gt_letter = letter
                            break

            samples.append(
                {
                    "id": f"{benchmark_name}_{idx}",
                    "question": question,
                    "answer": answer,
                    "gt_letter": gt_letter,
                    "has_options": has_options,
                }
            )

            if max_samples is not None and len(samples) >= max_samples:
                break

    return samples


def evaluate_benchmark(
    benchmark_name: str,
    samples: list[dict],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
    option_token_ids: dict[str, int],
    batch_size: int = 8,
) -> dict[str, Any]:
    """Evaluate a benchmark using HF model with batched logprob computation."""
    start_time = time.time()
    results: list[dict] = []
    correct = 0
    total = 0

    # Pre-build all prompts
    prompts = []
    for sample in samples:
        question_text = sample["question"] + "\nAnswer:"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(prompt)

    # Process in batches
    for batch_start in range(0, len(samples), batch_size):
        batch_end = min(batch_start + batch_size, len(samples))
        batch_prompts = prompts[batch_start:batch_end]
        batch_samples = samples[batch_start:batch_end]

        # Tokenize batch with left padding for correct last-token logprobs
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # logits shape: [batch, seq_len, vocab_size]
            # Get logits at the last non-padding position for each sample
            logits = outputs.logits

        for i, sample in enumerate(batch_samples):
            # Find last non-padding token position
            attention_mask = inputs["attention_mask"][i]
            last_pos = attention_mask.sum().item() - 1
            token_logits = logits[i, last_pos, :]

            # Get log probabilities for option tokens
            log_probs = torch.log_softmax(token_logits.float(), dim=-1)
            option_logprobs = {}
            for letter, tid in option_token_ids.items():
                option_logprobs[letter] = log_probs[tid].item()

            predicted = max(option_logprobs, key=lambda k: option_logprobs[k])

            gt_letter = sample.get("gt_letter", "")
            if not gt_letter:
                answer = sample.get("answer", "").strip().upper()
                if len(answer) == 1 and answer in "ABCDE":
                    gt_letter = answer

            is_correct = predicted == gt_letter if (predicted and gt_letter) else False
            if is_correct:
                correct += 1
            total += 1

            serializable_logprobs = {
                k: v if math.isfinite(v) else -1000.0
                for k, v in option_logprobs.items()
            }

            results.append(
                {
                    "index": batch_start + i,
                    "question": sample["question"][:200],
                    "gold_answer": gt_letter,
                    "answer_text": sample.get("answer", "")[:100],
                    "predicted": predicted,
                    "correct": is_correct,
                    "logprobs": serializable_logprobs,
                }
            )

        # Log progress
        if total % 100 == 0 or total == len(samples):
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            accuracy = correct / total if total > 0 else 0
            remaining = len(samples) - total
            eta = remaining / rate if rate > 0 else 0
            print(
                f"  [{benchmark_name}] {total}/{len(samples)} | "
                f"Accuracy: {accuracy:.1%} ({correct}/{total}) | "
                f"Rate: {rate:.1f} samples/s | "
                f"ETA: {eta:.0f}s"
            )

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0

    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "elapsed_seconds": round(elapsed, 2),
        "samples_per_second": round(total / elapsed, 2) if elapsed > 0 else 0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HF-based log-probability evaluation for MC-QA benchmarks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to HF model",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Override tokenizer path (default: same as model_path)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARK_FILES.keys()),
        choices=list(BENCHMARK_FILES.keys()),
        help="Benchmarks to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/logprob_hf",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to load model on (default: cuda:0)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = args.tokenizer_path or args.model_path
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Verify option token IDs
    option_token_ids = {}
    for letter in OPTION_LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        option_token_ids[letter] = ids[0]
    print(f"Option token IDs: {option_token_ids}")

    print(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model_path,
        "method": "logprob_hf",
        "benchmarks": {},
    }

    for benchmark_name in args.benchmarks:
        print(f"\n=== Evaluating {benchmark_name} ===")
        try:
            samples = load_benchmark(benchmark_name, args.max_samples)
        except (FileNotFoundError, ValueError) as e:
            print(f"  Skipping {benchmark_name}: {e}")
            continue

        print(f"  Loaded {len(samples)} samples")

        benchmark_result = evaluate_benchmark(
            benchmark_name=benchmark_name,
            samples=samples,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            option_token_ids=option_token_ids,
            batch_size=args.batch_size,
        )

        accuracy = benchmark_result["accuracy"]
        elapsed = benchmark_result["elapsed_seconds"]
        rate = benchmark_result["samples_per_second"]
        print(
            f"  Final: {accuracy:.1%} "
            f"({benchmark_result['correct']}/{benchmark_result['total']}) "
            f"in {elapsed:.1f}s ({rate:.1f} samples/s)"
        )

        result_path = output_dir / f"{benchmark_name}_logprob.json"
        with open(result_path, "w") as f:
            json.dump(benchmark_result, f, indent=2)
        print(f"  Saved results to {result_path}")

        summary["benchmarks"][benchmark_name] = {
            "accuracy": accuracy,
            "correct": benchmark_result["correct"],
            "total": benchmark_result["total"],
            "elapsed_seconds": elapsed,
        }

    # Save summary
    summary_path = output_dir / "summary_logprob.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print final summary table
    print("\n" + "=" * 60)
    print(f"{'Benchmark':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 60)
    for name, data in summary["benchmarks"].items():
        print(
            f"{name:<15} {data['accuracy']:>9.1%} "
            f"{data['correct']:>10} {data['total']:>10}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
