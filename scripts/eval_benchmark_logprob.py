#!/usr/bin/env python3
"""
Log-probability based evaluation for multiple-choice QA benchmarks.

Uses SGLang server via OpenAI-compatible API to compute log probabilities
for each answer option (A-E) as the next token, selecting the highest.

Each sample requires only 1 API call with max_tokens=1, making this
significantly faster than generation-based evaluation.
"""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


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


def load_benchmark(benchmark_name: str, max_samples: int | None = None) -> list[dict]:
    """Load benchmark data from JSONL (same format as eval_benchmark_textonly.py)."""
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

            # Extract options and map full-text answer to a letter
            gt_letter = ""
            if has_options:
                for letter in OPTION_LETTERS:
                    pat = rf"Option {letter}:\s*(.+?)(?=Option [A-E]:|$)"
                    m = re.search(pat, question, re.DOTALL)
                    if m:
                        opt_text = re.sub(r'\s*Option:\s*$', '', m.group(1).strip(), flags=re.DOTALL).strip()
                        if opt_text.lower() == answer.lower():
                            gt_letter = letter
                            break

            samples.append({
                "id": f"{benchmark_name}_{idx}",
                "question": question,
                "answer": answer,
                "gt_letter": gt_letter,
                "has_options": has_options,
            })

            if max_samples is not None and len(samples) >= max_samples:
                break

    return samples


def format_question(sample: dict) -> str:
    """Format a benchmark sample for log-prob evaluation.

    The question already includes options in 'Option A: ...' format.
    We append 'Answer:' so the model's next token is the option letter.
    """
    question = sample["question"]
    return question + "\nAnswer:"


def build_prompt(
    tokenizer: AutoTokenizer,
    question_text: str,
) -> str:
    """Build the full prompt using the tokenizer's chat template.

    Returns the prompt string ready for the completions API,
    ending with 'Answer:' so the model's next token is the option letter.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return prompt


def get_option_token_ids(tokenizer: AutoTokenizer) -> dict[str, int]:
    """Get token IDs for option letters A-E."""
    token_ids = {}
    for letter in OPTION_LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) == 1:
            token_ids[letter] = ids[0]
        else:
            # Some tokenizers produce multiple tokens for a single letter;
            # use only the first token as the representative.
            token_ids[letter] = ids[0]
    return token_ids


def query_logprobs(
    server_url: str,
    prompt: str,
    option_token_ids: dict[str, int],
) -> dict[str, float]:
    """Query SGLang server for log probabilities of option tokens.

    Uses SGLang's native /generate endpoint with return_logprob=True
    and top_logprobs_num to get per-token logprobs including option letters.
    Returns a mapping from option letter to its log probability.
    """
    url = f"{server_url}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0.0,
        },
        "return_logprob": True,
        "top_logprobs_num": 30,
        "logprob_start_len": -1,
    }

    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()

    # output_top_logprobs is a list of lists: [[logprob, token_id, None], ...]
    output_top_logprobs = result.get("meta_info", {}).get("output_top_logprobs", [])

    option_logprobs: dict[str, float] = {}
    if output_top_logprobs and len(output_top_logprobs) > 0:
        top_tokens = output_top_logprobs[0]  # First (only) generated position
        # Build token_id -> logprob mapping
        token_id_to_logprob = {entry[1]: entry[0] for entry in top_tokens}
        for letter in OPTION_LETTERS:
            tid = option_token_ids[letter]
            if tid in token_id_to_logprob:
                option_logprobs[letter] = token_id_to_logprob[tid]
            else:
                option_logprobs[letter] = float("-inf")
    else:
        for letter in OPTION_LETTERS:
            option_logprobs[letter] = float("-inf")

    return option_logprobs


def _process_sample(
    idx: int,
    sample: dict,
    tokenizer: AutoTokenizer,
    server_url: str,
    option_token_ids: dict[str, int],
) -> dict:
    """Process a single sample (thread-safe). Returns result dict."""
    import re

    question_text = format_question(sample)
    prompt = build_prompt(tokenizer, question_text)

    try:
        option_logprobs = query_logprobs(server_url, prompt, option_token_ids)
    except Exception as e:
        return {
            "index": idx,
            "question": sample["question"][:100],
            "gold_answer": sample.get("answer", sample.get("answer_idx", "")),
            "predicted": "ERROR",
            "correct": False,
            "error": str(e),
        }

    predicted = max(option_logprobs, key=lambda k: option_logprobs[k])

    gt_letter = sample.get("gt_letter", "")
    if not gt_letter:
        answer = sample.get("answer", "").strip().upper()
        if len(answer) == 1 and answer in "ABCDE":
            gt_letter = answer

    is_correct = predicted == gt_letter if (predicted and gt_letter) else False

    serializable_logprobs = {
        k: v if math.isfinite(v) else -1000.0
        for k, v in option_logprobs.items()
    }

    return {
        "index": idx,
        "question": sample["question"][:200],
        "gold_answer": gt_letter,
        "answer_text": sample.get("answer", "")[:100],
        "predicted": predicted,
        "correct": is_correct,
        "logprobs": serializable_logprobs,
    }


def evaluate_benchmark(
    benchmark_name: str,
    samples: list[dict],
    tokenizer: AutoTokenizer,
    server_urls: list[str],
    option_token_ids: dict[str, int],
    max_workers: int = 32,
) -> dict[str, Any]:
    """Evaluate a benchmark using concurrent requests across multiple servers."""
    start_time = time.time()
    results: list[dict] = [None] * len(samples)  # type: ignore
    correct = 0
    total = 0
    n_servers = len(server_urls)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, sample in enumerate(samples):
            url = server_urls[idx % n_servers]
            future = executor.submit(
                _process_sample, idx, sample, tokenizer, url, option_token_ids
            )
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result
            total += 1
            if result.get("correct", False):
                correct += 1

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
        description="Log-probability based evaluation for MC-QA benchmarks"
    )
    parser.add_argument(
        "--server-urls",
        nargs="+",
        default=["http://localhost:30000"],
        help="SGLang server URLs for load balancing (default: localhost:30000)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model (for loading tokenizer and chat template)",
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
        default="eval_results/logprob",
        help="Output directory for results (default: eval_results/logprob)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per benchmark (default: all)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Max concurrent requests (default: 32)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = args.tokenizer_path or args.model_path
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )
    option_token_ids = get_option_token_ids(tokenizer)
    print(f"Option token IDs: {option_token_ids}")

    print(f"Server URLs: {args.server_urls}")
    print(f"Benchmarks: {args.benchmarks}")
    if args.max_samples:
        print(f"Max samples per benchmark: {args.max_samples}")
    print()

    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model_path,
        "server_urls": args.server_urls,
        "method": "logprob",
        "benchmarks": {},
    }

    for benchmark_name in args.benchmarks:
        print(f"=== Evaluating {benchmark_name} ===")
        try:
            samples = load_benchmark(benchmark_name, args.max_samples)
        except (FileNotFoundError, ValueError) as e:
            print(f"  Skipping {benchmark_name}: {e}")
            continue

        print(f"  Loaded {len(samples)} samples")

        benchmark_result = evaluate_benchmark(
            benchmark_name=benchmark_name,
            samples=samples,
            tokenizer=tokenizer,
            server_urls=args.server_urls,
            option_token_ids=option_token_ids,
            max_workers=args.max_workers,
        )

        accuracy = benchmark_result["accuracy"]
        elapsed = benchmark_result["elapsed_seconds"]
        rate = benchmark_result["samples_per_second"]
        print(
            f"  Final: {accuracy:.1%} "
            f"({benchmark_result['correct']}/{benchmark_result['total']}) "
            f"in {elapsed:.1f}s ({rate:.1f} samples/s)"
        )
        print()

        # Save per-benchmark results
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
