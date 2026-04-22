#!/usr/bin/env python3
"""TextQA benchmark using SGLang server for batched inference.

Uses training-aligned tool-use format with concurrent requests via SGLang.

Prerequisites:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server \
        --model-path /path/to/model --port 30080 \
        --attention-backend triton --tp 8 --trust-remote-code --dtype bfloat16

Usage:
    python scripts/eval_textqa_sglang.py \
        --server-url http://localhost:30080 \
        --model-name v16_step60 \
        --output-dir results/benchmarks_tooluse/v16_step60 \
        --concurrency 16
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests as http_requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Load training-aligned system prompt
_TRAINING_PROMPTS_PATH = Path(__file__).parent / "verl" / "training_system_prompts.json"

# Tool-call format instructions (matching Qwen3.5 chat template)
_SUBMIT_TOOL_JSON = json.dumps({
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit your final answer to the medical question.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The answer letter (A, B, C, or D)"},
                "reasoning": {"type": "string", "description": "Brief reasoning for your answer choice."},
            },
            "required": ["answer"],
        },
    },
})

_TOOL_CALL_INSTRUCTIONS = f"""# Tools

You have access to the following functions:

<tools>
{_SUBMIT_TOOL_JSON}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

"""

if _TRAINING_PROMPTS_PATH.exists():
    with open(_TRAINING_PROMPTS_PATH) as f:
        _training_prompts = json.load(f)
    SYSTEM_PROMPT = _TOOL_CALL_INSTRUCTIONS + _training_prompts.get("medical_qa", "")
    print(f"  Loaded training system prompt: medical_qa ({len(SYSTEM_PROMPT)} chars)", flush=True)
else:
    SYSTEM_PROMPT = _TOOL_CALL_INSTRUCTIONS + (
        "You are a medical AI assistant. Answer the following medical question "
        "by calling the submit_answer tool with the correct answer letter and your reasoning."
    )
    print(f"  WARNING: training_system_prompts.json not found, using fallback", flush=True)

# Benchmark files
BENCHMARK_FILES = {
    "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
    "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
    "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_test.jsonl",
}


def extract_answer_from_response(response: str) -> str:
    """Extract answer letter from model response."""
    # Strip think tags
    clean = response
    if '<think>' in clean:
        clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split('</think>')[-1].strip()

    # 1. XML-style tool call: <parameter=answer>A</parameter>
    xml_match = re.search(r'<parameter=answer>\s*([A-Ea-e])\s*</parameter>', clean)
    if xml_match:
        return xml_match.group(1).upper()

    # 2. JSON submit_answer
    try:
        json_patterns = re.findall(r'\{[^{}]*"name"\s*:\s*"submit_answer"[^{}]*\}', clean, re.DOTALL)
        for pat in json_patterns:
            try:
                obj = json.loads(pat)
                if "arguments" in obj:
                    ans = obj["arguments"].get("answer", "")
                    for ch in ans:
                        if ch.upper() in "ABCDE":
                            return ch.upper()
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    # 3. "answer": "X"
    m = re.search(r'"answer"\s*:\s*"([A-Ea-e])"', clean)
    if m:
        return m.group(1).upper()

    # 4. <answer>X</answer>
    m = re.search(r'<answer>\s*([A-Ea-e])\s*</answer>', clean)
    if m:
        return m.group(1).upper()

    # 5. First A-E letter in clean response
    for ch in clean:
        if ch in "ABCDE":
            return ch

    return ""


def _text_answer_to_letter(question: str, answer_text: str) -> str:
    """Map text answer to letter by matching against options in question."""
    if len(answer_text) == 1 and answer_text.upper() in "ABCDE":
        return answer_text.upper()

    answer_lower = answer_text.lower().strip()
    for letter in "ABCDE":
        for pat in [rf"Option {letter}:\s*(.+?)(?:\n|$)", rf"\b{letter}[.)]\s*(.+?)(?:\n|$)"]:
            m = re.search(pat, question)
            if m and m.group(1).strip().lower() == answer_lower:
                return letter

    best_letter, best_overlap = "", 0
    for letter in "ABCDE":
        for pat in [rf"Option {letter}:\s*(.+?)(?:\n|$)", rf"\b{letter}[.)]\s*(.+?)(?:\n|$)"]:
            m = re.search(pat, question)
            if m:
                opt_text = m.group(1).strip().lower()
                if answer_lower in opt_text or opt_text in answer_lower:
                    overlap = len(opt_text)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_letter = letter
    return best_letter


def send_request(server_url, question, max_tokens=1024, no_think=True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    resp = http_requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TextQA eval via SGLang server")
    parser.add_argument("--server-url", type=str, default="http://localhost:30080")
    parser.add_argument("--model-name", type=str, default="model")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks_tooluse")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-think", action="store_true", default=True)
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="Comma-separated benchmark keys (e.g., medqa,medmcqa)")
    args = parser.parse_args()

    # Verify server
    try:
        http_requests.get(f"{args.server_url}/health", timeout=5)
        print(f"  SGLang server: OK", flush=True)
    except Exception as e:
        print(f"  ERROR: Cannot reach SGLang server: {e}", flush=True)
        sys.exit(1)

    benchmarks = args.benchmarks.split(",") if args.benchmarks else list(BENCHMARK_FILES.keys())

    print(f"\n{'#'*70}", flush=True)
    print(f"  TextQA (SGLang Batched) — Training-Aligned Evaluation", flush=True)
    print(f"  Model: {args.model_name}", flush=True)
    print(f"  Server: {args.server_url}", flush=True)
    print(f"  Concurrency: {args.concurrency}", flush=True)
    print(f"  Benchmarks: {benchmarks}", flush=True)
    print(f"{'#'*70}\n", flush=True)

    all_results = {}

    for bm_key in benchmarks:
        if bm_key not in BENCHMARK_FILES:
            print(f"  [SKIP] Unknown benchmark: {bm_key}", flush=True)
            continue

        bm_file = BENCHMARK_FILES[bm_key]
        full_path = PROJECT_ROOT / bm_file
        if not full_path.exists():
            print(f"  [SKIP] {bm_key}: {bm_file} not found", flush=True)
            continue

        data = []
        with open(full_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        if args.max_samples > 0:
            data = data[:args.max_samples]

        print(f"\n  [{args.model_name}] {bm_key}: {len(data)} samples", flush=True)

        correct = 0
        total = 0
        t0 = time.time()

        def process_item(idx_item):
            idx, item = idx_item
            instances = item.get("instances", {})
            question = instances.get("input", "") if isinstance(instances, dict) else ""
            answer = instances.get("output", "") if isinstance(instances, dict) else ""
            if not question:
                return None

            try:
                response = send_request(args.server_url, question, no_think=args.no_think)
            except Exception as e:
                response = f"ERROR: {e}"

            pred = extract_answer_from_response(response)
            ref = _text_answer_to_letter(question, answer.strip())

            return {"idx": idx, "pred": pred, "ref": ref, "correct": pred and ref and pred == ref}

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(process_item, (i, item)): i
                for i, item in enumerate(data)
            }

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue

                total += 1
                if result["correct"]:
                    correct += 1

                if total % 100 == 0:
                    acc = correct / max(total, 1)
                    elapsed = time.time() - t0
                    rate = total / elapsed
                    eta = (len(data) - total) / rate if rate > 0 else 0
                    print(
                        f"    {bm_key}: {total}/{len(data)} acc={acc:.3f} "
                        f"({elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s)",
                        flush=True,
                    )

        acc = correct / max(total, 1)
        elapsed = time.time() - t0
        all_results[bm_key] = {
            "accuracy": round(acc, 4),
            "correct": correct,
            "total": total,
            "elapsed_seconds": round(elapsed, 1),
            "throughput": round(total / max(elapsed, 0.1), 2),
        }
        print(
            f"  [{args.model_name}] {bm_key}: accuracy={acc:.4f} ({correct}/{total}) "
            f"{elapsed:.0f}s ({total/max(elapsed,0.1):.1f}/s)",
            flush=True,
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"textqa_sglang_{ts}.json"

    report = {
        "model_name": args.model_name,
        "server_url": args.server_url,
        "format": "tool-use (SGLang batched, training-aligned)",
        "concurrency": args.concurrency,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*70}", flush=True)
    print(f"  TextQA (SGLang) RESULTS: {args.model_name}", flush=True)
    print(f"  {'Benchmark':<20} {'Accuracy':>10} {'Correct':>8} {'Total':>6} {'Rate':>8}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for key, r in all_results.items():
        print(
            f"  {key:<20} {r['accuracy']:>10.4f} {r['correct']:>8} "
            f"{r['total']:>6} {r['throughput']:>6.1f}/s",
            flush=True,
        )
    print(f"\n  Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
