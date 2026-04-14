#!/usr/bin/env python3
"""MedLFQA benchmark using SGLang server for batched inference.

Same tool-use format as training, but uses SGLang for concurrent requests.

Prerequisites:
    # Launch SGLang server:
    CUDA_VISIBLE_DEVICES=6,7 python -m sglang.launch_server \
        --model-path /path/to/model \
        --port 30080 --attention-backend triton \
        --tp 2 --trust-remote-code --dtype bfloat16

Usage:
    python scripts/eval_medlfqa_sglang.py \
        --server-url http://localhost:30080 \
        --model-name v6_step80 \
        --output-dir results/benchmarks_tooluse/v6_step80 \
        --concurrency 8
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

SUBMIT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit your detailed answer to the medical question.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your comprehensive, evidence-based answer.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Key evidence and reasoning supporting your answer.",
                },
            },
            "required": ["answer"],
        },
    },
}

OPENAI_TOOLS = [SUBMIT_ANSWER_TOOL]

# Build system prompt with tool schema embedded (matching apply_chat_template(tools=) output)
_TOOL_JSON = json.dumps(SUBMIT_ANSWER_TOOL)
SYSTEM_PROMPT = f"""# Tools

You have access to the following functions:

<tools>
{_TOOL_JSON}
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

You are a medical expert. Provide detailed, accurate, evidence-based answers to medical questions. Submit your answer by calling the submit_answer tool with a comprehensive response."""

MEDLFQA_DATASETS = {
    "kqa_golden": {
        "path": "evaluations/OLAPH/MedLFQA/kqa_golden_test_MedLFQA.jsonl",
        "name": "KQA Golden",
    },
    "live_qa": {
        "path": "evaluations/OLAPH/MedLFQA/live_qa_test_MedLFQA.jsonl",
        "name": "LiveQA",
    },
    "medication_qa": {
        "path": "evaluations/OLAPH/MedLFQA/medication_qa_test_MedLFQA.jsonl",
        "name": "MedicationQA",
    },
    "healthsearch_qa": {
        "path": "evaluations/OLAPH/MedLFQA/healthsearch_qa_test_MedLFQA.jsonl",
        "name": "HealthSearchQA",
    },
    "kqa_silver": {
        "path": "evaluations/OLAPH/MedLFQA/kqa_silver_wogold_test_MedLFQA.jsonl",
        "name": "KQA Silver",
    },
}


def load_medlfqa_data(dataset_key, max_samples=0):
    info = MEDLFQA_DATASETS[dataset_key]
    data_path = PROJECT_ROOT / info["path"]
    if not data_path.exists():
        print(f"[WARN] Not found: {data_path}", flush=True)
        return []
    data = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if max_samples > 0:
        data = data[:max_samples]
    return data


def compute_rouge_l(prediction, reference):
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
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
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def compute_token_f1(prediction, reference):
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_answer_from_tool_call(response: str) -> str:
    clean = response
    if '<think>' in clean:
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split("</think>")[-1].strip()

    # XML-style
    xml_answer = re.search(r'<parameter=answer>\s*(.*?)\s*</parameter>', clean, re.DOTALL)
    if xml_answer:
        answer = xml_answer.group(1).strip()
        xml_reasoning = re.search(r'<parameter=reasoning>\s*(.*?)\s*</parameter>', clean, re.DOTALL)
        if xml_reasoning:
            return answer + "\n" + xml_reasoning.group(1).strip()
        return answer

    # JSON-style
    try:
        json_patterns = re.findall(
            r'\{[^{}]*"name"\s*:\s*"submit_answer"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})',
            clean, re.DOTALL,
        )
        for args_str in json_patterns:
            try:
                args_obj = json.loads(args_str)
                answer = args_obj.get("answer", "")
                if answer:
                    reasoning = args_obj.get("reasoning", "")
                    return (answer + "\n" + reasoning) if reasoning else answer
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    # Broader JSON
    try:
        for block in re.findall(r'\{[^{}]+\}', clean):
            try:
                obj = json.loads(block)
                if "answer" in obj:
                    answer = obj["answer"]
                    reasoning = obj.get("reasoning", "")
                    return (answer + "\n" + reasoning) if reasoning else answer
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    clean = re.sub(r'<tool_call>.*?</tool_call>', '', clean, flags=re.DOTALL).strip()
    clean = re.sub(r'<function=.*?</function>', '', clean, flags=re.DOTALL).strip()
    return clean


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

    resp = http_requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MedLFQA eval via SGLang server")
    parser.add_argument("--server-url", type=str, default="http://localhost:30080")
    parser.add_argument("--model-name", type=str, default="model")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks_tooluse")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-think", action="store_true", default=True)
    args = parser.parse_args()

    # Verify server
    try:
        http_requests.get(f"{args.server_url}/health", timeout=5)
        print(f"  SGLang server: OK", flush=True)
    except Exception as e:
        print(f"  ERROR: Cannot reach SGLang server: {e}", flush=True)
        sys.exit(1)

    print(f"\n{'#'*70}", flush=True)
    print(f"  MedLFQA (SGLang Batched) — Training-Aligned Evaluation", flush=True)
    print(f"  Model: {args.model_name}", flush=True)
    print(f"  Concurrency: {args.concurrency}", flush=True)
    print(f"{'#'*70}\n", flush=True)

    all_results = {}

    for ds_key, ds_info in MEDLFQA_DATASETS.items():
        data = load_medlfqa_data(ds_key, args.max_samples)
        if not data:
            print(f"  [SKIP] {ds_info['name']}: no data", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"  [{args.model_name}] {ds_info['name']}: {len(data)} samples", flush=True)
        print(f"{'='*60}", flush=True)

        results = []
        rouge_sum = 0.0
        f1_sum = 0.0
        t0 = time.time()
        completed = 0

        def process_item(idx_item):
            idx, item = idx_item
            question = item.get("Question", item.get("question", ""))
            reference = item.get("Free_form_answer", item.get("answer", item.get("long_answer", "")))
            if not question or not reference:
                return None

            try:
                response = send_request(args.server_url, question, no_think=args.no_think)
            except Exception as e:
                response = f"ERROR: {e}"

            prediction = extract_answer_from_tool_call(response)
            rouge = compute_rouge_l(prediction, reference)
            f1 = compute_token_f1(prediction, reference)

            return {
                "idx": idx,
                "question": question[:200],
                "reference": reference[:200],
                "prediction": prediction[:500],
                "rouge_l": rouge,
                "token_f1": f1,
            }

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(process_item, (i, item)): i
                for i, item in enumerate(data)
            }

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue

                results.append(result)
                rouge_sum += result["rouge_l"]
                f1_sum += result["token_f1"]
                completed += 1

                if completed % 50 == 0:
                    avg_rouge = rouge_sum / completed
                    avg_f1 = f1_sum / completed
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (len(data) - completed) / rate if rate > 0 else 0
                    print(
                        f"  {ds_info['name']}: {completed}/{len(data)} "
                        f"ROUGE-L={avg_rouge:.3f} F1={avg_f1:.3f} "
                        f"({elapsed:.0f}s, ETA {eta:.0f}s)",
                        flush=True,
                    )

        results.sort(key=lambda x: x["idx"])
        total = len(results)
        elapsed = time.time() - t0

        if total == 0:
            all_results[ds_key] = {"error": "No valid samples"}
            continue

        avg_rouge = rouge_sum / total
        avg_f1 = f1_sum / total

        all_results[ds_key] = {
            "name": ds_info["name"],
            "total": total,
            "avg_rouge_l": round(avg_rouge, 4),
            "avg_token_f1": round(avg_f1, 4),
            "elapsed_seconds": round(elapsed, 1),
            "throughput": round(total / elapsed, 2),
            "per_sample": results,
        }

        print(
            f"  [{args.model_name}] {ds_info['name']}: "
            f"ROUGE-L={avg_rouge:.4f} Token-F1={avg_f1:.4f} "
            f"({total} samples, {elapsed:.0f}s, {total/elapsed:.1f}/s)",
            flush=True,
        )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"medlfqa_sglang_{ts}.json"

    summary = {}
    for key, r in all_results.items():
        if "error" in r:
            summary[key] = r
        else:
            summary[key] = {k: v for k, v in r.items() if k != "per_sample"}

    report = {
        "model_name": args.model_name,
        "server_url": args.server_url,
        "format": "tool-use (SGLang batched)",
        "concurrency": args.concurrency,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": summary,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*70}", flush=True)
    print(f"  MedLFQA (SGLang) RESULTS: {args.model_name}", flush=True)
    print(f"  {'Dataset':<20} {'ROUGE-L':>10} {'Token-F1':>10} {'Total':>6} {'Rate':>8}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for key, r in all_results.items():
        if "error" in r:
            print(f"  {key:<20} {'ERROR':>10}", flush=True)
        else:
            print(
                f"  {r['name']:<20} {r['avg_rouge_l']:>10.4f} {r['avg_token_f1']:>10.4f} "
                f"{r['total']:>6} {r['throughput']:>6.1f}/s",
                flush=True,
            )
    print(f"\n  Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
