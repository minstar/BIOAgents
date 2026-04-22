#!/usr/bin/env python3
"""VQA benchmark using SGLang server for batched inference.

Uses the same tool-use prompt format as training, but sends requests
to an SGLang server for much faster throughput (~5-10x vs batch=1 HF).

Prerequisites:
    # Launch SGLang server (triton backend for Qwen3.5-9B's 256-dim heads):
    CUDA_VISIBLE_DEVICES=6,7 python -m sglang.launch_server \
        --model-path /path/to/model \
        --port 30080 \
        --attention-backend triton \
        --tp 2 --trust-remote-code --dtype bfloat16

Usage:
    python scripts/eval_vqa_sglang.py \
        --server-url http://localhost:30080 \
        --model-name v6_step80 \
        --output-dir results/benchmarks_tooluse/v6_step80 \
        --concurrency 16
"""

import base64
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Tool definitions matching training format ──
SUBMIT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit your answer to the medical visual question.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your answer to the question.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief reasoning for your answer.",
                },
            },
            "required": ["answer"],
        },
    },
}

OPENAI_TOOLS = [SUBMIT_ANSWER_TOOL]

# Load training-aligned system prompt with full domain tools
_TRAINING_PROMPTS_PATH = Path(__file__).parent / "verl" / "training_system_prompts.json"

def _build_system_prompt():
    """Build system prompt matching training format: tool-call instructions + domain tools."""
    # Tool-call format instructions (same as Qwen3.5 chat template injects)
    _TOOL_CALL_INSTRUCTIONS = """# Tools

You have access to the following functions:

<tools>
""" + json.dumps(SUBMIT_ANSWER_TOOL) + """
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
            training_prompts = json.load(f)
        domain_prompt = training_prompts.get("visual_diagnosis", "")
        print(f"  Loaded training system prompt: visual_diagnosis ({len(domain_prompt)} chars)", flush=True)
        return _TOOL_CALL_INSTRUCTIONS + domain_prompt
    else:
        print(f"  WARNING: training_system_prompts.json not found, using fallback", flush=True)
        return _TOOL_CALL_INSTRUCTIONS + "You are a medical imaging expert. Analyze the medical image and answer the question. Submit your answer by calling the submit_answer tool."

SYSTEM_PROMPT = _build_system_prompt()

# Only evaluate benchmarks used in the paper
VQA_BENCHMARKS = ["vqa_rad", "slake", "pathvqa"]


def extract_answer_from_tool_call(response: str) -> str:
    """Extract answer from model response — handles both XML and JSON formats."""
    clean = response
    if '<think>' in clean:
        clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split('</think>')[-1].strip()

    # 1. Qwen3.5 XML-style
    xml_match = re.search(r'<parameter=answer>\s*(.*?)\s*</parameter>', clean, re.DOTALL)
    if xml_match:
        return xml_match.group(1).strip()

    # 2. JSON-style tool call
    try:
        json_patterns = re.findall(
            r'\{[^{}]*"name"\s*:\s*"submit_answer"[^{}]*\}', clean, re.DOTALL
        )
        for pat in json_patterns:
            try:
                obj = json.loads(pat)
                if "arguments" in obj:
                    return obj["arguments"].get("answer", "").strip()
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    # 3. "answer": "..." pattern
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', clean)
    if m:
        return m.group(1).strip()

    # 4. Nested arguments
    try:
        args_match = re.search(r'"arguments"\s*:\s*(\{[^{}]+\})', clean)
        if args_match:
            args_obj = json.loads(args_match.group(1))
            if "answer" in args_obj:
                return str(args_obj["answer"]).strip()
    except Exception:
        pass

    # 5. Fallback
    for prefix in ["Answer:", "The answer is", "answer:"]:
        if clean.lower().startswith(prefix.lower()):
            clean = clean[len(prefix):].strip()
    return clean


def compute_exact_match(prediction: str, reference: str) -> float:
    pred = _normalize_answer(prediction)
    ref = _normalize_answer(reference)
    return 1.0 if pred == ref else 0.0


def compute_token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(_normalize_answer(prediction).split())
    ref_tokens = set(_normalize_answer(reference).split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    return text


def encode_image_base64(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def send_vqa_request(
    server_url: str,
    question: str,
    image_path: str | None,
    max_tokens: int = 512,
    no_think: bool = True,
) -> str:
    """Send a single VQA request to SGLang server."""
    # Build message content
    if image_path and os.path.exists(image_path):
        b64 = encode_image_base64(image_path)
        # Determine MIME type
        ext = Path(image_path).suffix.lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
            ext.lstrip("."), "image/jpeg"
        )
        content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": question},
        ]
    else:
        content = question

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    resp = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="VQA eval via SGLang server")
    parser.add_argument("--server-url", type=str, default="http://localhost:30080")
    parser.add_argument("--model-name", type=str, default="model")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks_tooluse")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of concurrent requests")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-think", action="store_true", default=True)
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="Comma-separated benchmarks (default: vqa_rad,slake,pathvqa)")
    args = parser.parse_args()

    benchmarks = args.benchmarks.split(",") if args.benchmarks else VQA_BENCHMARKS

    # Verify server is running
    try:
        r = requests.get(f"{args.server_url}/health", timeout=5)
        print(f"  SGLang server: OK", flush=True)
    except Exception as e:
        print(f"  ERROR: Cannot reach SGLang server at {args.server_url}: {e}", flush=True)
        sys.exit(1)

    print(f"\n{'#'*70}", flush=True)
    print(f"  VQA (SGLang Batched) — Training-Aligned Evaluation", flush=True)
    print(f"  Model: {args.model_name}", flush=True)
    print(f"  Server: {args.server_url}", flush=True)
    print(f"  Concurrency: {args.concurrency}", flush=True)
    print(f"{'#'*70}\n", flush=True)

    from bioagents.data_pipeline.vqa_loader import VQA_DATASET_REGISTRY

    all_results = {}

    for benchmark in benchmarks:
        if benchmark not in VQA_DATASET_REGISTRY:
            print(f"  [SKIP] Unknown benchmark: {benchmark}", flush=True)
            continue

        info = VQA_DATASET_REGISTRY[benchmark]
        loader = info["loader"]

        try:
            if benchmark in ("vqa_med_2021", "quilt_vqa"):
                data = loader(max_samples=args.max_samples if args.max_samples > 0 else None)
            else:
                data = loader(
                    max_samples=args.max_samples if args.max_samples > 0 else None,
                    split="test",
                )
        except Exception as e:
            print(f"  [ERROR] Failed to load {benchmark}: {e}", flush=True)
            all_results[benchmark] = {"error": str(e)}
            continue

        if not data:
            print(f"  [SKIP] No data found for {benchmark}", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"  [{args.model_name}] {benchmark}: {len(data)} samples", flush=True)
        print(f"{'='*60}", flush=True)

        per_sample_results = []
        metric_sums = Counter()
        t0 = time.time()
        completed = 0

        def process_item(idx_item):
            idx, item = idx_item
            question = item["question"]
            reference = item["answer"]
            image_path = item.get("image_path")

            if not question or not reference:
                return None

            try:
                response = send_vqa_request(
                    args.server_url, question, image_path,
                    no_think=args.no_think,
                )
            except Exception as e:
                response = f"ERROR: {e}"

            prediction = extract_answer_from_tool_call(response)
            em = compute_exact_match(prediction, reference)
            f1 = compute_token_f1(prediction, reference)

            return {
                "idx": idx,
                "id": item.get("id", f"{benchmark}_{idx}"),
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "raw_response": response[:500],
                "has_image": bool(image_path and os.path.exists(image_path)),
                "metrics": {"exact_match": em, "token_f1": f1},
            }

        # Process with thread pool
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(process_item, (i, item)): i
                for i, item in enumerate(data)
            }

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue

                per_sample_results.append(result)
                metric_sums["exact_match"] += result["metrics"]["exact_match"]
                metric_sums["token_f1"] += result["metrics"]["token_f1"]
                completed += 1

                # Progress logging
                if completed % 50 == 0:
                    n = len(per_sample_results)
                    avg_em = metric_sums["exact_match"] / n
                    avg_f1 = metric_sums["token_f1"] / n
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (len(data) - completed) / rate if rate > 0 else 0
                    print(
                        f"  Progress: {completed}/{len(data)} | "
                        f"EM={avg_em:.3f} F1={avg_f1:.3f} | "
                        f"{elapsed:.0f}s (ETA {eta:.0f}s, {rate:.1f} samples/s)",
                        flush=True,
                    )

        # Sort by original index
        per_sample_results.sort(key=lambda x: x["idx"])

        total = len(per_sample_results)
        if total == 0:
            all_results[benchmark] = {"error": "No valid samples"}
            continue

        avg_metrics = {m: v / total for m, v in metric_sums.items()}
        elapsed = time.time() - t0

        all_results[benchmark] = {
            "benchmark": benchmark,
            "total": total,
            "metrics": avg_metrics,
            "elapsed_seconds": round(elapsed, 1),
            "throughput": round(total / elapsed, 2),
            "per_sample": per_sample_results,
        }

        print(
            f"  [{args.model_name}] {benchmark}: "
            f"EM={avg_metrics['exact_match']:.4f} "
            f"F1={avg_metrics['token_f1']:.4f} "
            f"({total} samples, {elapsed:.0f}s, "
            f"{total/elapsed:.1f} samples/s)",
            flush=True,
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"vqa_sglang_{ts}.json"

    summary = {}
    for key, r in all_results.items():
        if "error" in r:
            summary[key] = r
        else:
            summary[key] = {
                k: v for k, v in r.items() if k != "per_sample"
            }

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

    # Save per-sample details
    detail_path = output_dir / f"vqa_sglang_detail_{ts}.json"
    detail = {
        "model_name": args.model_name,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {
            k: v.get("per_sample", []) for k, v in all_results.items() if "per_sample" in v
        },
    }
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)

    # Print summary
    print(f"\n{'='*70}", flush=True)
    print(f"  VQA (SGLang) RESULTS: {args.model_name}", flush=True)
    print(f"  {'Benchmark':<15} {'EM':>8} {'F1':>8} {'Total':>6} {'Time':>6} {'Rate':>10}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for key, r in all_results.items():
        if "error" in r:
            print(f"  {key:<15} {'ERROR':>8} {r['error']}", flush=True)
        else:
            m = r["metrics"]
            print(
                f"  {key:<15} {m['exact_match']:>8.4f} {m['token_f1']:>8.4f} "
                f"{r['total']:>6} {r['elapsed_seconds']:>5.0f}s "
                f"{r['throughput']:>7.1f}/s",
                flush=True,
            )
    print(f"\n  Summary: {out_path}", flush=True)
    print(f"  Details: {detail_path}", flush=True)


if __name__ == "__main__":
    main()
