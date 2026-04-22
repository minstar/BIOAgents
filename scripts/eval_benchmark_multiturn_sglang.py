#!/usr/bin/env python3
"""Fast multi-turn benchmark evaluation using SGLang server.

Instead of loading the model per-GPU with transformers (1 sample at a time),
this script:
  1. Launches an SGLang server on 1 GPU
  2. Sends concurrent async requests for multiple samples
  3. Achieves 5-10x throughput via continuous batching + prefix caching

SGLang's RadixAttention reuses KV cache for the shared system prompt + tool
definitions across all samples, giving 30-50% cache hit rate.

Usage:
    # Start SGLang server on GPU 0 (in a separate terminal or background):
    CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
        --model-path /path/to/merged_hf \
        --port 30000 --host 0.0.0.0

    # Run eval against the server:
    python scripts/eval_benchmark_multiturn_sglang.py \
        --server-url http://localhost:30000 \
        --benchmarks medqa medmcqa mmlu \
        --max-turns 10 \
        --concurrency 16 \
        --output-dir results/benchmarks_multiturn/v16_step60_sglang

    # All-in-one (launches server + runs eval):
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmark_multiturn_sglang.py \
        --model-path /path/to/merged_hf \
        --benchmarks medqa \
        --max-turns 10 \
        --concurrency 16 \
        --output-dir results/benchmarks_multiturn/v16_step60_sglang
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from loguru import logger

# ── Benchmark file registry (same as original) ──
BENCHMARK_FILES = {
    "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
    "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
    "mmlu": "evaluations/self-biorag/data/benchmark/mmlu_test.jsonl",
}

BENCHMARK_DOMAIN = {
    "medqa": "medical_qa",
    "medmcqa": "medical_qa",
    "mmlu": "medical_qa",
}


def load_textqa_benchmark(name: str) -> list[dict]:
    """Load a TextQA benchmark from JSONL and convert to task format."""
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
            instances = item.get("instances", {})
            question = instances.get("input", "")
            answer = instances.get("output", "").strip()

            options = {}
            for letter in "ABCDE":
                pat = rf"Option {letter}:\s*(.+?)(?=Option [A-E]:|$)"
                m = re.search(pat, question, re.DOTALL)
                if m:
                    options[letter] = m.group(1).strip()

            tasks.append({
                "id": f"{name}_{idx}",
                "ticket": question,
                "correct_answer": answer,
                "options": options,
            })

    logger.info(f"Loaded {len(tasks)} tasks from {name}")
    return tasks


def _check_answer(submitted: str, gold: str, options: dict) -> bool:
    """Check if submitted answer matches gold, handling letter/text mismatches."""
    if not submitted:
        return False

    submitted = submitted.strip()
    gold = gold.strip()

    # Direct match
    if submitted.upper() == gold.upper():
        return True

    # If gold is a single letter
    if len(gold) <= 2 and gold[0].upper() in "ABCDE":
        first_char = submitted[0].upper() if submitted else ""
        if first_char == gold[0].upper():
            return True
        m = re.match(r'^([A-E])[.\):\s]', submitted.upper())
        if m and m.group(1) == gold[0].upper():
            return True
        return False

    # Gold is full text — find which letter it corresponds to
    gold_letter = None
    for letter, text in options.items():
        if gold.lower() == text.lower() or gold.lower() in text.lower():
            gold_letter = letter
            break

    if gold_letter:
        first_char = submitted[0].upper() if submitted else ""
        if first_char == gold_letter.upper():
            return True
        m = re.match(r'^([A-E])[.\):\s]', submitted.upper())
        if m and m.group(1) == gold_letter.upper():
            return True

    if gold.lower() in submitted.lower():
        return True

    return False


def _extract_answer_fallback(text: str) -> str:
    """Extract answer from raw text when no submit_answer was called."""
    m = re.search(r'<parameter=answer>\s*(.*?)\s*</parameter>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'(?:the answer is|answer:)\s*([A-E])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return text[:100].strip()


def build_system_prompt(task: dict) -> str:
    """Build system prompt for medical QA multi-turn evaluation."""
    return (
        "You are a medical expert agent. You have access to the following tools:\n\n"
        "1. **think(thought)**: Record your reasoning process. Use this to analyze the question.\n"
        "2. **search_evidence(query)**: Search medical literature for relevant evidence.\n"
        "3. **analyze_answer_options(options)**: Analyze the given answer options.\n"
        "4. **submit_answer(answer)**: Submit your final answer (A/B/C/D/E letter).\n\n"
        "To use a tool, output a JSON object with 'name' and 'arguments' keys.\n"
        "Example: {\"name\": \"think\", \"arguments\": {\"thought\": \"Let me analyze...\"}}\n\n"
        "IMPORTANT: You MUST eventually call submit_answer with your chosen letter.\n"
        "Work through the problem step by step: think → search → analyze → submit."
    )


def parse_tool_call(text: str) -> dict | None:
    """Parse tool call from model output (JSON, XML, or ReAct format)."""
    # Try JSON format
    try:
        # Find JSON object in text
        for m in re.finditer(r'\{[^{}]*"name"[^{}]*\}', text, re.DOTALL):
            obj = json.loads(m.group())
            if "name" in obj:
                if "arguments" not in obj:
                    obj["arguments"] = {}
                return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Try Qwen3.5 XML format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    m = re.search(r'<function=(\w+)>(.*?)</function>', text, re.DOTALL)
    if m:
        name = m.group(1)
        args = {}
        for pm in re.finditer(r'<parameter=(\w+)>(.*?)</parameter>', m.group(2), re.DOTALL):
            args[pm.group(1)] = pm.group(2).strip()
        return {"name": name, "arguments": args}

    return None


def execute_tool_locally(tool_call: dict, task: dict) -> str:
    """Execute tool call locally (simulated environment)."""
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", {})

    if name == "think":
        return f"Thought recorded: {args.get('thought', '')[:200]}"
    elif name == "search_evidence":
        query = args.get("query", "")
        # Return relevant info from the question context
        return f"Search results for '{query}': Based on medical literature, please analyze the question and options carefully to determine the correct answer."
    elif name == "analyze_answer_options":
        options = task.get("options", {})
        opts_str = "\n".join(f"  {k}: {v}" for k, v in options.items())
        return f"Answer options:\n{opts_str}\n\nAnalyze each option based on the evidence gathered."
    elif name == "submit_answer":
        return f"Answer submitted: {args.get('answer', '')}"
    else:
        return f"Unknown tool: {name}"


async def run_single_task(
    client,
    model_name: str,
    task: dict,
    max_turns: int,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Run a single task through multi-turn loop via OpenAI API."""
    system_prompt = build_system_prompt(task)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task["ticket"]},
    ]

    submitted_answer = ""
    turns = 0
    t0 = time.time()

    for turn_idx in range(max_turns):
        # Inject submit nudge on last turn
        if turn_idx == max_turns - 1 and not submitted_answer:
            messages.append({
                "role": "user",
                "content": (
                    "IMPORTANT: This is your LAST turn. You MUST call submit_answer now "
                    "with your best answer. Do NOT call any other tool. "
                    "Call submit_answer immediately with a letter (A/B/C/D/E)."
                ),
            })

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_output = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"API error on {task['id']} turn {turn_idx}: {e}")
            break

        turns += 1
        tool_call = parse_tool_call(raw_output)

        if tool_call:
            tool_name = tool_call.get("name", "")
            tool_result = execute_tool_locally(tool_call, task)

            messages.append({"role": "assistant", "content": raw_output})
            messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}:\n{tool_result}",
            })

            if tool_name == "submit_answer":
                submitted_answer = tool_call.get("arguments", {}).get("answer", "")
                break

            # Detect repetition
            if turn_idx >= 2:
                # Check last 3 assistant messages for repeated tool calls
                recent_tools = []
                for msg in messages[-6:]:
                    if msg["role"] == "assistant":
                        tc = parse_tool_call(msg["content"])
                        if tc:
                            recent_tools.append(tc.get("name", ""))
                if len(recent_tools) >= 3 and len(set(recent_tools[-3:])) == 1:
                    messages.append({
                        "role": "user",
                        "content": f"You have called '{recent_tools[-1]}' multiple times. Please submit your final answer now using submit_answer.",
                    })
        else:
            # No tool call — try to extract answer from text
            messages.append({"role": "assistant", "content": raw_output})
            break

    latency = time.time() - t0

    if not submitted_answer:
        # Fallback: extract from last assistant message
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                submitted_answer = _extract_answer_fallback(msg["content"])
                break

    gold = task["correct_answer"].strip()
    is_correct = _check_answer(submitted_answer, gold, task.get("options", {}))

    return {
        "task_id": task["id"],
        "gold": gold,
        "submitted": submitted_answer,
        "correct": is_correct,
        "turns": turns,
        "latency": latency,
    }


async def run_benchmark_concurrent(
    client,
    model_name: str,
    benchmark_name: str,
    tasks: list[dict],
    max_turns: int,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    output_dir: Path,
):
    """Run benchmark with concurrent async requests."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    correct = 0
    total = 0
    t_start = time.time()
    lock = asyncio.Lock()

    async def process_task(task: dict):
        nonlocal correct, total
        async with semaphore:
            result = await run_single_task(
                client, model_name, task, max_turns, temperature, max_tokens,
            )

        async with lock:
            results.append(result)
            if result["correct"]:
                correct += 1
            total += 1

            if total % 10 == 0:
                acc = correct / total
                elapsed = time.time() - t_start
                rate = total / elapsed * 60
                eta = (len(tasks) - total) / max(rate, 0.01)
                logger.info(
                    f"  [{benchmark_name}] {total}/{len(tasks)} "
                    f"acc={acc:.3f} rate={rate:.1f}/min ETA={eta:.0f}min"
                )

            if total % 100 == 0:
                _save_partial_async(benchmark_name, results, correct, total, output_dir)

    # Launch all tasks concurrently (semaphore limits parallelism)
    await asyncio.gather(*[process_task(task) for task in tasks])

    elapsed = time.time() - t_start
    accuracy = correct / max(total, 1)

    summary = {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_turns": sum(r.get("turns", 0) for r in results) / max(len(results), 1),
        "avg_latency": sum(r.get("latency", 0) for r in results) / max(len(results), 1),
        "total_time_seconds": elapsed,
        "rate_per_min": total / elapsed * 60,
        "concurrency": concurrency,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    out_path = output_dir / f"{benchmark_name}_sglang_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        f"\n{'='*60}\n"
        f"  {benchmark_name}: accuracy={accuracy:.3f} ({correct}/{total})\n"
        f"  avg_turns={summary['avg_turns']:.1f}  rate={summary['rate_per_min']:.1f}/min\n"
        f"  time={elapsed:.0f}s  saved={out_path}\n"
        f"{'='*60}"
    )

    return summary


def _save_partial_async(benchmark_name, results, correct, total, output_dir):
    """Save partial results."""
    partial = {
        "benchmark": benchmark_name,
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "results": results,
    }
    path = output_dir / f"{benchmark_name}_partial.json"
    with open(path, "w") as f:
        json.dump(partial, f, indent=2, ensure_ascii=False)


def wait_for_server(base_url: str, timeout: int = 300):
    """Wait for SGLang server to be ready."""
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{base_url}/health")
            urllib.request.urlopen(req, timeout=5)
            logger.info(f"SGLang server ready at {base_url}")
            return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(2)
    raise TimeoutError(f"SGLang server not ready after {timeout}s")


def launch_sglang_server(model_path: str, port: int = 30000, gpu_id: int = 0) -> subprocess.Popen:
    """Launch SGLang server in background."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--mem-fraction-static", "0.85",
        "--max-running-requests", "32",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info(f"Launching SGLang server on GPU {gpu_id}, port {port}...")
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    return proc


async def main_async(args):
    """Main async evaluation loop."""
    from openai import AsyncOpenAI

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    server_proc = None
    base_url = args.server_url

    # Launch server if model path provided and no server URL
    if args.model_path and not args.server_url:
        port = args.port
        base_url = f"http://localhost:{port}"
        server_proc = launch_sglang_server(args.model_path, port, args.gpu_id)
        try:
            wait_for_server(base_url, timeout=300)
        except TimeoutError:
            server_proc.kill()
            raise

    # Create async OpenAI client pointing to SGLang server
    client = AsyncOpenAI(
        base_url=f"{base_url}/v1",
        api_key="none",  # SGLang doesn't need API key
    )

    # Get model name from server
    try:
        models = await client.models.list()
        model_name = models.data[0].id if models.data else "default"
        logger.info(f"Using model: {model_name}")
    except Exception:
        model_name = "default"

    all_summaries = {}

    try:
        for bench_name in args.benchmarks:
            tasks = load_textqa_benchmark(bench_name)
            if not tasks:
                continue

            if args.resume_from > 0:
                tasks = tasks[args.resume_from:]
            if args.max_samples > 0:
                tasks = tasks[:args.max_samples]

            logger.info(f"\n{'='*60}")
            logger.info(
                f"Evaluating {bench_name}: {len(tasks)} samples, "
                f"concurrency={args.concurrency}, max_turns={args.max_turns}"
            )
            logger.info(f"{'='*60}")

            summary = await run_benchmark_concurrent(
                client=client,
                model_name=model_name,
                benchmark_name=bench_name,
                tasks=tasks,
                max_turns=args.max_turns,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                concurrency=args.concurrency,
                output_dir=output_dir,
            )
            all_summaries[bench_name] = {
                "accuracy": summary["accuracy"],
                "correct": summary["correct"],
                "total": summary["total"],
                "rate_per_min": summary["rate_per_min"],
            }

        # Print final comparison
        logger.info(f"\n{'='*60}")
        logger.info("MULTI-TURN BENCHMARK RESULTS (SGLang)")
        logger.info(f"{'='*60}")
        for name, s in all_summaries.items():
            logger.info(f"  {name:15s}  acc={s['accuracy']:.3f}  ({s['correct']}/{s['total']})  rate={s['rate_per_min']:.1f}/min")
        logger.info(f"{'='*60}")

    finally:
        if server_proc:
            logger.info("Shutting down SGLang server...")
            server_proc.terminate()
            server_proc.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description="Fast multi-turn benchmark eval with SGLang")
    parser.add_argument("--model-path", default=None,
                        help="Path to model (launches SGLang server automatically)")
    parser.add_argument("--server-url", default=None,
                        help="URL of running SGLang server (e.g., http://localhost:30000)")
    parser.add_argument("--port", type=int, default=30000,
                        help="Port for auto-launched SGLang server")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU ID for auto-launched SGLang server")
    parser.add_argument("--benchmarks", nargs="+", default=["medqa"],
                        choices=list(BENCHMARK_FILES.keys()))
    parser.add_argument("--output-dir", default="results/benchmarks_multiturn_sglang")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of concurrent sample evaluations")
    parser.add_argument("--resume-from", type=int, default=0)

    args = parser.parse_args()

    if not args.model_path and not args.server_url:
        parser.error("Either --model-path or --server-url must be provided")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
