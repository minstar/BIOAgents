#!/usr/bin/env python3
"""Optimized Full Benchmark Evaluation — Batch Inference, No Sample Limits.

Key optimizations:
1. Batch inference (batch_size=8-16) for 3-5x speedup
2. Left-padding for causal LMs
3. No max_samples limit — evaluates ALL test data
4. Subprocess parallelism across GPUs
5. Progress checkpointing (resume on failure)

Usage:
    # Run everything in parallel across GPUs
    python scripts/run_optimized_eval.py --mode parallel

    # Run specific category for specific model
    python scripts/run_optimized_eval.py --mode single --category medlfqa --model qwen3 --gpus 0,1
    python scripts/run_optimized_eval.py --mode single --category vqa --model qwen2vl --gpus 2,3
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).parent.parent

MODELS = {
    "qwen3": {
        "name": "Qwen3-8B-Base",
        "path": "/data/project/private/minstar/models/Qwen3-8B-Base",
        "type": "causal",
        "supports_vision": False,
    },
    "qwen2vl": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "path": str(PROJECT_ROOT / "checkpoints/models/Qwen2.5-VL-7B-Instruct"),
        "type": "vlm",
        "supports_vision": True,
    },
    "lingshu": {
        "name": "Lingshu-7B",
        "path": str(PROJECT_ROOT / "checkpoints/models/Lingshu-7B"),
        "type": "vlm",
        "supports_vision": True,
    },
}

MEDLFQA_DATASETS = {
    "kqa_golden": ("evaluations/OLAPH/MedLFQA/kqa_golden_test_MedLFQA.jsonl", "KQA Golden"),
    "live_qa": ("evaluations/OLAPH/MedLFQA/live_qa_test_MedLFQA.jsonl", "LiveQA"),
    "medication_qa": ("evaluations/OLAPH/MedLFQA/medication_qa_test_MedLFQA.jsonl", "MedicationQA"),
    "healthsearch_qa": ("evaluations/OLAPH/MedLFQA/healthsearch_qa_test_MedLFQA.jsonl", "HealthSearchQA"),
    "kqa_silver": ("evaluations/OLAPH/MedLFQA/kqa_silver_wogold_test_MedLFQA.jsonl", "KQA Silver"),
}

# ──────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────

def compute_rouge_l(pred: str, ref: str) -> float:
    pt, rt = pred.lower().split(), ref.lower().split()
    if not pt or not rt:
        return 0.0
    m, n = len(pt), len(rt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if pt[i-1] == rt[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p, r = lcs / m, lcs / n
    return 2 * p * r / (p + r)

def compute_token_f1(pred: str, ref: str) -> float:
    pt, rt = set(pred.lower().split()), set(ref.lower().split())
    if not pt or not rt:
        return 1.0 if pt == rt else 0.0
    common = pt & rt
    if not common:
        return 0.0
    p, r = len(common) / len(pt), len(common) / len(rt)
    return 2 * p * r / (p + r)

def compute_must_have(pred: str, must_have: list) -> float:
    if not must_have:
        return 1.0
    pl = pred.lower()
    covered = 0
    for item in must_have:
        words = [w for w in item.lower().split() if len(w) > 3]
        if words and sum(1 for w in words if w in pl) / len(words) >= 0.5:
            covered += 1
        elif not words and item.lower() in pl:
            covered += 1
    return covered / len(must_have)

# ──────────────────────────────────────────────────────────────
#  Batch MedLFQA Evaluator
# ──────────────────────────────────────────────────────────────

def batch_generate(model, tokenizer, prompts, max_new_tokens=512, batch_size=8):
    """Batch generate responses with left-padding."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
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


def evaluate_medlfqa_batched(model_key: str, output_dir: Path, batch_size: int = 8):
    """Evaluate MedLFQA with batch inference — ALL data, no limits."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_info = MODELS[model_key]
    model_name = model_info["name"]
    model_path = model_info["path"]

    print(f"\n{'='*70}", flush=True)
    print(f"  MedLFQA Batched Evaluation: {model_name} (batch_size={batch_size})", flush=True)
    print(f"{'='*70}", flush=True)

    # Load model
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(model_config, "model_type", "")
    is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl")

    load_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", attn_implementation="sdpa")

    if is_qwen_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[{model_name}] Model loaded", flush=True)

    # Check for checkpoint
    checkpoint_path = output_dir / f"medlfqa_{model_key}_checkpoint.json"
    completed_datasets = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed_datasets = json.load(f)
        print(f"[{model_name}] Resuming from checkpoint: {list(completed_datasets.keys())}", flush=True)

    all_results = dict(completed_datasets)

    for dataset_key, (rel_path, ds_name) in MEDLFQA_DATASETS.items():
        if dataset_key in completed_datasets:
            print(f"[{model_name}] Skipping {ds_name} (already completed)", flush=True)
            continue

        data_path = PROJECT_ROOT / rel_path
        if not data_path.exists():
            print(f"[WARN] Not found: {data_path}", flush=True)
            all_results[dataset_key] = {"error": "No data"}
            continue

        data = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        print(f"\n[{model_name}] Evaluating {ds_name} ({len(data)} examples, batch_size={batch_size})...", flush=True)

        # Prepare all prompts
        valid_indices = []
        prompts = []
        for i, item in enumerate(data):
            q = item.get("Question", "")
            if not q:
                continue
            valid_indices.append(i)
            messages = [
                {"role": "system", "content": "You are a medical expert. Provide detailed, accurate, evidence-based answers."},
                {"role": "user", "content": f"Question: {q}\n\nProvide a comprehensive answer."},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)

        # Batch generate
        t0 = time.time()
        print(f"  Generating {len(prompts)} responses...", flush=True)

        responses = []
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

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
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for j in range(len(batch_prompts)):
                input_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
                generated = outputs[j][input_len:]
                resp = tokenizer.decode(generated, skip_special_tokens=True).strip()
                responses.append(resp)

            done = len(responses)
            if done % (batch_size * 10) == 0 or done == len(prompts):
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                print(f"  [{model_name}] {ds_name}: {done}/{len(prompts)} ({speed:.1f} ex/s, {elapsed:.0f}s)", flush=True)

        # Compute metrics
        metrics_sum = {"rouge_l": 0, "token_f1": 0, "must_have": 0, "nice_to_have": 0}
        n = 0
        for idx, resp_idx in enumerate(valid_indices):
            if idx >= len(responses):
                break
            item = data[resp_idx]
            ref = item.get("Free_form_answer", "")
            mh = item.get("Must_have", [])
            nh = item.get("Nice_to_have", [])
            resp = responses[idx]

            metrics_sum["rouge_l"] += compute_rouge_l(resp, ref)
            metrics_sum["token_f1"] += compute_token_f1(resp, ref)
            metrics_sum["must_have"] += compute_must_have(resp, mh)
            metrics_sum["nice_to_have"] += compute_must_have(resp, nh)
            n += 1

        elapsed = time.time() - t0
        avg = {k: v / n for k, v in metrics_sum.items()} if n > 0 else metrics_sum

        all_results[dataset_key] = {"name": ds_name, "total": n, **avg, "time_sec": elapsed}
        print(f"  [{model_name}] {ds_name}: ROUGE-L={avg.get('rouge_l',0):.3f} "
              f"Token-F1={avg.get('token_f1',0):.3f} Must-Have={avg.get('must_have',0):.3f} "
              f"({n} examples, {elapsed:.0f}s, {n/elapsed:.1f} ex/s)", flush=True)

        # Save checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save final results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"medlfqa_{model_key}_full_{ts}.json"
    report = {
        "model_name": model_name, "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "category": "medlfqa", "batch_size": batch_size,
        "benchmarks": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}", flush=True)
    print(f"  MedLFQA FINAL RESULTS: {model_name}", flush=True)
    print(f"  {'Dataset':<20} {'ROUGE-L':>10} {'Token-F1':>10} {'Must-Have':>10} {'N':>6} {'Speed':>10}", flush=True)
    print(f"  {'-'*66}", flush=True)
    for key, r in all_results.items():
        if isinstance(r, dict) and "error" not in r and "total" in r:
            spd = f"{r['total']/r.get('time_sec',1):.1f} ex/s" if r.get('time_sec') else ""
            print(f"  {r['name']:<20} {r['rouge_l']:>10.3f} {r['token_f1']:>10.3f} {r['must_have']:>10.3f} {r['total']:>6} {spd:>10}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Saved: {out_path}", flush=True)

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    del model
    torch.cuda.empty_cache()
    return all_results


def evaluate_vqa_full(model_key: str, output_dir: Path, batch_size: int = 4):
    """Evaluate VQA with full data — no sample limits."""
    model_info = MODELS[model_key]
    if not model_info["supports_vision"]:
        print(f"[SKIP] {model_info['name']} does not support vision.", flush=True)
        return {}

    sys.path.insert(0, str(PROJECT_ROOT))
    from bioagents.evaluation.vqa_benchmark_eval import VQABenchmarkConfig, VQABenchmarkEvaluator

    vqa_config = VQABenchmarkConfig(
        model_name_or_path=model_info["path"],
        model_name=model_info["name"],
        benchmarks=["vqa_rad", "slake", "pathvqa", "pmc_vqa", "vqa_med_2021", "quilt_vqa"],
        max_samples=0,  # ALL data
        batch_size=batch_size,
        output_dir=str(output_dir / "vqa"),
        use_images=True,
    )
    evaluator = VQABenchmarkEvaluator(vqa_config)
    results = evaluator.evaluate_all()
    del evaluator
    torch.cuda.empty_cache()
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimized Full Benchmark Evaluation")
    parser.add_argument("--mode", choices=["single", "parallel"], default="single")
    parser.add_argument("--category", choices=["vqa", "medlfqa", "all"], default="all")
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="logs/full_baseline")
    args = parser.parse_args()

    if args.mode == "parallel":
        # Launch all evaluations in parallel using subprocess
        output_dir = PROJECT_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        jobs = [
            # MedLFQA for all 3 models
            ("medlfqa", "qwen3", "0,1", 8),
            ("medlfqa", "lingshu", "4,5", 8),
            ("medlfqa", "qwen2vl", "6,7", 8),
            # VQA for VL models
            ("vqa", "qwen2vl", "2,3", 4),
            ("vqa", "lingshu", "2,3", 4),  # Will wait for qwen2vl VQA to finish
        ]

        processes = []
        vqa_procs = []

        for category, model, gpus, bs in jobs:
            if category == "vqa":
                # VQA jobs run sequentially on same GPU
                vqa_procs.append((category, model, gpus, bs))
                continue

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpus
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [
                sys.executable, __file__,
                "--mode", "single",
                "--category", category,
                "--model", model,
                "--gpus", gpus,
                "--batch-size", str(bs),
                "--output-dir", args.output_dir,
            ]
            log_path = output_dir / f"{category}_{model}_full.log"
            with open(log_path, "w") as log_f:
                p = subprocess.Popen(
                    cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT)
                )
            processes.append((f"{category}_{model}", p, log_path))
            print(f"  Launched {category}/{model} on GPU {gpus} (PID {p.pid})", flush=True)

        # Launch VQA jobs sequentially
        for category, model, gpus, bs in vqa_procs:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpus
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [
                sys.executable, __file__,
                "--mode", "single",
                "--category", category,
                "--model", model,
                "--gpus", gpus,
                "--batch-size", str(bs),
                "--output-dir", args.output_dir,
            ]
            log_path = output_dir / f"{category}_{model}_full.log"
            with open(log_path, "w") as log_f:
                p = subprocess.Popen(
                    cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT)
                )
            processes.append((f"{category}_{model}", p, log_path))
            print(f"  Launched {category}/{model} on GPU {gpus} (PID {p.pid})", flush=True)

        print(f"\n  Total {len(processes)} evaluation jobs launched.", flush=True)
        print(f"  Monitoring progress...\n", flush=True)

        # Wait for all
        for name, p, log_path in processes:
            p.wait()
            status = "OK" if p.returncode == 0 else f"FAILED (code={p.returncode})"
            print(f"  {name}: {status}", flush=True)

        print(f"\nAll evaluations complete.", flush=True)
        return

    # Single mode — only set CUDA_VISIBLE_DEVICES if not already set by parent
    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["PYTHONUNBUFFERED"] = "1"

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = MODELS[args.model]
    print(f"\n{'#'*70}", flush=True)
    print(f"  Optimized Evaluation — {model_info['name']}", flush=True)
    print(f"  Category: {args.category} | GPUs: {args.gpus} | Batch: {args.batch_size}", flush=True)
    print(f"{'#'*70}\n", flush=True)

    if args.category in ("vqa", "all"):
        evaluate_vqa_full(args.model, output_dir, args.batch_size)

    if args.category in ("medlfqa", "all"):
        evaluate_medlfqa_batched(args.model, output_dir, args.batch_size)

    print(f"\nEVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
