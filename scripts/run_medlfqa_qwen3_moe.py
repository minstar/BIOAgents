#!/usr/bin/env python3
"""OLAPH MedLFQA Evaluation — Qwen3-30B-A3B (MoE)

Standalone script for evaluating the Qwen3-30B-A3B MoE model on all 5 MedLFQA datasets.
Uses left-padding + batch inference for efficiency.

Usage:
    # Basic (uses default paths, GPU 0,1)
    CUDA_VISIBLE_DEVICES=0,1 python scripts/run_medlfqa_qwen3_moe.py

    # Custom paths and GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_medlfqa_qwen3_moe.py \
        --model_path /data/project/public/checkpoints/Qwen3-30B-A3B \
        --data_dir evaluations/OLAPH/MedLFQA \
        --batch_size 4 \
        --output_dir results/medlfqa_qwen3_moe

Requirements:
    pip install torch transformers accelerate
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MEDLFQA_DATASETS = {
    "kqa_golden": ("kqa_golden_test_MedLFQA.jsonl", "KQA Golden"),
    "live_qa": ("live_qa_test_MedLFQA.jsonl", "LiveQA"),
    "medication_qa": ("medication_qa_test_MedLFQA.jsonl", "MedicationQA"),
    "healthsearch_qa": ("healthsearch_qa_test_MedLFQA.jsonl", "HealthSearchQA"),
    "kqa_silver": ("kqa_silver_wogold_test_MedLFQA.jsonl", "KQA Silver"),
}


def compute_rouge_l(pred: str, ref: str) -> float:
    pt, rt = pred.lower().split(), ref.lower().split()
    if not pt or not rt:
        return 0.0
    m, n = len(pt), len(rt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if pt[i - 1] == rt[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
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


def compute_coverage(pred: str, items: list) -> float:
    if not items:
        return 1.0
    pl = pred.lower()
    covered = 0
    for item in items:
        words = [w for w in item.lower().split() if len(w) > 3]
        if words and sum(1 for w in words if w in pl) / len(words) >= 0.5:
            covered += 1
        elif not words and item.lower() in pl:
            covered += 1
    return covered / len(items)


def load_dataset(data_dir: Path, filename: str) -> list:
    path = data_dir / filename
    if not path.exists():
        print(f"[WARN] Not found: {path}", flush=True)
        return []
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_path
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size

    model_name = Path(model_path).name

    print(f"\n{'=' * 70}", flush=True)
    print(f"  OLAPH MedLFQA Evaluation", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"  Path:  {model_path}", flush=True)
    print(f"  Data:  {data_dir}", flush=True)
    print(f"  Batch: {batch_size}", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    print("[Loading model...]", flush=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[Model loaded in {time.time() - t_load:.1f}s]", flush=True)
    print(f"[Model type: {model.config.model_type}, dtype: {model.dtype}]", flush=True)

    checkpoint_path = output_dir / f"medlfqa_{model_name}_checkpoint.json"
    completed = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        print(f"[Resuming from checkpoint: {list(completed.keys())}]", flush=True)

    all_results = dict(completed)
    total_start = time.time()

    for ds_key, (filename, ds_name) in MEDLFQA_DATASETS.items():
        if ds_key in completed:
            print(f"\n[SKIP] {ds_name} (already completed)", flush=True)
            continue

        data = load_dataset(data_dir, filename)
        if not data:
            all_results[ds_key] = {"error": "No data"}
            continue

        print(f"\n{'─' * 50}", flush=True)
        print(f"  {ds_name}: {len(data)} examples (batch_size={batch_size})", flush=True)
        print(f"{'─' * 50}", flush=True)

        valid_items = [(i, item) for i, item in enumerate(data) if item.get("Question", "")]
        prompts = []
        for _, item in valid_items:
            q = item["Question"]
            messages = [
                {"role": "system", "content": "You are a medical expert. Provide detailed, accurate, evidence-based answers."},
                {"role": "user", "content": f"Question: {q}\n\nProvide a comprehensive answer."},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)

        t0 = time.time()
        responses = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]

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
            if done % (batch_size * 5) == 0 or done == len(prompts):
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                print(f"  Progress: {done}/{len(prompts)} ({speed:.1f} ex/s, {elapsed:.0f}s)", flush=True)

        metrics_sum = {"rouge_l": 0, "token_f1": 0, "must_have": 0, "nice_to_have": 0}
        n = 0
        per_question = []

        for idx, (data_idx, item) in enumerate(valid_items):
            if idx >= len(responses):
                break
            ref = item.get("Free_form_answer", "")
            mh = item.get("Must_have", [])
            nh = item.get("Nice_to_have", [])
            resp = responses[idx]

            rl = compute_rouge_l(resp, ref)
            tf1 = compute_token_f1(resp, ref)
            mh_cov = compute_coverage(resp, mh)
            nh_cov = compute_coverage(resp, nh)

            metrics_sum["rouge_l"] += rl
            metrics_sum["token_f1"] += tf1
            metrics_sum["must_have"] += mh_cov
            metrics_sum["nice_to_have"] += nh_cov
            n += 1

            per_question.append({
                "idx": data_idx,
                "question": item.get("Question", "")[:100],
                "rouge_l": round(rl, 4),
                "token_f1": round(tf1, 4),
                "must_have_cov": round(mh_cov, 4),
                "nice_to_have_cov": round(nh_cov, 4),
            })

        elapsed = time.time() - t0
        avg = {k: v / n for k, v in metrics_sum.items()} if n > 0 else metrics_sum

        all_results[ds_key] = {
            "name": ds_name,
            "total": n,
            **{k: round(v, 4) for k, v in avg.items()},
            "time_sec": round(elapsed, 1),
            "speed_ex_per_sec": round(n / elapsed, 1) if elapsed > 0 else 0,
        }

        print(f"\n  >> {ds_name} Results:", flush=True)
        print(f"     ROUGE-L:    {avg.get('rouge_l', 0):.4f}", flush=True)
        print(f"     Token-F1:   {avg.get('token_f1', 0):.4f}", flush=True)
        print(f"     Must-Have:  {avg.get('must_have', 0):.4f}", flush=True)
        print(f"     Nice-Have:  {avg.get('nice_to_have', 0):.4f}", flush=True)
        print(f"     ({n} examples, {elapsed:.0f}s, {n / elapsed:.1f} ex/s)", flush=True)

        with open(checkpoint_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - total_start

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"medlfqa_{model_name}_{ts}.json"

    valid_results = {k: v for k, v in all_results.items() if isinstance(v, dict) and "error" not in v and "total" in v}

    if valid_results:
        overall_n = sum(v["total"] for v in valid_results.values())
        overall_rl = sum(v["rouge_l"] * v["total"] for v in valid_results.values()) / overall_n if overall_n > 0 else 0
        overall_tf1 = sum(v["token_f1"] * v["total"] for v in valid_results.values()) / overall_n if overall_n > 0 else 0
        overall_mh = sum(v["must_have"] * v["total"] for v in valid_results.values()) / overall_n if overall_n > 0 else 0
        overall_nh = sum(v["nice_to_have"] * v["total"] for v in valid_results.values()) / overall_n if overall_n > 0 else 0
    else:
        overall_n = overall_rl = overall_tf1 = overall_mh = overall_nh = 0

    report = {
        "model_name": model_name,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "total_time_sec": round(total_elapsed, 1),
        "batch_size": batch_size,
        "overall": {
            "total_examples": overall_n,
            "rouge_l": round(overall_rl, 4),
            "token_f1": round(overall_tf1, 4),
            "must_have": round(overall_mh, 4),
            "nice_to_have": round(overall_nh, 4),
        },
        "per_dataset": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'=' * 70}", flush=True)
    print(f"  OLAPH MedLFQA FINAL RESULTS: {model_name}", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  {'Dataset':<20} {'ROUGE-L':>10} {'Token-F1':>10} {'Must-Have':>10} {'Nice-Have':>10} {'N':>6}", flush=True)
    print(f"  {'─' * 66}", flush=True)
    for key, r in all_results.items():
        if isinstance(r, dict) and "error" not in r and "total" in r:
            print(f"  {r['name']:<20} {r['rouge_l']:>10.4f} {r['token_f1']:>10.4f} "
                  f"{r['must_have']:>10.4f} {r['nice_to_have']:>10.4f} {r['total']:>6}", flush=True)
    print(f"  {'─' * 66}", flush=True)
    print(f"  {'OVERALL (weighted)':<20} {overall_rl:>10.4f} {overall_tf1:>10.4f} "
          f"{overall_mh:>10.4f} {overall_nh:>10.4f} {overall_n:>6}", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_path}", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return report


def main():
    parser = argparse.ArgumentParser(description="OLAPH MedLFQA Eval — Qwen3-30B-A3B (MoE)")
    parser.add_argument("--model_path", type=str,
                        default="/data/project/public/checkpoints/Qwen3-30B-A3B")
    parser.add_argument("--data_dir", type=str,
                        default="evaluations/OLAPH/MedLFQA")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (MoE uses more memory, default 4)")
    parser.add_argument("--output_dir", type=str,
                        default="results/medlfqa_qwen3_moe")
    args = parser.parse_args()

    os.environ["PYTHONUNBUFFERED"] = "1"

    evaluate(args)


if __name__ == "__main__":
    main()
