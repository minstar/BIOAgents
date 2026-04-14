#!/usr/bin/env python3
"""MedLFQA benchmark with training-aligned tool-use format.

Uses the same prompt format as GRPO training:
- System prompt with tool instructions
- apply_chat_template(messages, tools=openai_tools)
- Model responds with submit_answer tool call
- Parse answer from tool call JSON

This gives a fair evaluation of RL-trained models that learned to use tools
for long-form medical question answering.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_medlfqa_tooluse_format.py \
        --model_path /path/to/merged_hf \
        --output-dir results/benchmarks_tooluse/v6_step80
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Tool definitions matching training format ──
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
                    "description": "Your comprehensive, evidence-based answer to the medical question.",
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

# Tools passed to apply_chat_template
OPENAI_TOOLS = [SUBMIT_ANSWER_TOOL]

# System prompt for long-form medical QA
SYSTEM_PROMPT = (
    "You are a medical expert. Provide detailed, accurate, evidence-based answers to medical questions. "
    "Submit your answer by calling the submit_answer tool with a comprehensive response."
)

# ── MedLFQA datasets ──
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
    """Load MedLFQA dataset from JSONL file."""
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


# ── Metrics (copied from run_full_benchmark_suite.py) ──

def compute_rouge_l(prediction, reference):
    """Compute ROUGE-L score using LCS."""
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
    """Compute token-level F1 score (bag-of-words overlap)."""
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


def compute_must_have_coverage(prediction, must_have):
    """Compute coverage of must-have key terms."""
    if not must_have:
        return 1.0
    pred_lower = prediction.lower()
    covered = 0
    for item in must_have:
        words = [w for w in item.lower().split() if len(w) > 3]
        if words:
            match_count = sum(1 for w in words if w in pred_lower)
            if match_count / len(words) >= 0.5:
                covered += 1
        elif item.lower() in pred_lower:
            covered += 1
    return covered / len(must_have)


def compute_nice_to_have_coverage(prediction, nice_to_have):
    """Compute coverage of nice-to-have terms."""
    if not nice_to_have:
        return 1.0
    pred_lower = prediction.lower()
    covered = 0
    for item in nice_to_have:
        words = [w for w in item.lower().split() if len(w) > 3]
        if words:
            match_count = sum(1 for w in words if w in pred_lower)
            if match_count / len(words) >= 0.5:
                covered += 1
        elif item.lower() in pred_lower:
            covered += 1
    return covered / len(nice_to_have)


# ── Answer extraction from tool call response ──

def extract_answer_from_tool_call(response: str) -> str:
    """Extract long-form answer from model response with tool call format.

    Handles both Qwen3.5 XML-style and JSON-style tool calls:
      XML: <parameter=answer>VALUE</parameter>
      JSON: {"name": "submit_answer", "arguments": {"answer": "..."}}
    """
    # 0. Strip think tags first
    clean = response
    if '<think>' in clean:
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split("</think>")[-1].strip()

    # 1. Try Qwen3.5 XML-style: <parameter=answer>VALUE</parameter>
    xml_answer = re.search(r'<parameter=answer>\s*(.*?)\s*</parameter>', clean, re.DOTALL)
    if xml_answer:
        answer = xml_answer.group(1).strip()
        # Also try to get reasoning
        xml_reasoning = re.search(r'<parameter=reasoning>\s*(.*?)\s*</parameter>', clean, re.DOTALL)
        if xml_reasoning:
            return answer + "\n" + xml_reasoning.group(1).strip()
        return answer

    # 2. Try JSON-style: {"name": "submit_answer", "arguments": {"answer": "..."}}
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
                    if reasoning:
                        return answer + "\n" + reasoning
                    return answer
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    # 3. Try broader JSON parsing — find any JSON with "answer" key
    try:
        json_blocks = re.findall(r'\{[^{}]+\}', clean)
        for block in json_blocks:
            try:
                obj = json.loads(block)
                if "answer" in obj:
                    answer = obj["answer"]
                    reasoning = obj.get("reasoning", "")
                    if reasoning:
                        return answer + "\n" + reasoning
                    return answer
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    # 4. Fallback: return cleaned text (XML markup stripped)
    clean = re.sub(r'<tool_call>.*?</tool_call>', '', clean, flags=re.DOTALL).strip()
    clean = re.sub(r'<function=.*?</function>', '', clean, flags=re.DOTALL).strip()
    return clean


def main():
    import argparse
    import torch
    from transformers import AutoConfig, AutoTokenizer

    parser = argparse.ArgumentParser(description="MedLFQA with training-aligned tool-use format")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/benchmarks_tooluse")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size (0=auto)")
    args = parser.parse_args()

    # Only set CUDA_VISIBLE_DEVICES if not already set by shell (e.g., parallel launch)
    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["PYTHONUNBUFFERED"] = "1"

    model_path = args.model_path
    model_name = Path(model_path).name

    print(f"\n{'#'*70}", flush=True)
    print(f"  MedLFQA (Tool-Use Format) — Training-Aligned Evaluation", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"  Format: apply_chat_template(tools=openai_tools)", flush=True)
    print(f"{'#'*70}\n", flush=True)

    # Load model
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(model_config, "model_type", "")
    is_qwen3_5 = model_type == "qwen3_5"

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
        attn_implementation="eager",
    )

    if is_qwen3_5:
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"  # Critical for correct batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test that apply_chat_template works with tools
    test_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Test question"},
    ]
    try:
        test_text = tokenizer.apply_chat_template(
            test_messages, tools=OPENAI_TOOLS,
            tokenize=False, add_generation_prompt=True,
        )
        print(f"  Chat template with tools: OK ({len(test_text)} chars)", flush=True)
        print(f"  Preview: {test_text[:500]}...", flush=True)
    except Exception as e:
        print(f"  WARNING: apply_chat_template with tools failed: {e}", flush=True)
        print(f"  Falling back to tools-in-system-prompt format", flush=True)
        OPENAI_TOOLS.clear()

    # Build EOS token IDs for early stopping
    eos_ids = [tokenizer.eos_token_id]
    for special in ["</tool_call>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(special)
        if tid is not None and tid != tokenizer.unk_token_id:
            eos_ids.append(tid)

    BATCH_SIZE = args.batch_size if args.batch_size > 0 else (4 if is_qwen3_5 else 8)
    print(f"  Batch size: {BATCH_SIZE}", flush=True)
    print(f"  Max new tokens: 1024", flush=True)
    print(f"  EOS token IDs: {eos_ids}", flush=True)

    all_results = {}

    for dataset_key, dataset_info in MEDLFQA_DATASETS.items():
        print(f"\n[{model_name}] Evaluating {dataset_info['name']}...", flush=True)
        data = load_medlfqa_data(dataset_key, max_samples=args.max_samples)
        if not data:
            all_results[dataset_key] = {"error": "No data"}
            continue

        print(f"  Loaded {len(data)} examples", flush=True)
        metrics_sum = {"rouge_l": 0, "token_f1": 0, "must_have": 0, "nice_to_have": 0}
        per_question = []
        t0 = time.time()

        valid_items = [(i, item) for i, item in enumerate(data) if item.get("Question", "")]

        for batch_start in range(0, len(valid_items), BATCH_SIZE):
            batch = valid_items[batch_start:batch_start + BATCH_SIZE]
            batch_prompts = []
            batch_refs = []

            for idx, item in batch:
                question = item.get("Question", "")
                reference = item.get("Free_form_answer", "")
                must_have = item.get("Must_have", [])
                nice_to_have = item.get("Nice_to_have", [])
                batch_refs.append((idx, reference, must_have, nice_to_have))

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}\n\nProvide a comprehensive answer."},
                ]

                try:
                    if OPENAI_TOOLS:
                        text = tokenizer.apply_chat_template(
                            messages, tools=OPENAI_TOOLS,
                            tokenize=False, add_generation_prompt=True,
                        )
                    else:
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                        )
                except Exception:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )

                batch_prompts.append(text)

            if not batch_prompts:
                continue

            # Batch tokenize with left-padding
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", truncation=True,
                max_length=4096, padding=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )

            # Decode each response in the batch
            for j, (idx, reference, must_have, nice_to_have) in enumerate(batch_refs):
                input_len = inputs["input_ids"].shape[-1]
                generated = outputs[j][input_len:]
                response = tokenizer.decode(generated, skip_special_tokens=True).strip()

                # Extract answer from tool call format
                prediction = extract_answer_from_tool_call(response)

                rouge_l = compute_rouge_l(prediction, reference)
                token_f1 = compute_token_f1(prediction, reference)
                mh_cov = compute_must_have_coverage(prediction, must_have)
                nth_cov = compute_nice_to_have_coverage(prediction, nice_to_have)

                metrics_sum["rouge_l"] += rouge_l
                metrics_sum["token_f1"] += token_f1
                metrics_sum["must_have"] += mh_cov
                metrics_sum["nice_to_have"] += nth_cov
                per_question.append({
                    "idx": idx,
                    "rouge_l": rouge_l,
                    "token_f1": token_f1,
                    "must_have_coverage": mh_cov,
                    "nice_to_have_coverage": nth_cov,
                })

            done = batch_start + len(batch)
            if done % 50 < BATCH_SIZE or done == len(valid_items):
                elapsed = time.time() - t0
                avg_rl = metrics_sum["rouge_l"] / max(len(per_question), 1)
                print(
                    f"  [{model_name}] {dataset_info['name']}: {done}/{len(valid_items)} "
                    f"ROUGE-L={avg_rl:.3f} {elapsed:.0f}s",
                    flush=True,
                )

        n = len(per_question)
        if n == 0:
            all_results[dataset_key] = {"error": "No valid questions"}
            continue

        avg_metrics = {k: v / n for k, v in metrics_sum.items()}
        all_results[dataset_key] = {
            "name": dataset_info["name"],
            "total": n,
            **avg_metrics,
        }
        elapsed = time.time() - t0
        print(
            f"  [{model_name}] {dataset_info['name']}: ROUGE-L={avg_metrics['rouge_l']:.3f} "
            f"Token-F1={avg_metrics['token_f1']:.3f} Must-Have={avg_metrics['must_have']:.3f} "
            f"Nice-to-Have={avg_metrics['nice_to_have']:.3f} ({n} examples, {elapsed:.0f}s)",
            flush=True,
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"medlfqa_tooluse_{ts}.json"
    report = {
        "model_name": model_name,
        "model_path": model_path,
        "format": "tool-use (training-aligned)",
        "system_prompt": SYSTEM_PROMPT,
        "tools": [t["function"]["name"] for t in OPENAI_TOOLS],
        "timestamp": datetime.now().isoformat(),
        "category": "medlfqa_tooluse",
        "benchmarks": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_question"}
            if isinstance(v, dict) else v
            for k, v in all_results.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*70}", flush=True)
    print(f"  MedLFQA (Tool-Use Format) RESULTS: {model_name}", flush=True)
    print(
        f"  {'Dataset':<20} {'ROUGE-L':>10} {'Token-F1':>10} "
        f"{'Must-Have':>10} {'Nice-Have':>10} {'Total':>6}",
        flush=True,
    )
    print(f"  {'-'*66}", flush=True)
    for key, r in all_results.items():
        if "error" in r:
            print(f"  {key:<20} {'ERROR':>10}", flush=True)
        else:
            print(
                f"  {r.get('name', key):<20} {r['rouge_l']:>10.4f} {r['token_f1']:>10.4f} "
                f"{r['must_have']:>10.4f} {r['nice_to_have']:>10.4f} {r['total']:>6}",
                flush=True,
            )
    print(f"\n  Saved: {out_path}", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
