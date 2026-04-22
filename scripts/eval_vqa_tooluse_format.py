#!/usr/bin/env python3
"""VQA benchmark with training-aligned tool-use format.

Uses the same prompt format as GRPO training:
- System prompt with tool instructions
- apply_chat_template(messages, tools=openai_tools)
- Model responds with submit_answer tool call
- Parse answer from tool call JSON

This gives a fair evaluation of RL-trained VL models that learned to use tools.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_vqa_tooluse_format.py \
        --model_path /path/to/merged_hf \
        --output-dir results/benchmarks_tooluse/v6_step80
"""

import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False  # Workaround for cuDNN init failure with Qwen3.5-VL Conv3D

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

# Tools passed to apply_chat_template — single placeholder to trigger tool-call instructions
OPENAI_TOOLS = [SUBMIT_ANSWER_TOOL]

# Load training-aligned system prompt with full domain tools
_TRAINING_PROMPTS_PATH = Path(__file__).parent / "verl" / "training_system_prompts.json"
if _TRAINING_PROMPTS_PATH.exists():
    with open(_TRAINING_PROMPTS_PATH) as _f:
        _TRAINING_PROMPTS = json.load(_f)
    SYSTEM_PROMPT = _TRAINING_PROMPTS.get("visual_diagnosis", "")
    print(f"  Loaded training system prompt: visual_diagnosis ({len(SYSTEM_PROMPT)} chars)", flush=True)
else:
    SYSTEM_PROMPT = (
        "You are a medical imaging expert. Analyze the medical image and answer the question. "
        "Submit your answer by calling the submit_answer tool."
    )
    print(f"  WARNING: training_system_prompts.json not found, using fallback prompt", flush=True)

# VQA benchmarks to evaluate
VQA_BENCHMARKS = ["vqa_rad", "slake", "pathvqa", "pmc_vqa", "vqa_med_2021", "quilt_vqa"]


def extract_answer_from_tool_call(response: str) -> str:
    """Extract answer from model response — handles both XML and JSON tool call formats.

    Qwen3.5 uses XML-style tool calls:
        <tool_call>
        <function=submit_answer>
        <parameter=answer>yes</parameter>
        </function>
        </tool_call>
    """
    # 0. Strip think tags first
    clean = response
    if '<think>' in clean:
        clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split('</think>')[-1].strip()

    # 1. Try Qwen3.5 XML-style: <parameter=answer>VALUE</parameter>
    xml_match = re.search(
        r'<parameter=answer>\s*(.*?)\s*</parameter>', clean, re.DOTALL
    )
    if xml_match:
        return xml_match.group(1).strip()

    # 2. Try XML-style with newlines: <parameter=answer>\nVALUE\n</parameter>
    xml_match2 = re.search(
        r'<parameter=answer>(.*?)</parameter>', clean, re.DOTALL
    )
    if xml_match2:
        return xml_match2.group(1).strip()

    # 3. Try JSON-style tool call: {"name": "submit_answer", "arguments": {"answer": "..."}}
    try:
        json_patterns = re.findall(
            r'\{[^{}]*"name"\s*:\s*"submit_answer"[^{}]*\}', clean, re.DOTALL
        )
        if not json_patterns:
            json_patterns = re.findall(
                r'\{.*?"submit_answer".*?\}', clean, re.DOTALL
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

    # 4. Try to find "answer": "..." pattern
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', clean)
    if m:
        return m.group(1).strip()

    # 5. Try nested arguments JSON block
    try:
        args_match = re.search(r'"arguments"\s*:\s*(\{[^{}]+\})', clean)
        if args_match:
            args_obj = json.loads(args_match.group(1))
            if "answer" in args_obj:
                return str(args_obj["answer"]).strip()
    except Exception:
        pass

    # 6. Fallback: return cleaned text
    for prefix in ["Answer:", "The answer is", "answer:"]:
        if clean.lower().startswith(prefix.lower()):
            clean = clean[len(prefix):].strip()

    return clean


def compute_exact_match(prediction: str, reference: str) -> float:
    """Exact match after normalization."""
    pred = _normalize_answer(prediction)
    ref = _normalize_answer(reference)
    return 1.0 if pred == ref else 0.0


def compute_token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score (precision-recall of word overlap)."""
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
    """Normalize answer text for comparison."""
    text = text.lower().strip()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Collapse whitespace
    text = ' '.join(text.split())
    return text


def main():
    import argparse

    import torch
    from transformers import AutoConfig, AutoProcessor

    parser = argparse.ArgumentParser(description="VQA with training-aligned tool-use format")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/benchmarks_tooluse")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default=1 for VQA)")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking by appending </think> to prompt")
    parser.add_argument("--benchmarks", type=str, default="",
                        help="Comma-separated subset of benchmarks to run (default: all)")
    args = parser.parse_args()

    # Only set CUDA_VISIBLE_DEVICES if not already set by shell
    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["PYTHONUNBUFFERED"] = "1"

    model_path = args.model_path
    model_name = Path(model_path).name

    print(f"\n{'#'*70}", flush=True)
    print(f"  VQA (Tool-Use Format) — Training-Aligned Evaluation", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"  Format: apply_chat_template(tools=openai_tools)", flush=True)
    print(f"{'#'*70}\n", flush=True)

    # Load model config to determine type
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(model_config, "model_type", "")

    print(f"  Model type: {model_type}", flush=True)

    # device_map={"": 0} ensures model loads on a single GPU
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    load_kwargs["attn_implementation"] = "eager"

    # Load model and processor based on type
    if model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration

        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        print(f"  Loaded Qwen3.5-VL with processor", flush=True)
    elif model_type in ("qwen2_5_vl", "qwen2_vl"):
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        print(f"  Loaded Qwen2.5-VL with processor", flush=True)
    else:
        print(f"  ERROR: Unsupported model type '{model_type}' for VQA evaluation", flush=True)
        print(f"  Supported: qwen3_5, qwen2_5_vl, qwen2_vl", flush=True)
        sys.exit(1)

    model.eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params: {param_count:.0f}M", flush=True)

    # Test that apply_chat_template works with tools
    test_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Test"},
    ]
    try:
        test_text = processor.apply_chat_template(
            test_messages, tools=OPENAI_TOOLS, tokenize=False, add_generation_prompt=True
        )
        print(f"  Chat template with tools: OK ({len(test_text)} chars)", flush=True)
        print(f"  Preview: {test_text[:500]}...", flush=True)
    except Exception as e:
        print(f"  WARNING: apply_chat_template with tools failed: {e}", flush=True)
        print(f"  Falling back to tools-in-system-prompt format", flush=True)
        OPENAI_TOOLS.clear()

    # Build EOS token IDs for stopping generation
    eos_ids = [tokenizer.eos_token_id]
    for special in ["<|im_end|>", "</tool_call>"]:
        tid = tokenizer.convert_tokens_to_ids(special)
        if tid is not None and tid != tokenizer.unk_token_id:
            eos_ids.append(tid)
    print(f"  EOS token IDs: {eos_ids}", flush=True)

    # Import VQA data loader
    from bioagents.data_pipeline.vqa_loader import VQA_DATASET_REGISTRY

    # Import vision utils
    from qwen_vl_utils import process_vision_info

    all_results = {}

    benchmarks_to_run = args.benchmarks.split(",") if args.benchmarks else VQA_BENCHMARKS
    for benchmark in benchmarks_to_run:
        if benchmark not in VQA_DATASET_REGISTRY:
            print(f"  [SKIP] Unknown benchmark: {benchmark}", flush=True)
            continue

        info = VQA_DATASET_REGISTRY[benchmark]
        loader = info["loader"]

        # Load dataset — some loaders don't accept split
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
            all_results[benchmark] = {"error": "No data found"}
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"  [{model_name}] {benchmark}: {len(data)} samples", flush=True)
        print(f"{'='*60}", flush=True)

        per_sample_results = []
        metric_sums = Counter()
        t0 = time.time()

        for i, item in enumerate(data):
            question = item["question"]
            reference = item["answer"]
            image_path = item.get("image_path")

            if not question or not reference:
                continue

            # Build messages with image for VL model
            if image_path and os.path.exists(image_path):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{image_path}"},
                            {"type": "text", "text": question},
                        ],
                    },
                ]
            else:
                # Fallback: text-only if no image available
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]

            # Apply chat template with tools
            try:
                if OPENAI_TOOLS:
                    text = processor.apply_chat_template(
                        messages,
                        tools=OPENAI_TOOLS,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            except Exception:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            # Disable thinking: append </think> to skip thinking phase
            if args.no_think:
                # Template ends with "<think>\n", append "</think>\n" to close it
                if text.rstrip().endswith("<think>"):
                    text = text.rstrip() + "\n</think>\n"
                elif "<think>" in text and "</think>" not in text:
                    text += "</think>\n"

            # Process vision info and prepare inputs
            try:
                image_inputs, video_inputs = process_vision_info(messages)
            except Exception:
                image_inputs, video_inputs = None, None

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # Generate response
            max_tokens = 512 if args.no_think else 8192
            t_gen = time.time()
            print(f"  [GEN] #{i} starting (input_tokens={len(text.split())})...", flush=True)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )

            # Decode generated tokens only
            input_len = inputs["input_ids"].shape[-1]
            generated = outputs[0][input_len:]
            gen_len = len(generated)
            response = processor.decode(generated, skip_special_tokens=True).strip()
            print(f"  [GEN] #{i} done in {time.time()-t_gen:.1f}s ({gen_len} tokens)", flush=True)
            del outputs, generated, inputs
            torch.cuda.empty_cache()

            # Extract answer from tool call
            prediction = extract_answer_from_tool_call(response)

            # Compute metrics
            em = compute_exact_match(prediction, reference)
            f1 = compute_token_f1(prediction, reference)

            per_sample_results.append({
                "id": item.get("id", f"{benchmark}_{i}"),
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "raw_response": response[:500],
                "answer_type": item.get("answer_type", "open_ended"),
                "has_image": bool(image_path and os.path.exists(image_path)),
                "metrics": {"exact_match": em, "token_f1": f1},
            })

            metric_sums["exact_match"] += em
            metric_sums["token_f1"] += f1

            # Debug: print first 5 samples + every 50th sample
            if i < 5 or (i + 1) % 50 == 0:
                print(
                    f"  [DBG] #{i} ref='{reference}' pred='{prediction}' "
                    f"raw='{response[:200]}' EM={em} F1={f1:.3f}",
                    flush=True,
                )

            # Progress logging every 50 samples
            if (i + 1) % 50 == 0:
                n = len(per_sample_results)
                avg_em = metric_sums["exact_match"] / n
                avg_f1 = metric_sums["token_f1"] / n
                elapsed = time.time() - t0
                print(
                    f"  Progress: {i+1}/{len(data)} | "
                    f"EM={avg_em:.3f} F1={avg_f1:.3f} | "
                    f"{elapsed:.0f}s",
                    flush=True,
                )

        total = len(per_sample_results)
        if total == 0:
            all_results[benchmark] = {"error": "No valid samples evaluated"}
            continue

        # Aggregate metrics
        avg_metrics = {m: v / total for m, v in metric_sums.items()}

        # Per answer-type breakdown
        by_type = {}
        for r in per_sample_results:
            at = r["answer_type"]
            if at not in by_type:
                by_type[at] = {"count": 0, "metrics": Counter()}
            by_type[at]["count"] += 1
            for m, v in r["metrics"].items():
                by_type[at]["metrics"][m] += v

        for at, at_info in by_type.items():
            at_info["metrics"] = {
                m: v / at_info["count"] for m, v in at_info["metrics"].items()
            }

        elapsed = time.time() - t0
        all_results[benchmark] = {
            "benchmark": benchmark,
            "total": total,
            "metrics": avg_metrics,
            "by_answer_type": by_type,
            "elapsed_seconds": round(elapsed, 1),
            "per_sample": per_sample_results,
        }

        print(
            f"  [{model_name}] {benchmark}: "
            f"EM={avg_metrics['exact_match']:.4f} "
            f"F1={avg_metrics['token_f1']:.4f} "
            f"({total} samples, {elapsed:.0f}s)",
            flush=True,
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"vqa_tooluse_{ts}.json"

    # Build summary (without per-sample for the top-level view)
    summary_benchmarks = {}
    for key, r in all_results.items():
        if "error" in r:
            summary_benchmarks[key] = r
        else:
            summary_benchmarks[key] = {
                "total": r["total"],
                "metrics": r["metrics"],
                "by_answer_type": r["by_answer_type"],
                "elapsed_seconds": r.get("elapsed_seconds"),
            }

    report = {
        "model_name": model_name,
        "model_path": model_path,
        "format": "tool-use (training-aligned)",
        "system_prompt": SYSTEM_PROMPT,
        "tools": [t["function"]["name"] for t in OPENAI_TOOLS] if OPENAI_TOOLS else [],
        "timestamp": datetime.now().isoformat(),
        "category": "vqa_tooluse",
        "benchmarks": summary_benchmarks,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save per-sample details separately (can be large with many samples)
    detail_path = output_dir / f"vqa_tooluse_detail_{ts}.json"
    detail_report = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {
            k: v.get("per_sample", []) for k, v in all_results.items() if "per_sample" in v
        },
    }
    with open(detail_path, "w") as f:
        json.dump(detail_report, f, indent=2)

    # Print summary
    print(f"\n{'='*70}", flush=True)
    print(f"  VQA (Tool-Use Format) RESULTS: {model_name}", flush=True)
    print(f"  {'Benchmark':<20} {'EM':>10} {'F1':>10} {'Total':>8} {'Time':>8}", flush=True)
    print(f"  {'-'*56}", flush=True)
    for key, r in all_results.items():
        if "error" in r:
            print(f"  {key:<20} {'ERROR':>10} {r['error']}", flush=True)
        else:
            m = r["metrics"]
            print(
                f"  {key:<20} {m['exact_match']:>10.4f} {m['token_f1']:>10.4f} "
                f"{r['total']:>8} {r.get('elapsed_seconds', 0):>7.0f}s",
                flush=True,
            )

    # Print per-type breakdown
    for key, r in all_results.items():
        if "by_answer_type" in r and len(r["by_answer_type"]) > 1:
            print(f"\n  {key} by answer type:", flush=True)
            for at, at_info in r["by_answer_type"].items():
                at_m = at_info["metrics"]
                print(
                    f"    {at:<15} EM={at_m.get('exact_match', 0):.3f} "
                    f"F1={at_m.get('token_f1', 0):.3f} (n={at_info['count']})",
                    flush=True,
                )

    print(f"\n  Summary saved: {out_path}", flush=True)
    print(f"  Details saved: {detail_path}", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
