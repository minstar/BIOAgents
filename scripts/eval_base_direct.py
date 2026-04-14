#!/usr/bin/env python3
"""Evaluate base Qwen3.5-9B VLM with direct prompting (no tool-use format).

Covers all benchmarks: TextQA (MedQA, MedMCQA, MMLU), VQA, MedLFQA.
Uses simple direct answer prompts for fair comparison against RL-trained models.

Usage:
    # TextQA only
    CUDA_VISIBLE_DEVICES=5 python scripts/eval_base_direct.py \
        --model_path checkpoints/models/Qwen3.5-9B \
        --eval-type textqa --no-think \
        --output-dir results/benchmarks_base/qwen35_9b

    # VQA only
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_base_direct.py \
        --model_path checkpoints/models/Qwen3.5-9B \
        --eval-type vqa --no-think \
        --output-dir results/benchmarks_base/qwen35_9b

    # MedLFQA only
    CUDA_VISIBLE_DEVICES=6 python scripts/eval_base_direct.py \
        --model_path checkpoints/models/Qwen3.5-9B \
        --eval-type medlfqa --no-think \
        --output-dir results/benchmarks_base/qwen35_9b
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

# ── System prompts for direct evaluation ──
TEXTQA_SYSTEM = (
    "You are a medical AI assistant. Answer the following medical question "
    "by selecting the best option. Reply with ONLY the answer letter (A, B, C, or D)."
)

VQA_SYSTEM = (
    "You are a medical AI assistant specialized in analyzing medical images. "
    "Answer the question about the given medical image. Be concise and accurate."
)

MEDLFQA_SYSTEM = (
    "You are a medical AI assistant. Provide a detailed, accurate answer "
    "to the following medical question based on current medical knowledge."
)


def extract_mcq_answer(response: str) -> str:
    """Extract answer letter from direct response."""
    # Try <answer>X</answer>
    m = re.search(r'<answer>\s*([A-Ea-e])\s*</answer>', response)
    if m:
        return m.group(1).upper()

    # Strip thinking
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split('</think>')[-1].strip()

    # Try "answer": "X" (in case model outputs JSON anyway)
    m = re.search(r'"answer"\s*:\s*"([A-Ea-e])"', clean)
    if m:
        return m.group(1).upper()

    # First letter A-E
    for ch in clean:
        if ch in "ABCDE":
            return ch

    return ""


def text_answer_to_letter(question: str, answer_text: str) -> str:
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


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Simple ROUGE-L F1."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == pred_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]

    if lcs_len == 0:
        return 0.0
    prec = lcs_len / n
    rec = lcs_len / m
    return 2 * prec * rec / (prec + rec)


def eval_textqa(model, tokenizer, args):
    """Evaluate TextQA benchmarks (MedQA, MedMCQA, MMLU Clinical)."""
    import torch

    benchmark_files = {
        "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
        "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
        "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_test.jsonl",
    }

    all_results = {}
    for bm_key, bm_file in benchmark_files.items():
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

        correct, total = 0, 0
        t0 = time.time()

        for i, item in enumerate(data):
            instances = item.get("instances", {})
            question = instances.get("input", "") if isinstance(instances, dict) else ""
            answer = instances.get("output", "") if isinstance(instances, dict) else ""
            if not question:
                continue

            ref_letter = text_answer_to_letter(question, answer.strip())

            messages = [
                {"role": "system", "content": TEXTQA_SYSTEM},
                {"role": "user", "content": question},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if args.no_think:
                if text.rstrip().endswith("<think>"):
                    text = text.rstrip() + "\n</think>\n"
                elif "<think>" in text and "</think>" not in text:
                    text += "</think>\n"

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            max_tokens = 64 if args.no_think else 512
            t_gen = time.time()
            with torch.no_grad():
                eos_ids = [tokenizer.eos_token_id]
                for special in ["<|im_end|>"]:
                    tid = tokenizer.convert_tokens_to_ids(special)
                    if tid is not None and tid != tokenizer.unk_token_id:
                        eos_ids.append(tid)
                outputs = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids,
                )

            input_len = inputs["input_ids"].shape[-1]
            generated = outputs[0][input_len:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()

            pred = extract_mcq_answer(response)
            total += 1
            if pred and ref_letter and pred == ref_letter:
                correct += 1

            if total % 50 == 0:
                elapsed = time.time() - t0
                acc = correct / max(total, 1)
                print(f"    {bm_key}: {total}/{len(data)} acc={acc:.3f} {elapsed:.0f}s", flush=True)

            del outputs, generated, inputs
            torch.cuda.empty_cache()

        acc = correct / max(total, 1)
        all_results[bm_key] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  [{args.model_name}] {bm_key}: accuracy={acc:.4f} ({correct}/{total})", flush=True)

    return all_results


def eval_vqa(model, processor, tokenizer, args):
    """Evaluate VQA benchmarks."""
    import torch
    from bioagents.data_pipeline.vqa_loader import VQA_DATASET_REGISTRY

    VQA_BENCHMARKS = ["vqa_rad", "slake", "pathvqa", "pmc_vqa", "vqa_med_2021", "quilt_vqa"]

    all_results = {}
    for bm_name in VQA_BENCHMARKS:
        if bm_name not in VQA_DATASET_REGISTRY:
            print(f"  [SKIP] Unknown benchmark: {bm_name}", flush=True)
            continue

        info = VQA_DATASET_REGISTRY[bm_name]
        loader = info["loader"]
        try:
            if bm_name in ("vqa_med_2021", "quilt_vqa"):
                samples = loader(max_samples=args.max_samples if args.max_samples > 0 else None)
            else:
                samples = loader(max_samples=args.max_samples if args.max_samples > 0 else None, split="test")
        except Exception as e:
            print(f"  [ERROR] Failed to load {bm_name}: {e}", flush=True)
            continue

        if not samples:
            print(f"  [SKIP] No data for {bm_name}", flush=True)
            continue

        print(f"\n  [{args.model_name}] {bm_name}: {len(samples)} samples", flush=True)

        em_sum, f1_sum, total = 0, 0, 0
        t0 = time.time()

        for i, sample in enumerate(samples):
            question = sample.get("question", "")
            ref_answer = sample.get("answer", "")
            image = sample.get("image", None)

            if not question:
                continue

            if image is not None:
                messages = [
                    {"role": "system", "content": VQA_SYSTEM},
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ]},
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                if args.no_think:
                    if text.rstrip().endswith("<think>"):
                        text = text.rstrip() + "\n</think>\n"
                    elif "<think>" in text and "</think>" not in text:
                        text += "</think>\n"

                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt"
                )
            else:
                messages = [
                    {"role": "system", "content": VQA_SYSTEM},
                    {"role": "user", "content": question},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if args.no_think:
                    if text.rstrip().endswith("<think>"):
                        text = text.rstrip() + "\n</think>\n"
                    elif "<think>" in text and "</think>" not in text:
                        text += "</think>\n"

                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            max_tokens = 512 if args.no_think else 8192
            t_gen = time.time()
            print(f"  [GEN] #{i} starting (input_tokens={inputs['input_ids'].shape[-1]})...", flush=True)

            with torch.no_grad():
                eos_ids = [tokenizer.eos_token_id]
                for special in ["<|im_end|>"]:
                    tid = tokenizer.convert_tokens_to_ids(special)
                    if tid is not None and tid != tokenizer.unk_token_id:
                        eos_ids.append(tid)
                outputs = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids,
                )

            input_len = inputs["input_ids"].shape[-1]
            generated = outputs[0][input_len:]
            gen_len = len(generated)
            response = processor.decode(generated, skip_special_tokens=True).strip()
            print(f"  [GEN] #{i} done in {time.time()-t_gen:.1f}s ({gen_len} tokens)", flush=True)

            # Strip thinking
            clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            if '</think>' in clean:
                clean = clean.split('</think>')[-1].strip()

            # Compute EM and F1
            pred_lower = clean.lower().strip()
            ref_lower = str(ref_answer).lower().strip()

            em = 1.0 if pred_lower == ref_lower else 0.0

            pred_tokens = set(pred_lower.split())
            ref_tokens = set(ref_lower.split())
            if pred_tokens and ref_tokens:
                common = pred_tokens & ref_tokens
                prec = len(common) / len(pred_tokens) if pred_tokens else 0
                rec = len(common) / len(ref_tokens) if ref_tokens else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            else:
                f1 = 0.0

            em_sum += em
            f1_sum += f1
            total += 1

            if total % 50 == 0:
                elapsed = time.time() - t0
                print(f"  Progress: {total}/{len(samples)} | EM={em_sum/total:.3f} F1={f1_sum/total:.3f} | {elapsed:.0f}s", flush=True)

            del outputs, generated, inputs
            torch.cuda.empty_cache()

        em_avg = em_sum / max(total, 1)
        f1_avg = f1_sum / max(total, 1)
        all_results[bm_name] = {"em": em_avg, "f1": f1_avg, "total": total}
        print(f"  [{args.model_name}] {bm_name}: EM={em_avg:.4f} F1={f1_avg:.4f} ({total} samples, {time.time()-t0:.0f}s)", flush=True)

    return all_results


def eval_medlfqa(model, tokenizer, args):
    """Evaluate MedLFQA benchmarks."""
    import torch

    benchmark_files = {
        "kqa_golden": {
            "path": "evaluations/OLAPH/MedLFQA/kqa_golden_test_MedLFQA.jsonl",
            "question_key": "Question",
            "answer_key": "Free_form_answer",
        },
        "liveqa": {
            "path": "evaluations/OLAPH/MedLFQA/live_qa_test_MedLFQA.jsonl",
            "question_key": "Question",
            "answer_key": "Free_form_answer",
        },
        "medicationqa": {
            "path": "evaluations/OLAPH/MedLFQA/medication_qa_test_MedLFQA.jsonl",
            "question_key": "Question",
            "answer_key": "Free_form_answer",
        },
        "healthsearchqa": {
            "path": "evaluations/OLAPH/MedLFQA/healthsearch_qa_test_MedLFQA.jsonl",
            "question_key": "Question",
            "answer_key": "Free_form_answer",
        },
        "kqa_silver": {
            "path": "evaluations/OLAPH/MedLFQA/kqa_silver_wogold_test_MedLFQA.jsonl",
            "question_key": "Question",
            "answer_key": "Free_form_answer",
        },
    }

    all_results = {}
    for bm_key, bm_info in benchmark_files.items():
        full_path = PROJECT_ROOT / bm_info["path"]
        q_key = bm_info["question_key"]
        a_key = bm_info["answer_key"]
        if not full_path.exists():
            print(f"  [SKIP] {bm_key}: {bm_info['path']} not found", flush=True)
            continue

        data = []
        with open(full_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        if args.max_samples > 0:
            data = data[:args.max_samples]

        print(f"\n  [{args.model_name}] Evaluating {bm_key}...", flush=True)
        print(f"  Loaded {len(data)} examples", flush=True)

        rouge_sum, total = 0.0, 0
        t0 = time.time()

        for i, item in enumerate(data):
            question = item.get(q_key, "") or item.get("question", "") or item.get("instances", {}).get("input", "")
            reference = item.get(a_key, "") or item.get("answer", "") or item.get("instances", {}).get("output", "")
            if not question or not reference:
                continue

            messages = [
                {"role": "system", "content": MEDLFQA_SYSTEM},
                {"role": "user", "content": question},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if args.no_think:
                if text.rstrip().endswith("<think>"):
                    text = text.rstrip() + "\n</think>\n"
                elif "<think>" in text and "</think>" not in text:
                    text += "</think>\n"

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            max_tokens = 1024 if args.no_think else 2048
            t_gen = time.time()
            print(f"    [{bm_key}] #{i} generating (input={inputs['input_ids'].shape[-1]} tokens)...", flush=True)
            with torch.no_grad():
                eos_ids = [tokenizer.eos_token_id]
                for special in ["<|im_end|>"]:
                    tid = tokenizer.convert_tokens_to_ids(special)
                    if tid is not None and tid != tokenizer.unk_token_id:
                        eos_ids.append(tid)
                outputs = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids,
                )

            input_len = inputs["input_ids"].shape[-1]
            generated = outputs[0][input_len:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()

            # Strip thinking
            clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            if '</think>' in clean:
                clean = clean.split('</think>')[-1].strip()

            rouge = compute_rouge_l(clean, reference)
            rouge_sum += rouge
            total += 1
            print(f"    [{bm_key}] #{i} done in {time.time()-t_gen:.1f}s ROUGE-L={rouge:.3f}", flush=True)

            if total % 50 == 0 or total == len(data):
                elapsed = time.time() - t0
                avg_rouge = rouge_sum / max(total, 1)
                print(f"  [{args.model_name}] {bm_key}: {total}/{len(data)} ROUGE-L={avg_rouge:.3f} {elapsed:.0f}s", flush=True)

            del outputs, generated, inputs
            torch.cuda.empty_cache()

        avg_rouge = rouge_sum / max(total, 1)
        all_results[bm_key] = {"rouge_l": avg_rouge, "total": total}
        print(f"  [{args.model_name}] {bm_key}: ROUGE-L={avg_rouge:.4f} ({total} examples, {time.time()-t0:.0f}s)", flush=True)

    return all_results


def main():
    import argparse
    import torch
    from transformers import AutoConfig, AutoTokenizer

    parser = argparse.ArgumentParser(description="Base model direct prompting evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/benchmarks_base")
    parser.add_argument("--eval-type", type=str, required=True, choices=["textqa", "vqa", "medlfqa", "all"])
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-think", action="store_true", help="Disable thinking")
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["PYTHONUNBUFFERED"] = "1"

    model_path = args.model_path
    args.model_name = Path(model_path).name

    print(f"\n{'#'*70}", flush=True)
    print(f"  Base Model Direct Prompting Evaluation", flush=True)
    print(f"  Model: {args.model_name}", flush=True)
    print(f"  Eval: {args.eval_type}", flush=True)
    print(f"  Format: Direct prompting (no tools)", flush=True)
    print(f"{'#'*70}\n", flush=True)

    # Load model
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(model_config, "model_type", "")
    is_qwen3_5 = model_type == "qwen3_5"

    load_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": 0})
    load_kwargs["attn_implementation"] = "eager"

    processor = None
    if is_qwen3_5:
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For VQA, we need the processor
    if args.eval_type in ("vqa", "all"):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    all_results = {}
    eval_types = ["textqa", "vqa", "medlfqa"] if args.eval_type == "all" else [args.eval_type]

    for et in eval_types:
        if et == "textqa":
            results = eval_textqa(model, tokenizer, args)
            all_results["textqa"] = results
        elif et == "vqa":
            results = eval_vqa(model, processor or tokenizer, tokenizer, args)
            all_results["vqa"] = results
        elif et == "medlfqa":
            results = eval_medlfqa(model, tokenizer, args)
            all_results["medlfqa"] = results

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"base_direct_{args.eval_type}_{ts}.json"
    report = {
        "model_name": args.model_name,
        "model_path": model_path,
        "format": "direct prompting (no tools)",
        "no_think": args.no_think,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"  Base Model Direct Evaluation RESULTS: {args.model_name}", flush=True)
    for category, results in all_results.items():
        print(f"\n  [{category}]", flush=True)
        for key, r in results.items():
            if "accuracy" in r:
                print(f"    {key:<25} acc={r['accuracy']:.4f} ({r['correct']}/{r['total']})", flush=True)
            elif "em" in r:
                print(f"    {key:<25} EM={r['em']:.4f} F1={r['f1']:.4f} ({r['total']})", flush=True)
            elif "rouge_l" in r:
                print(f"    {key:<25} ROUGE-L={r['rouge_l']:.4f} ({r['total']})", flush=True)
    print(f"\n  Saved: {out_path}", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
