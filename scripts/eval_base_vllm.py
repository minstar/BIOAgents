#!/usr/bin/env python3
"""Base model evaluation using vLLM offline batch inference.

Fast single-turn evaluation for base Qwen3.5-9B without tool-use.
Uses vLLM for batched generation — 50-100x faster than sequential HF generate.

Usage:
    # TextQA (MedQA + MedMCQA + MMLU subtypes)
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_base_vllm.py \
        --model_path checkpoints/models/Qwen3.5-9B \
        --eval-type textqa \
        --output-dir results/benchmarks_base/qwen35_9b

    # VQA (text-only, no images — extracts question text)
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_base_vllm.py \
        --model_path checkpoints/models/Qwen3.5-9B \
        --eval-type vqa \
        --output-dir results/benchmarks_base/qwen35_9b

    # MedLFQA (long-form)
    CUDA_VISIBLE_DEVICES=5 python scripts/eval_base_vllm.py \
        --model_path checkpoints/models/Qwen3.5-9B \
        --eval-type medlfqa \
        --output-dir results/benchmarks_base/qwen35_9b
"""

import argparse
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

from rouge_score import rouge_scorer as _rouge_module
_rouge_scorer = _rouge_module.RougeScorer(["rougeL"], use_stemmer=True)

# ── System prompts ──
TEXTQA_SYSTEM = (
    "You are a medical AI assistant. Answer the following medical question "
    "by selecting the best option. Reply with ONLY the answer letter (A, B, C, or D)."
)

MEDLFQA_SYSTEM = (
    "You are a medical AI assistant. Provide a detailed, accurate answer "
    "to the following medical question based on current medical knowledge."
)

VQA_SYSTEM = (
    "You are a medical AI assistant. Answer the medical question concisely and accurately."
)

# ── Benchmark files ──
TEXTQA_BENCHMARKS = {
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

VQA_BENCHMARKS = {
    "vqa_rad": "evaluations/vqa/vqa_rad/test.json",
    "pathvqa": "evaluations/vqa/pathvqa/test.json",
    "slake": "evaluations/vqa/slake/test.json",
}

MEDLFQA_BENCHMARKS = {
    "kqa_golden": "evaluations/OLAPH/MedLFQA/kqa_golden_test_MedLFQA.jsonl",
    "live_qa": "evaluations/OLAPH/MedLFQA/live_qa_test_MedLFQA.jsonl",
    "medication_qa": "evaluations/OLAPH/MedLFQA/medication_qa_test_MedLFQA.jsonl",
    "healthsearch_qa": "evaluations/OLAPH/MedLFQA/healthsearch_qa_test_MedLFQA.jsonl",
    "kqa_silver": "evaluations/OLAPH/MedLFQA/kqa_silver_test_MedLFQA.jsonl",
}


def extract_mcq_answer(response: str) -> str:
    """Extract answer letter from response."""
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split('</think>')[-1].strip()

    m = re.search(r'"answer"\s*:\s*"([A-Ea-e])"', clean)
    if m:
        return m.group(1).upper()

    for ch in clean:
        if ch in "ABCDE":
            return ch
    return ""


def get_correct_letter(question: str, answer_text: str) -> str:
    """Map text answer to letter by matching against options."""
    if len(answer_text) <= 2 and answer_text.upper() in "ABCDE":
        return answer_text.upper()

    answer_lower = answer_text.lower().strip()
    options = dict(re.findall(r'Option\s+([A-E]):\s*(.*?)(?=Option\s+[A-E]:|$)', question, re.DOTALL))

    for letter, text in options.items():
        if text.strip().lower() == answer_lower:
            return letter

    for letter, text in options.items():
        if answer_lower in text.strip().lower() or text.strip().lower() in answer_lower:
            return letter

    for ch in answer_text:
        if ch in "ABCDE":
            return ch
    return "B"


def load_textqa(name: str) -> list[dict]:
    """Load TextQA benchmark."""
    filepath = PROJECT_ROOT / TEXTQA_BENCHMARKS[name]
    if not filepath.exists():
        print(f"  [SKIP] {name}: not found")
        return []

    tasks = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            instances = item.get("instances", {})
            question = instances.get("input", "")
            answer = instances.get("output", "").strip()
            if question:
                tasks.append({"question": question, "answer": answer, "type": "mcq"})
    return tasks


def load_vqa(name: str) -> list[dict]:
    """Load VQA benchmark (text-only, no images for base model)."""
    filepath = PROJECT_ROOT / VQA_BENCHMARKS[name]
    if not filepath.exists():
        print(f"  [SKIP] {name}: not found")
        return []

    with open(filepath) as f:
        items = json.load(f) if filepath.suffix == ".json" else [json.loads(l) for l in f if l.strip()]

    tasks = []
    for item in items:
        question = item.get("question", item.get("input", ""))
        answer = str(item.get("answer", item.get("output", ""))).strip()
        if question:
            tasks.append({"question": question, "answer": answer, "type": "vqa"})
    return tasks


def load_medlfqa(name: str) -> list[dict]:
    """Load MedLFQA benchmark."""
    filepath = PROJECT_ROOT / MEDLFQA_BENCHMARKS[name]
    if not filepath.exists():
        print(f"  [SKIP] {name}: not found")
        return []

    tasks = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            question = item.get("Question", "")
            answer = item.get("Free_form_answer", "").strip()
            must_have = item.get("Must_have", [])
            nice_to_have = item.get("Nice_to_have", [])
            if question:
                tasks.append({
                    "question": question, "answer": answer, "type": "lfqa",
                    "must_have": must_have, "nice_to_have": nice_to_have,
                })
    return tasks


def run_vllm_batch(model_path: str, benchmarks: dict, loader_fn, eval_type: str,
                   system_prompt: str, output_dir: Path, max_samples: int = 0):
    """Run vLLM offline batch inference for a set of benchmarks."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"  Loading model: {model_path}")
    print(f"  Eval type: {eval_type}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    max_tokens = 64 if eval_type in ("textqa",) else (512 if eval_type == "vqa" else 1024)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    all_summary = {}

    for bm_name, bm_path in benchmarks.items():
        tasks = loader_fn(bm_name)
        if not tasks:
            continue
        if max_samples > 0:
            tasks = tasks[:max_samples]

        print(f"\n  [{bm_name}] {len(tasks)} samples", flush=True)

        # Build prompts
        prompts = []
        for task in tasks:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task["question"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Disable thinking for base model
            if text.rstrip().endswith("<think>"):
                text = text.rstrip() + "\n</think>\n"
            elif "<think>" in text and "</think>" not in text:
                text += "</think>\n"
            prompts.append(text)

        # Batch generate
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - t0

        # Score
        results = []
        correct = 0
        rouge_l_sum = 0.0

        for task, output in zip(tasks, outputs):
            response = output.outputs[0].text.strip()

            if eval_type == "textqa":
                pred = extract_mcq_answer(response)
                ref = get_correct_letter(task["question"], task["answer"])
                is_correct = pred == ref and pred != ""
                result = {"pred": pred, "ref": ref, "correct": is_correct, "response": response[:200]}
            elif eval_type == "vqa":
                pred = response.strip()
                gold = task["answer"]
                is_correct = gold.lower() in pred.lower() or pred.lower() in gold.lower()
                result = {"pred": pred[:200], "gold": gold, "correct": is_correct}
            else:  # medlfqa
                gold = task["answer"]
                rl = _rouge_scorer.score(gold, response)["rougeL"].fmeasure
                rouge_l_sum += rl
                is_correct = rl >= 0.3
                result = {"response": response[:500], "gold": gold[:200], "rouge_l": round(rl, 4), "correct": is_correct}

            if is_correct:
                correct += 1
            results.append(result)

        total = len(results)
        acc = correct / max(total, 1)

        summary = {
            "benchmark": bm_name,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "time_seconds": elapsed,
            "throughput": total / max(elapsed, 0.01),
            "timestamp": datetime.now().isoformat(),
        }
        if eval_type == "medlfqa":
            summary["avg_rouge_l"] = rouge_l_sum / max(total, 1)

        all_summary[bm_name] = summary

        if eval_type == "medlfqa":
            print(f"  [{bm_name}] rouge_l={summary['avg_rouge_l']:.3f} acc@0.3={acc:.3f} "
                  f"({correct}/{total}) {elapsed:.0f}s ({summary['throughput']:.1f}/s)", flush=True)
        else:
            print(f"  [{bm_name}] accuracy={acc:.3f} ({correct}/{total}) "
                  f"{elapsed:.0f}s ({summary['throughput']:.1f}/s)", flush=True)

        # Save per-benchmark results
        out_path = output_dir / f"base_{bm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w") as f:
            json.dump({**summary, "results": results}, f, indent=2, ensure_ascii=False)

    return all_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/benchmarks_base/qwen35_9b")
    parser.add_argument("--eval-type", type=str, required=True, choices=["textqa", "vqa", "medlfqa", "all"])
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_type in ("textqa", "all"):
        run_vllm_batch(args.model_path, TEXTQA_BENCHMARKS, load_textqa,
                       "textqa", TEXTQA_SYSTEM, output_dir, args.max_samples)

    if args.eval_type in ("vqa", "all"):
        run_vllm_batch(args.model_path, VQA_BENCHMARKS, load_vqa,
                       "vqa", VQA_SYSTEM, output_dir, args.max_samples)

    if args.eval_type in ("medlfqa", "all"):
        run_vllm_batch(args.model_path, MEDLFQA_BENCHMARKS, load_medlfqa,
                       "medlfqa", MEDLFQA_SYSTEM, output_dir, args.max_samples)


if __name__ == "__main__":
    main()
