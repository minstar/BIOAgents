#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge LoRA adapter to disk, then evaluate with fixed script.

Two-phase approach to avoid PeftModel memory issues with device_map.
Phase 1: Load base + adapter on single GPU, merge, save to temp dir
Phase 2: Load merged model and evaluate

Usage:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/merge_and_eval_fixed.py
"""
import json, os, re, sys, time, torch, shutil
from pathlib import Path
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

BASE_MODEL = "checkpoints/sft_warmup_lingshu7b_v2_merged/merged"
RESULTS_DIR = "results/algorithm_comparison"
TEMP_MERGED = "/tmp/temp_merged_model"

TEXT_BENCHMARKS = {
    "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
    "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
    "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_clinical_knowledge_test.jsonl",
}

# Build checkpoint list
CHECKPOINTS = []
for seed_name, seed_dir in [
    ("grpo_seed123", "grpo_seed123_lingshu7b"),
    ("grpo_seed456", "grpo_seed456_lingshu7b"),
]:
    for step in [50, 100, 150, 200, 250, 300, 350]:
        ckpt = f"checkpoints/{seed_dir}/checkpoint-{step}"
        rname = f"{seed_name}_step{step}_fixed"
        if os.path.exists(ckpt):
            out_dir = os.path.join(RESULTS_DIR, rname)
            if os.path.exists(out_dir):
                jsons = [f for f in os.listdir(out_dir) if f.endswith('.json')]
                if jsons:
                    print(f"SKIP {rname}: already done")
                    continue
            CHECKPOINTS.append((ckpt, rname))

# Also add baseline late checkpoints
for step in [850, 900, 950]:
    ckpt = f"checkpoints/grpo_baseline_lingshu7b/checkpoint-{step}"
    rname = f"grpo_baseline_step{step}_fixed"
    if os.path.exists(ckpt):
        out_dir = os.path.join(RESULTS_DIR, rname)
        if os.path.exists(out_dir):
            jsons = [f for f in os.listdir(out_dir) if f.endswith('.json')]
            if jsons:
                print(f"SKIP {rname}: already done")
                continue
        CHECKPOINTS.append((ckpt, rname))


def parse_item(item):
    instances = item.get("instances", {})
    raw_input = instances.get("input", "")
    raw_output = instances.get("output", "")
    if not raw_input:
        q = item.get("question", "")
        opts = item.get("options", item.get("choices", {}))
        if isinstance(opts, list):
            opts = {chr(65+i): o for i, o in enumerate(opts)}
        ans = item.get("answer_idx", item.get("answer", ""))
        if isinstance(ans, int):
            ans = chr(65 + ans)
        return q, opts, str(ans).strip().upper()
    rest = raw_input.split("QUESTION:", 1)[1].strip() if "QUESTION:" in raw_input else raw_input.strip()
    option_matches = re.findall(r'Option\s+([A-E]):\s*(.*?)(?=Option\s+[A-E]:|$)', rest, re.DOTALL)
    question, options = "", {}
    if option_matches:
        first_pos = rest.find("Option A:")
        if first_pos == -1:
            first_pos = rest.find("Option B:")
        if first_pos > 0:
            question = rest[:first_pos].strip()
        for letter, text in option_matches:
            options[letter.strip()] = text.strip()
    else:
        question = rest
    correct = ""
    if raw_output:
        raw_out = raw_output.strip()
        for letter, text in options.items():
            if text.strip().lower() == raw_out.lower() or raw_out.lower() in text.lower():
                correct = letter
                break
        if not correct and raw_out and raw_out[0].upper() in options:
            correct = raw_out[0].upper()
        if not correct:
            correct = "B"
    return question, options, correct


def evaluate_model(model, tokenizer):
    device = next(model.parameters()).device
    results = {}
    for bm_key, bm_path in TEXT_BENCHMARKS.items():
        full_path = Path(bm_path)
        if not full_path.exists():
            continue
        data = []
        with open(full_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        correct_count, total = 0, len(data)
        t0 = time.time()
        BATCH = 8
        for i in range(0, len(data), BATCH):
            batch = data[i:i+BATCH]
            prompts, answers = [], []
            for item in batch:
                q, opts, ans = parse_item(item)
                if not q or not opts:
                    continue
                prompt = f"Question: {q}\n\nOptions:\n"
                for letter, text in sorted(opts.items()):
                    prompt += f"  {letter}) {text}\n"
                prompt += "\nAnswer with only the letter (A, B, C, or D):"
                msgs = [
                    {"role": "system", "content": "You are a medical expert. Answer with only the correct option letter."},
                    {"role": "user", "content": prompt},
                ]
                try:
                    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                except:
                    text = f"<|im_start|>system\nYou are a medical expert.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                prompts.append(text)
                answers.append(ans)
            if not prompts:
                continue
            inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=4096, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            for j in range(len(prompts)):
                input_len = inputs["input_ids"].shape[-1]
                gen = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True).strip()
                pred = re.search(r'[A-D]', gen)
                pred = pred.group(0) if pred else ""
                if pred == answers[j]:
                    correct_count += 1
        acc = correct_count / total if total > 0 else 0
        elapsed = time.time() - t0
        print(f"  {bm_key}: {acc*100:.1f}% ({correct_count}/{total}) [{elapsed:.1f}s]", flush=True)
        results[bm_key] = {"accuracy": acc, "correct": correct_count, "total": total}
    return results


if not CHECKPOINTS:
    print("Nothing to evaluate!")
    sys.exit(0)

print(f"Evaluating {len(CHECKPOINTS)} checkpoints (merge-to-disk + eval)")
for ap, rn in CHECKPOINTS:
    print(f"  {rn}")

for idx, (adapter_path, result_name) in enumerate(CHECKPOINTS):
    print(f"\n{'='*60}")
    print(f"[{idx+1}/{len(CHECKPOINTS)}] {result_name}")
    print(f"{'='*60}", flush=True)

    # Phase 1: Merge to disk
    print("Phase 1: Loading base + merging adapter...", flush=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    # Save merged model to temp dir
    if os.path.exists(TEMP_MERGED):
        shutil.rmtree(TEMP_MERGED)
    model.save_pretrained(TEMP_MERGED)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(TEMP_MERGED)
    # Copy the ORIGINAL config to avoid rope_scaling/rope_parameters mismatch
    shutil.copy2(os.path.join(BASE_MODEL, "config.json"), os.path.join(TEMP_MERGED, "config.json"))

    del model, base_model
    torch.cuda.empty_cache()
    print("Merged to disk.", flush=True)

    # Phase 2: Load merged model and evaluate
    print("Phase 2: Loading merged model for eval...", flush=True)
    merged_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        TEMP_MERGED, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, attn_implementation="sdpa"
    )
    merged_tokenizer = AutoTokenizer.from_pretrained(TEMP_MERGED, trust_remote_code=True)
    merged_tokenizer.padding_side = "left"
    if merged_tokenizer.pad_token is None:
        merged_tokenizer.pad_token = merged_tokenizer.eos_token
    merged_model.eval()

    results = evaluate_model(merged_model, merged_tokenizer)

    out_dir = os.path.join(RESULTS_DIR, result_name)
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "model_name": result_name,
        "adapter_path": adapter_path,
        "eval_version": "fixed_v2_merge_to_disk",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "category": "textqa_custom",
        "benchmarks": results,
    }
    outpath = os.path.join(out_dir, f"textqa_{out['timestamp']}.json")
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {outpath}", flush=True)

    del merged_model
    torch.cuda.empty_cache()

# Cleanup
if os.path.exists(TEMP_MERGED):
    shutil.rmtree(TEMP_MERGED)

print("\nALL FIXED EVALUATIONS COMPLETE")
