#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate pre-SFT baselines with FIXED methodology.

Evaluates base models (no SFT, no RL) to establish true baselines
for comparison with SFT and SFT+RL checkpoints.

Uses left-padding + inputs["input_ids"].shape[-1] for correct answer extraction.

Usage:
    CUDA_VISIBLE_DEVICES=X .venv/bin/python scripts/eval_baselines_fixed.py [model_name]

    model_name: lingshu | qwen | step3  (default: lingshu)
"""
import json, os, re, sys, time, torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

MODEL_MAP = {
    "lingshu": ("checkpoints/models/Lingshu-7B", "lingshu7b_presft"),
    "qwen": ("checkpoints/models/Qwen2.5-VL-7B-Instruct", "qwen25vl7b_presft"),
    "step3": ("checkpoints/models/Step3-VL-10B", "step3vl10b_presft"),
    "sft": ("checkpoints/sft_warmup_lingshu7b_v2_merged/merged", "lingshu7b_sft_fixed"),
    "h5_step150": ("checkpoints/grpo_direct_medqa_lingshu7b/checkpoint-150-merged", "h5_direct_medqa_step150"),
    "h5_step200": ("checkpoints/grpo_direct_medqa_lingshu7b/checkpoint-200-merged", "h5_direct_medqa_step200"),
    "h5_step250": ("checkpoints/grpo_direct_medqa_lingshu7b/checkpoint-250-merged", "h5_direct_medqa_step250"),
    "h5_step300": ("checkpoints/grpo_direct_medqa_lingshu7b/checkpoint-300-merged", "h5_direct_medqa_step300"),
    "drgrpo_step700": ("checkpoints/drgrpo_lingshu7b/checkpoint-700-merged", "drgrpo_step700"),
    "drgrpo_step800": ("checkpoints/drgrpo_lingshu7b/checkpoint-800-merged", "drgrpo_step800"),
    "drgrpo_step950": ("checkpoints/drgrpo_lingshu7b/checkpoint-950-merged", "drgrpo_step950"),
    "h12_step300": ("checkpoints/grpo_lr1e6_direct_medqa_lingshu7b/checkpoint-300-merged", "h12_lr1e6_direct_step300"),
    "h12_step500": ("checkpoints/grpo_lr1e6_direct_medqa_lingshu7b/checkpoint-500-merged", "h12_lr1e6_direct_step500"),
}

model_key = sys.argv[1] if len(sys.argv) > 1 else "lingshu"
MODEL_PATH, OUTPUT_NAME = MODEL_MAP.get(model_key, MODEL_MAP["lingshu"])

RESULTS_DIR = "results/fixed_baselines"
os.makedirs(RESULTS_DIR, exist_ok=True)

TEXT_BENCHMARKS = {
    "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
    "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
    "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_clinical_knowledge_test.jsonl",
}


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


print(f"Loading model: {MODEL_PATH}")
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
model_type = getattr(config, "model_type", "")

if model_type in ("qwen2_5_vl", "qwen2_vl"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa"
    )
else:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()
print(f"Model loaded! ({model_type})")

results = {}
for bm_key, bm_path in TEXT_BENCHMARKS.items():
    full_path = Path(bm_path)
    if not full_path.exists():
        print(f"SKIP {bm_key}: {full_path}")
        continue
    data = []
    with open(full_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    correct_count, total = 0, len(data)
    t0 = time.time()
    print(f"\n{bm_key}: {total} questions")
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
            except Exception:
                text = f"<|im_start|>system\nYou are a medical expert.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append(text)
            answers.append(ans)
        if not prompts:
            continue
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=4096, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for j in range(len(prompts)):
            input_len = inputs["input_ids"].shape[-1]  # CORRECT for left-padding
            gen = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True).strip()
            pred = re.search(r'[A-D]', gen)
            pred = pred.group(0) if pred else ""
            if pred == answers[j]:
                correct_count += 1
        if (i // BATCH) % 20 == 0:
            print(f"  {min(i+BATCH,total)}/{total} acc={correct_count/max(min(i+BATCH,total),1)*100:.1f}%", flush=True)
    acc = correct_count / total if total > 0 else 0
    elapsed = time.time() - t0
    print(f"  {bm_key}: {acc*100:.1f}% ({correct_count}/{total}) [{elapsed:.1f}s]")
    results[bm_key] = {"accuracy": acc, "correct": correct_count, "total": total}

out = {
    "model_name": OUTPUT_NAME,
    "model_path": MODEL_PATH,
    "eval_version": "fixed_v3_baselines",
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "category": "textqa_fixed",
    "benchmarks": results,
}
outpath = os.path.join(RESULTS_DIR, f"{OUTPUT_NAME}_{out['timestamp']}.json")
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {outpath}")
print(f"\nSUMMARY ({OUTPUT_NAME}):")
for k, v in results.items():
    print(f"  {k}: {v['accuracy']*100:.1f}% ({v['correct']}/{v['total']})")
