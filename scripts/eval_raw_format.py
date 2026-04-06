#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick eval using raw question format matching run_full_benchmark_suite.py
but with CORRECT answer extraction (parse_item, not heuristic first-capital-letter).

Usage:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/eval_raw_format.py <model_path> <output_name>
"""
import json, os, re, sys, time, torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sft_warmup_lingshu7b_v2_merged/merged"
OUTPUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "test"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

BENCHMARKS = {
    "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
}

print(f"Loading model from {MODEL_PATH}...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, attn_implementation="sdpa"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# NOTE: Using default padding (right) to match original run_full_benchmark_suite.py
model.eval()
print("Model loaded!")


def get_correct_letter(item):
    """Correct answer extraction using text matching."""
    raw_input = item.get("instances", {}).get("input", "")
    raw_output = item.get("instances", {}).get("output", "")
    options = dict(re.findall(r'Option\s+([A-E]):\s*(.*?)(?=Option\s+[A-E]:|$)', raw_input, re.DOTALL))
    if raw_output and options:
        raw_out = raw_output.strip()
        for letter, text in options.items():
            if text.strip().lower() == raw_out.lower() or raw_out.lower() in text.strip().lower():
                return letter
        for ch in raw_out:
            if ch in "ABCDE":
                return ch
    return "B"


for bm_key, bm_path in BENCHMARKS.items():
    data = []
    with open(bm_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    correct, total = 0, len(data)
    t0 = time.time()
    print(f"\n{bm_key}: {total} items (raw format, right-padding)")

    BATCH = 8
    for i in range(0, len(data), BATCH):
        batch = data[i:i+BATCH]
        prompts, answers = [], []
        for item in batch:
            question = item.get("instances", {}).get("input", "")
            if not question:
                continue
            ans = get_correct_letter(item)
            msgs = [
                {"role": "system", "content": "Answer the medical question by selecting the best option. Reply with ONLY the letter (A, B, C, or D)."},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            answers.append(ans)
        if not prompts:
            continue
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=4096, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for j in range(len(prompts)):
            # Right-padding: non-pad tokens are at the start
            input_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            gen = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True).strip()
            pred = ""
            for ch in gen:
                if ch in "ABCD":
                    pred = ch
                    break
            if pred == answers[j]:
                correct += 1
        if (i // BATCH) % 20 == 0:
            print(f"  {min(i+BATCH,total)}/{total} acc={correct/max(min(i+BATCH,total),1)*100:.1f}%", flush=True)

    acc = correct / total if total > 0 else 0
    print(f"  RESULT: {acc*100:.1f}% ({correct}/{total}) [{time.time()-t0:.1f}s]")
    print(f"  [{OUTPUT_NAME}]")
