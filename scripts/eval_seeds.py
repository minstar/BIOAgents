#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate 3-seed GRPO checkpoints at key steps (100, 250, 300)."""
import json, os, re, sys, time, torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

BASE_MODEL = "checkpoints/sft_warmup_lingshu7b_v2_merged/merged"
RESULTS_DIR = "results/algorithm_comparison"

# Key steps to evaluate for seeds
SEED_CHECKPOINTS = []
for seed_name, seed_dir in [("grpo_seed123", "grpo_seed123_lingshu7b"),
                              ("grpo_seed456", "grpo_seed456_lingshu7b")]:
    for step in [100, 250, 300]:
        ckpt_path = f"checkpoints/{seed_dir}/checkpoint-{step}"
        result_name = f"{seed_name}_step{step}_textqa"
        if os.path.exists(ckpt_path):
            out_dir = os.path.join(RESULTS_DIR, result_name)
            # Skip if already evaluated
            if os.path.exists(out_dir):
                jsons = [f for f in os.listdir(out_dir) if f.endswith('.json')]
                if jsons:
                    with open(os.path.join(out_dir, jsons[0])) as f:
                        d = json.load(f)
                    if d.get('benchmarks', {}).get('medqa', {}).get('accuracy', 0) > 0:
                        print(f"SKIP {result_name}: already done")
                        continue
            SEED_CHECKPOINTS.append((ckpt_path, result_name))

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


def evaluate_model(model, tokenizer, device):
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
            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"].shape[-1]  # total padded input length (correct for left-padding)
                gen = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
                pred = re.search(r'[A-D]', gen)
                pred = pred.group(0) if pred else ""
                if pred == answers[j]:
                    correct_count += 1
        acc = correct_count / total if total > 0 else 0
        elapsed = time.time() - t0
        print(f"  {bm_key}: {acc*100:.1f}% ({correct_count}/{total}) [{elapsed:.1f}s]")
        results[bm_key] = {"accuracy": acc, "correct": correct_count, "total": total}
    return results


if not SEED_CHECKPOINTS:
    print("Nothing to evaluate!")
    sys.exit(0)

print(f"Evaluating {len(SEED_CHECKPOINTS)} seed checkpoints")
for ap, rn in SEED_CHECKPOINTS:
    print(f"  {rn}")

print(f"\nLoading base model from {BASE_MODEL}...")
config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
model_type = getattr(config, "model_type", "")
if model_type in ("qwen2_5_vl", "qwen2_vl"):
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

for idx, (adapter_path, result_name) in enumerate(SEED_CHECKPOINTS):
    print(f"\n{'='*60}")
    print(f"Evaluating: {result_name}")
    print(f"{'='*60}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    device = next(model.parameters()).device
    results = evaluate_model(model, tokenizer, device)
    out_dir = os.path.join(RESULTS_DIR, result_name)
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "model_name": result_name,
        "adapter_path": adapter_path,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "category": "textqa_custom",
        "benchmarks": results,
    }
    outpath = os.path.join(out_dir, f"textqa_{out['timestamp']}.json")
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {outpath}")
    if idx < len(SEED_CHECKPOINTS) - 1:
        del model
        torch.cuda.empty_cache()
        if model_type in ("qwen2_5_vl", "qwen2_vl"):
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )

print("\nALL SEED EVALUATIONS COMPLETE")
