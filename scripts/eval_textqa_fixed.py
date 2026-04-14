#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TextQA evaluation matching the run_full_benchmark_suite.py approach.

Key differences from eval_textqa_custom.py:
- Passes raw question text from instances.input (not reformatted)
- Uses parse_item for correct answer-to-letter mapping
- Matches the system prompt from run_full_benchmark_suite.py
"""
import json, os, re, sys, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sft_warmup_lingshu7b_v2_merged/merged"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "results/test_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

TEXT_BENCHMARKS = {
    "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
    "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
    "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_clinical_knowledge_test.jsonl",
}

print(f"Loading model from {MODEL_PATH}...")
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
model_type = getattr(config, "model_type", "")
load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, attn_implementation="sdpa")

if model_type in ("qwen2_5_vl", "qwen2_vl"):
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, **load_kwargs)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Model loaded!")


def get_correct_letter(item):
    """Extract correct answer letter from item."""
    instances = item.get("instances", {})
    raw_input = instances.get("input", "")
    raw_output = instances.get("output", "")

    if not raw_input:
        ans = item.get("answer_idx", item.get("answer", ""))
        if isinstance(ans, int):
            return chr(65 + ans)
        return str(ans).strip().upper()[:1]

    # Parse options from raw input
    option_matches = re.findall(r'Option\s+([A-E]):\s*(.*?)(?=Option\s+[A-E]:|$)', raw_input, re.DOTALL)
    options = {letter.strip(): text.strip() for letter, text in option_matches}

    if raw_output and options:
        raw_out = raw_output.strip()
        # Try exact match first
        for letter, text in options.items():
            if text.strip().lower() == raw_out.lower():
                return letter
        # Try substring match
        for letter, text in options.items():
            if raw_out.lower() in text.lower() or text.lower() in raw_out.lower():
                return letter
        # Try first capital A-D from output text
        for ch in raw_out:
            if ch in "ABCDE":
                return ch
    return "B"  # fallback


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
    print(f"\n{bm_key}: {len(data)} questions")
    BATCH = 8
    for i in range(0, len(data), BATCH):
        batch = data[i:i+BATCH]
        prompts, answers = [], []
        for item in batch:
            instances = item.get("instances", {})
            question = instances.get("input", "") if isinstance(instances, dict) else ""
            if not question:
                # Fallback: construct from top-level fields
                q = item.get("question", "")
                opts = item.get("options", item.get("choices", {}))
                if isinstance(opts, list):
                    opts_str = "\n".join(f"Option {chr(65+i)}: {o}" for i, o in enumerate(opts))
                elif isinstance(opts, dict):
                    opts_str = "\n".join(f"Option {k}: {v}" for k, v in sorted(opts.items()))
                else:
                    opts_str = ""
                question = f"QUESTION: {q}\n{opts_str}"

            ans = get_correct_letter(item)

            # Match run_full_benchmark_suite.py format exactly
            msgs = [
                {"role": "system", "content": "Answer the medical question by selecting the best option. Reply with ONLY the letter (A, B, C, or D)."},
                {"role": "user", "content": question},
            ]
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except:
                text = f"<|im_start|>system\nAnswer the medical question by selecting the best option. Reply with ONLY the letter (A, B, C, or D).<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append(text)
            answers.append(ans)
        if not prompts:
            continue
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=4096, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for j in range(len(prompts)):
            input_len = inputs["input_ids"].shape[-1]
            gen = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True).strip()
            pred = ""
            for ch in gen:
                if ch in "ABCD":
                    pred = ch
                    break
            if pred == answers[j]:
                correct_count += 1
        if (i // BATCH) % 20 == 0:
            print(f"  Progress: {min(i+BATCH, total)}/{total} | acc={correct_count/max(min(i+BATCH, total),1)*100:.1f}%", flush=True)
    acc = correct_count / total if total > 0 else 0
    elapsed = time.time() - t0
    print(f"  {bm_key}: {acc*100:.1f}% ({correct_count}/{total}) [{elapsed:.1f}s]")
    results[bm_key] = {"accuracy": acc, "correct": correct_count, "total": total}

out = {
    "model_name": os.path.basename(MODEL_PATH),
    "model_path": MODEL_PATH,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "category": "textqa_custom",
    "benchmarks": results,
}
outpath = os.path.join(OUTPUT_DIR, f"textqa_{out['timestamp']}.json")
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {outpath}")
print(f"\nSUMMARY:")
for k, v in results.items():
    print(f"  {k}: {v['accuracy']*100:.1f}% ({v['correct']}/{v['total']})")
