#!/usr/bin/env python3
"""Quick TextQA evaluation on a custom model checkpoint."""
import json, os, re, sys, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/grpo_baseline_lingshu7b/checkpoint-800-merged"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "results/algorithm_comparison/grpo_baseline_new_step800_textqa"
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
if model_type in ("qwen2_5_vl", "qwen2_vl"):
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Model loaded!")

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
    # Parse self-biorag format
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
    correct, total = 0, len(data)
    t0 = time.time()
    print(f"\n{bm_key}: {len(data)} questions")
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
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"].shape[-1]  # total padded input length (correct for left-padding)
            gen = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            pred = re.search(r'[A-D]', gen)
            pred = pred.group(0) if pred else ""
            if pred == answers[j]:
                correct += 1
        if (i // BATCH) % 20 == 0:
            print(f"  Progress: {i+len(batch)}/{total} | acc={correct/(i+len(batch))*100:.1f}%", flush=True)
    acc = correct / total if total > 0 else 0
    elapsed = time.time() - t0
    print(f"  {bm_key}: {acc*100:.1f}% ({correct}/{total}) [{elapsed:.1f}s]")
    results[bm_key] = {"accuracy": acc, "correct": correct, "total": total}

out = {
    "model_name": os.path.basename(MODEL_PATH),
    "model_path": MODEL_PATH,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "category": "textqa_custom",
    "benchmarks": results,
}
outpath = os.path.join(OUTPUT_DIR, f"textqa_custom_{out['timestamp']}.json")
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {outpath}")
print(f"\nSUMMARY:")
for k, v in results.items():
    print(f"  {k}: {v['accuracy']*100:.1f}% ({v['correct']}/{v['total']})")
