#!/usr/bin/env python3
"""Full 21-Benchmark Evaluation — 3 Models × 4 Categories, Parallel.

Evaluates ALL 3 VL models on the COMPLETE benchmark suite:

  Category 1: Text MC QA (8 benchmarks, 7,634 examples)
    - MedQA: 1,273
    - MedMCQA: 4,183
    - MMLU ×6: anatomy(135), clinical(265), professional(272),
               genetics(100), biology(144), college_med(173)

  Category 2: Vision QA (6 benchmarks, VL models only)
    - VQA-RAD, SLAKE, PathVQA, PMC-VQA, VQA-Med-2021, Quilt-VQA

  Category 3: Long-Form QA — MedLFQA (5 benchmarks, 4,948 examples)
    - KQA Golden(201), LiveQA(100), MedicationQA(666),
      HealthSearchQA(3,077), KQA Silver(904)

  Category 4: EHR Benchmarks (2 databases, 100 tasks)
    - MIMIC-III(50), eICU(50)

  TOTAL: 21 benchmarks, ~12,700+ test examples

Models (3 VL, fixed):
    - Lingshu-7B (GPUs 0,1)
    - Qwen2.5-VL-7B-Instruct (GPUs 2,3)
    - Step3-VL-10B (GPUs 4,5,6,7)

Usage:
    # Run everything (3 models in parallel, all 21 benchmarks)
    python scripts/run_all_benchmarks_parallel.py

    # Single model
    python scripts/run_all_benchmarks_parallel.py --model lingshu

    # Specific category
    python scripts/run_all_benchmarks_parallel.py --category textqa
    python scripts/run_all_benchmarks_parallel.py --category vqa
    python scripts/run_all_benchmarks_parallel.py --category medlfqa
    python scripts/run_all_benchmarks_parallel.py --category ehr
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use .venv Python if available (project-specific env with all deps)
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists():
    PYTHON_EXE = str(VENV_PYTHON)
else:
    PYTHON_EXE = sys.executable

# ============================================================
# Model Configuration — 3 VL Models, 8 GPUs total
# ============================================================

MODELS = {
    "lingshu": {
        "name": "Lingshu-7B",
        "path": str(PROJECT_ROOT / "checkpoints/models/Lingshu-7B"),
        "gpus": "0,1",          # 7B → 2 GPUs
        "supports_vision": True,
    },
    "qwen2vl": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "path": str(PROJECT_ROOT / "checkpoints/models/Qwen2.5-VL-7B-Instruct"),
        "gpus": "2,3",          # 7B → 2 GPUs
        "supports_vision": True,
    },
    "step3vl": {
        "name": "Step3-VL-10B",
        "path": str(PROJECT_ROOT / "checkpoints/models/Step3-VL-10B"),
        "gpus": "4",             # 10B bf16 ≈ 20GB, single A100-80GB, custom attn needs single device
        "supports_vision": True,
    },
}

# ============================================================
# Worker Script — Runs ALL benchmarks for a single model
# ============================================================

WORKER_SCRIPT = '''#!/usr/bin/env python3
"""Worker: evaluate a single model on all benchmark categories."""
import json, os, re, sys, time, gc
from datetime import datetime
from pathlib import Path
from collections import Counter

# ── PROJECT_ROOT injected from master script (absolute path) ──
PROJECT_ROOT = Path("{project_root}")
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_KEY = "{model_key}"
MODEL_NAME = "{model_name}"
MODEL_PATH = "{model_path}"
CATEGORIES = {categories}
OUTPUT_DIR = Path("{output_dir}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ── Helpers ─────────────────────────────────────────────

def _load_step_vl_model(model_path):
    """Load Step3-VL model with manual weight key remapping.

    The checkpoint stores weights as:
      - vision_model.XXX, model.XXX (language model), vit_large_projector.XXX
    But Step3VL10BForCausalLM expects:
      - model.vision_model.XXX, model.language_model.XXX, model.vit_large_projector.XXX

    We load the config, instantiate model on meta device, remap weights, then dispatch.
    """
    import safetensors.torch
    from pathlib import Path
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
    from transformers import AutoConfig, AutoModelForCausalLM

    model_dir = Path(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    print(f"  [Step3-VL] Loading safetensors and remapping keys...", flush=True)

    # Load all shards and remap keys
    shard_files = sorted(model_dir.glob("model-*.safetensors"))
    remapped = {{}}
    for shard in shard_files:
        sd = safetensors.torch.load_file(str(shard))
        for k, v in sd.items():
            new_key = k
            if k.startswith("vision_model."):
                new_key = "model." + k
            elif k.startswith("vit_large_projector."):
                new_key = "model." + k
            elif k.startswith("model.") and not k.startswith("model.language_model.") and not k.startswith("model.vision_model."):
                suffix = k[len("model."):]
                new_key = "model.language_model." + suffix
            elif k == "lm_head.weight":
                new_key = k
            remapped[new_key] = v.to(torch.bfloat16)

    # Save remapped checkpoint to temp file
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / "model_remapped.safetensors"
    safetensors.torch.save_file(remapped, str(tmp_path))
    del remapped
    gc.collect()

    print(f"  [Step3-VL] Loading model with remapped weights...", flush=True)

    # Initialize model on meta device
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Load with accelerate dispatch — single GPU to avoid cross-device issues
    # 10B model in bf16 ≈ 20GB, fits in a single A100-80GB
    model = load_checkpoint_and_dispatch(
        model, str(tmp_path), device_map={{"": 0}}, dtype=torch.bfloat16
    )

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"  [Step3-VL] Model loaded with remapped weights successfully!", flush=True)
    return model


def load_model():
    """Load model with correct class for VL vs causal."""
    print(f"[{{MODEL_NAME}}] Loading model...", flush=True)
    t0 = time.time()
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    architectures = getattr(config, "architectures", [])

    load_kwargs = dict(
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # Detect model class based on config
    is_qwen_vl = "qwen2" in model_type.lower() and "vl" in model_type.lower()
    is_step_vl = "step" in model_type.lower() or any("step" in a.lower() for a in architectures)

    if is_qwen_vl:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, **load_kwargs)
        except Exception as e:
            print(f"[{{MODEL_NAME}}] Qwen2_5_VL class failed, fallback to AutoModel: {{e}}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)
    elif is_step_vl:
        # Step3-VL: custom loading with weight key remapping
        # _checkpoint_conversion_mapping doesn't work in this transformers version
        model = _load_step_vl_model(MODEL_PATH)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Critical for batched generation with decoder-only models

    print(f"[{{MODEL_NAME}}] Model loaded in {{time.time()-t0:.1f}}s (type={{type(model).__name__}})", flush=True)
    return model, tokenizer


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def extract_mc_answer(response, valid="ABCDE"):
    response = response.strip()
    if response and response[0].upper() in valid:
        return response[0].upper()
    for pat in [r"(?:answer|option)\\s*(?:is|:)\\s*\\(?([A-E])\\)?",
                r"\\b([A-E])\\b\\s*(?:\\)|\\.|:)",
                r"^\\s*\\(?([A-E])\\)?"]:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    for ch in valid:
        if ch in response.upper():
            return ch
    return "A"


def safe_generate(model, tokenizer, inputs, max_new_tokens=32):
    """Generate with error handling for different model types."""
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return outputs
    except Exception as e:
        print(f"  [WARNING] Generate failed: {{e}}", flush=True)
        return None


# ══════════════════════════════════════════════════════════
#  Category 1: Text MC QA (8 benchmarks)
# ══════════════════════════════════════════════════════════

TEXT_BENCHMARKS = {{
    "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
    "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
    "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_clinical_knowledge_test.jsonl",
    "mmlu_professional": "evaluations/self-biorag/data/benchmark/mmlu_professional_medicine_test.jsonl",
    "mmlu_anatomy": "evaluations/self-biorag/data/benchmark/mmlu_anatomy_test.jsonl",
    "mmlu_genetics": "evaluations/self-biorag/data/benchmark/mmlu_medical_genetics_test.jsonl",
    "mmlu_biology": "evaluations/self-biorag/data/benchmark/mmlu_college_biology_test.jsonl",
    "mmlu_college_med": "evaluations/self-biorag/data/benchmark/mmlu_college_medicine_test.jsonl",
}}


def parse_textqa_item(item):
    instances = item.get("instances", {{}})
    raw_input = instances.get("input", "")
    raw_output = instances.get("output", "")
    if not raw_input:
        question = item.get("question", "")
        options = item.get("options", {{}})
        if isinstance(options, list):
            options = {{chr(65+i): o for i, o in enumerate(options)}}
        return question, options, item.get("answer_idx", item.get("answer", ""))
    rest = raw_input.split("QUESTION:", 1)[1].strip() if "QUESTION:" in raw_input else raw_input.strip()
    option_matches = re.findall(r'Option\\s+([A-E]):\\s*(.*?)(?=Option\\s+[A-E]:|$)', rest, re.DOTALL)
    question = ""
    options = {{}}
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


def run_textqa(model, tokenizer):
    print(f"\\n[{{MODEL_NAME}}] === TEXT MC QA (8 benchmarks) ===", flush=True)
    all_results = {{}}
    grand_correct, grand_total = 0, 0
    # Step3-VL custom attention doesn't support padded batches; use bs=1
    is_step = "step" in type(model).__name__.lower()
    BATCH_SIZE = 1 if is_step else 8

    for bm_key, bm_path in TEXT_BENCHMARKS.items():
        full_path = PROJECT_ROOT / bm_path
        if not full_path.exists():
            print(f"  [SKIP] {{bm_key}}: file not found at {{full_path}}", flush=True)
            all_results[bm_key] = {{"error": "file not found"}}
            continue

        data = []
        with open(full_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))

        correct, total = 0, 0
        t0 = time.time()
        print(f"  [{{MODEL_NAME}}] {{bm_key}}: {{len(data)}} questions", flush=True)

        for batch_start in range(0, len(data), BATCH_SIZE):
            batch = data[batch_start:batch_start+BATCH_SIZE]
            prompts, answers = [], []
            for item in batch:
                q, opts, ans = parse_textqa_item(item)
                if not q or not opts:
                    continue
                prompt = f"Question: {{q}}\\n\\nOptions:\\n"
                for letter, text in sorted(opts.items()):
                    prompt += f"  {{letter}}) {{text}}\\n"
                prompt += "\\nAnswer with only the letter (A, B, C, or D):"
                msgs = [
                    {{"role": "system", "content": "You are a medical expert. Answer with only the correct option letter."}},
                    {{"role": "user", "content": prompt}},
                ]
                try:
                    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    # Fallback if chat template not supported
                    text = f"<|im_start|>system\\nYou are a medical expert. Answer with only the correct option letter.<|im_end|>\\n<|im_start|>user\\n{{prompt}}<|im_end|>\\n<|im_start|>assistant\\n"
                prompts.append(text)
                answers.append(ans)

            if not prompts:
                continue

            inputs = tokenizer(prompts, return_tensors="pt", truncation=True,
                             max_length=4096, padding=True)
            inputs = {{k: v.to(model.device) for k, v in inputs.items()}}

            outputs = safe_generate(model, tokenizer, inputs, max_new_tokens=32)
            if outputs is None:
                total += len(prompts)
                continue

            for j in range(len(prompts)):
                gen = outputs[j][inputs["input_ids"].shape[-1]:]
                resp = tokenizer.decode(gen, skip_special_tokens=True).strip()
                pred = extract_mc_answer(resp)
                ref = extract_mc_answer(answers[j]) if answers[j] else ""
                total += 1
                if pred == ref:
                    correct += 1

            done = min(batch_start+BATCH_SIZE, len(data))
            if done % 200 < BATCH_SIZE or done >= len(data):
                elapsed = time.time()-t0
                acc = correct/max(total,1)
                print(f"    {{bm_key}}: {{done}}/{{len(data)}} acc={{acc:.3f}} {{elapsed:.0f}}s", flush=True)

        acc = correct / max(total, 1)
        all_results[bm_key] = {{"accuracy": acc, "correct": correct, "total": total}}
        grand_correct += correct
        grand_total += total
        print(f"  [{{MODEL_NAME}}] {{bm_key}}: {{acc:.4f}} ({{correct}}/{{total}})", flush=True)

    overall = grand_correct / max(grand_total, 1)
    all_results["_overall"] = {{"accuracy": overall, "correct": grand_correct, "total": grand_total}}
    print(f"  [{{MODEL_NAME}}] TextQA OVERALL: {{overall:.4f}} ({{grand_correct}}/{{grand_total}})", flush=True)
    return all_results


# ══════════════════════════════════════════════════════════
#  Category 2: VQA (6 benchmarks)
# ══════════════════════════════════════════════════════════

def run_vqa(model, tokenizer):
    print(f"\\n[{{MODEL_NAME}}] === VISION QA (6 benchmarks) ===", flush=True)
    try:
        from bioagents.evaluation.vqa_benchmark_eval import VQABenchmarkConfig, VQABenchmarkEvaluator
        vqa_config = VQABenchmarkConfig(
            model_name_or_path=MODEL_PATH,
            model_name=MODEL_NAME,
            benchmarks=["vqa_rad", "slake", "pathvqa", "pmc_vqa", "vqa_med_2021", "quilt_vqa"],
            max_samples=0,
            output_dir=str(OUTPUT_DIR / "vqa"),
            use_images=True,
        )
        # Pass existing model to avoid reloading
        evaluator = VQABenchmarkEvaluator(vqa_config)
        results = evaluator.evaluate_all()
        return results
    except Exception as e:
        print(f"  [{{MODEL_NAME}}] VQA evaluation error: {{e}}", flush=True)
        import traceback; traceback.print_exc()
        return {{"error": str(e)}}


# ══════════════════════════════════════════════════════════
#  Category 3: MedLFQA (5 benchmarks)
# ══════════════════════════════════════════════════════════

MEDLFQA_DATASETS = {{
    "kqa_golden": ("evaluations/OLAPH/MedLFQA/kqa_golden_test_MedLFQA.jsonl", "KQA Golden"),
    "live_qa": ("evaluations/OLAPH/MedLFQA/live_qa_test_MedLFQA.jsonl", "LiveQA"),
    "medication_qa": ("evaluations/OLAPH/MedLFQA/medication_qa_test_MedLFQA.jsonl", "MedicationQA"),
    "healthsearch_qa": ("evaluations/OLAPH/MedLFQA/healthsearch_qa_test_MedLFQA.jsonl", "HealthSearchQA"),
    "kqa_silver": ("evaluations/OLAPH/MedLFQA/kqa_silver_wogold_test_MedLFQA.jsonl", "KQA Silver"),
}}


def compute_token_f1(pred, ref):
    pred_t = set(pred.lower().split())
    ref_t = set(ref.lower().split())
    if not pred_t or not ref_t:
        return 1.0 if pred_t == ref_t else 0.0
    common = pred_t & ref_t
    if not common:
        return 0.0
    p = len(common) / len(pred_t)
    r = len(common) / len(ref_t)
    return 2*p*r / (p+r)


def run_medlfqa(model, tokenizer):
    print(f"\\n[{{MODEL_NAME}}] === MedLFQA (5 benchmarks, 4,948 examples) ===", flush=True)
    all_results = {{}}
    is_step = "step" in type(model).__name__.lower()
    BATCH_SIZE = 1 if is_step else 4

    for ds_key, (ds_path, ds_name) in MEDLFQA_DATASETS.items():
        full_path = PROJECT_ROOT / ds_path
        if not full_path.exists():
            print(f"  [SKIP] {{ds_name}}: not found at {{full_path}}", flush=True)
            all_results[ds_key] = {{"error": "not found"}}
            continue

        data = []
        with open(full_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))

        print(f"  [{{MODEL_NAME}}] {{ds_name}}: {{len(data)}} examples", flush=True)
        f1_sum, n = 0.0, 0
        t0 = time.time()

        valid = [(i, item) for i, item in enumerate(data) if item.get("Question", "")]

        for batch_start in range(0, len(valid), BATCH_SIZE):
            batch = valid[batch_start:batch_start+BATCH_SIZE]
            prompts, refs = [], []
            for idx, item in batch:
                q = item["Question"]
                ref = item.get("Free_form_answer", "")
                msgs = [
                    {{"role": "system", "content": "You are a medical expert. Provide detailed, accurate answers."}},
                    {{"role": "user", "content": f"Question: {{q}}\\n\\nProvide a comprehensive answer."}},
                ]
                try:
                    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    text = f"<|im_start|>system\\nYou are a medical expert. Provide detailed, accurate answers.<|im_end|>\\n<|im_start|>user\\nQuestion: {{q}}\\n\\nProvide a comprehensive answer.<|im_end|>\\n<|im_start|>assistant\\n"
                prompts.append(text)
                refs.append(ref)

            if not prompts:
                continue

            inputs = tokenizer(prompts, return_tensors="pt", truncation=True,
                             max_length=4096, padding=True)
            inputs = {{k: v.to(model.device) for k, v in inputs.items()}}

            outputs = safe_generate(model, tokenizer, inputs, max_new_tokens=512)
            if outputs is None:
                n += len(prompts)
                continue

            for j in range(len(prompts)):
                gen = outputs[j][inputs["input_ids"].shape[-1]:]
                resp = tokenizer.decode(gen, skip_special_tokens=True).strip()
                f1 = compute_token_f1(resp, refs[j])
                f1_sum += f1
                n += 1

            done = min(batch_start+BATCH_SIZE, len(valid))
            if done % 100 < BATCH_SIZE or done >= len(valid):
                elapsed = time.time()-t0
                avg = f1_sum / max(n, 1)
                print(f"    {{ds_name}}: {{done}}/{{len(valid)}} F1={{avg:.3f}} {{elapsed:.0f}}s", flush=True)

        avg_f1 = f1_sum / max(n, 1)
        all_results[ds_key] = {{"name": ds_name, "token_f1": avg_f1, "total": n}}
        print(f"  [{{MODEL_NAME}}] {{ds_name}}: Token-F1={{avg_f1:.4f}} ({{n}} examples)", flush=True)

    return all_results


# ══════════════════════════════════════════════════════════
#  Category 4: EHR Benchmarks (2 databases)
# ══════════════════════════════════════════════════════════

def run_ehr(model, tokenizer):
    # Stratified sample 5000 tasks per database (from 57K+ total) to keep
    # evaluation tractable (~3 days → ~3-4 hours with bs=1 for Step3).
    EHR_MAX_SAMPLES = 5000
    print(f"\\n[{{MODEL_NAME}}] === EHR Benchmarks (MIMIC-III + eICU, stratified {{EHR_MAX_SAMPLES}}/db) ===", flush=True)
    try:
        from bioagents.evaluation.ehr_benchmark_eval import EHRBenchmarkConfig, EHRBenchmarkEvaluator
        ehr_config = EHRBenchmarkConfig(
            model_name_or_path=MODEL_PATH,
            model_name=MODEL_NAME,
            benchmarks=["mimic_iii", "eicu"],
            max_samples=EHR_MAX_SAMPLES,
            max_turns=15,
            output_dir=str(OUTPUT_DIR / "ehr"),
        )
        evaluator = EHRBenchmarkEvaluator(ehr_config)
        results = evaluator.evaluate_all()
        return results
    except Exception as e:
        print(f"  [{{MODEL_NAME}}] EHR evaluation error: {{e}}", flush=True)
        import traceback; traceback.print_exc()
        return {{"error": str(e)}}


# ══════════════════════════════════════════════════════════
#  Main Worker
# ══════════════════════════════════════════════════════════

def main():
    print(f"\\n{{'='*70}}", flush=True)
    print(f"  Full Benchmark Evaluation: {{MODEL_NAME}}", flush=True)
    print(f"  Categories: {{CATEGORIES}}", flush=True)
    print(f"  GPUs: {{os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}}", flush=True)
    print(f"  PROJECT_ROOT: {{PROJECT_ROOT}}", flush=True)
    print(f"  Output: {{OUTPUT_DIR}}", flush=True)
    print(f"{{'='*70}}\\n", flush=True)

    all_results = {{"model_name": MODEL_NAME, "model_path": MODEL_PATH,
                    "timestamp": datetime.now().isoformat()}}

    model, tokenizer = load_model()

    if "textqa" in CATEGORIES:
        all_results["textqa"] = run_textqa(model, tokenizer)

    if "medlfqa" in CATEGORIES:
        all_results["medlfqa"] = run_medlfqa(model, tokenizer)

    # Free model before VQA/EHR (they load their own)
    free_model(model)

    if "vqa" in CATEGORIES:
        all_results["vqa"] = run_vqa(None, None)

    if "ehr" in CATEGORIES:
        all_results["ehr"] = run_ehr(None, None)

    # Save final results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"full_21bench_{{MODEL_KEY}}_{{ts}}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\\n{{'#'*70}}", flush=True)
    print(f"  FINAL RESULTS: {{MODEL_NAME}}", flush=True)
    print(f"{{'#'*70}}", flush=True)

    # Print TextQA summary
    if "textqa" in all_results and isinstance(all_results["textqa"], dict):
        print(f"\\n  [Text MCQA]", flush=True)
        for k, v in all_results["textqa"].items():
            if k.startswith("_") or not isinstance(v, dict) or "accuracy" not in v:
                continue
            print(f"    {{k:25s}}: {{v['accuracy']:.4f}} ({{v['correct']}}/{{v['total']}})", flush=True)
        ov = all_results["textqa"].get("_overall", {{}})
        if ov:
            print(f"    {{'OVERALL':25s}}: {{ov['accuracy']:.4f}} ({{ov['correct']}}/{{ov['total']}})", flush=True)

    # Print MedLFQA summary
    if "medlfqa" in all_results and isinstance(all_results["medlfqa"], dict):
        print(f"\\n  [MedLFQA]", flush=True)
        for k, v in all_results["medlfqa"].items():
            if not isinstance(v, dict) or "token_f1" not in v:
                continue
            print(f"    {{v.get('name',k):25s}}: F1={{v['token_f1']:.4f}} (n={{v['total']}})", flush=True)

    print(f"\\n  Saved: {{out_path}}", flush=True)
    print(f"{{'='*70}}", flush=True)


if __name__ == "__main__":
    main()
'''


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Full 21-Benchmark Suite — 3 Models × 4 Categories, Parallel"
    )
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                        help="Which model to evaluate (default: all)")
    parser.add_argument("--category", choices=["textqa", "vqa", "medlfqa", "ehr", "all"],
                        default="all", help="Which category to evaluate (default: all)")
    parser.add_argument("--output-dir", default="results/full_21bench",
                        help="Output directory for results")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.category == "all":
        categories = ["textqa", "vqa", "medlfqa", "ehr"]
    else:
        categories = [args.category]

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    print(f"\n{'#'*70}")
    print(f"  Healthcare AI GYM — Full 21-Benchmark Evaluation")
    print(f"  Python: {PYTHON_EXE}")
    print(f"  Models: {', '.join(MODELS[m]['name'] for m in models_to_run)}")
    print(f"  Categories: {', '.join(categories)}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*70}\n")

    # Generate worker scripts and launch in parallel
    worker_dir = output_dir / "_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    for model_key in models_to_run:
        model_info = MODELS[model_key]
        model_output = output_dir / model_key
        model_output.mkdir(parents=True, exist_ok=True)

        # Generate worker script with absolute PROJECT_ROOT injected
        worker_code = WORKER_SCRIPT.format(
            project_root=str(PROJECT_ROOT),
            model_key=model_key,
            model_name=model_info["name"],
            model_path=model_info["path"],
            categories=repr(categories),
            output_dir=str(model_output),
        )
        worker_path = worker_dir / f"worker_{model_key}.py"
        with open(worker_path, "w") as f:
            f.write(worker_code)

        # Launch subprocess with .venv Python
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = model_info["gpus"]
        env["PYTHONUNBUFFERED"] = "1"
        # Ensure PYTHONPATH includes project root for bioagents module
        env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")

        log_path = output_dir / f"{model_key}_eval.log"
        print(f"  Launching {model_info['name']} on GPUs {model_info['gpus']} → {log_path}")

        with open(log_path, "w") as log_f:
            p = subprocess.Popen(
                [PYTHON_EXE, str(worker_path)],
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
        processes.append((model_key, p, log_path))

    # Monitor progress
    print(f"\n  All {len(processes)} models launched. Monitoring...\n")
    start_time = time.time()
    completed = set()

    while len(completed) < len(processes):
        time.sleep(30)
        elapsed = time.time() - start_time
        for model_key, p, log_path in processes:
            if model_key in completed:
                continue
            retcode = p.poll()
            if retcode is not None:
                completed.add(model_key)
                status = "SUCCESS" if retcode == 0 else f"FAILED (code {retcode})"
                print(f"  [{elapsed/60:.0f}m] {MODELS[model_key]['name']}: {status}", flush=True)
            else:
                # Print last progress line
                try:
                    with open(log_path) as f:
                        lines = f.readlines()
                    last_progress = ""
                    for line in reversed(lines):
                        line = line.strip()
                        if line and ("/" in line or "acc=" in line or "F1=" in line):
                            last_progress = line
                            break
                    if last_progress:
                        print(f"  [{elapsed/60:.0f}m] {MODELS[model_key]['name']}: {last_progress}", flush=True)
                except Exception:
                    pass

    total_time = time.time() - start_time
    print(f"\n  All evaluations complete in {total_time/60:.1f} minutes.")

    # ── Aggregate Results ──────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  AGGREGATED RESULTS — 21 Benchmarks × {len(models_to_run)} Models")
    print(f"{'#'*70}\n")

    all_model_results = {}
    for model_key in models_to_run:
        model_output = output_dir / model_key
        result_files = sorted(model_output.glob("full_21bench_*.json"))
        if result_files:
            with open(result_files[-1]) as f:
                all_model_results[model_key] = json.load(f)

    if all_model_results:
        # Print Text MCQA comparison
        print(f"  ── Text MC QA (8 benchmarks) ──")
        header = f"  {'Benchmark':<25}"
        for mk in models_to_run:
            header += f" | {MODELS[mk]['name']:>20}"
        print(header)
        print(f"  {'-'*len(header)}")

        for bm_key in ["medqa", "medmcqa", "mmlu_clinical", "mmlu_professional",
                        "mmlu_anatomy", "mmlu_genetics", "mmlu_biology", "mmlu_college_med", "_overall"]:
            row = f"  {bm_key:<25}"
            for mk in models_to_run:
                tq = all_model_results.get(mk, {}).get("textqa", {}).get(bm_key, {})
                if isinstance(tq, dict) and "accuracy" in tq:
                    row += f" | {tq['accuracy']:>19.1%}"
                else:
                    row += f" | {'N/A':>20}"
            print(row)

        # Print MedLFQA comparison
        print(f"\n  ── MedLFQA (5 benchmarks) ──")
        header = f"  {'Benchmark':<25}"
        for mk in models_to_run:
            header += f" | {MODELS[mk]['name']:>20}"
        print(header)
        print(f"  {'-'*len(header)}")

        for ds_key in ["kqa_golden", "live_qa", "medication_qa", "healthsearch_qa", "kqa_silver"]:
            row = f"  {ds_key:<25}"
            for mk in models_to_run:
                lf = all_model_results.get(mk, {}).get("medlfqa", {}).get(ds_key, {})
                if isinstance(lf, dict) and "token_f1" in lf:
                    row += f" | {lf['token_f1']:>19.4f}"
                else:
                    row += f" | {'N/A':>20}"
            print(row)

        # Save aggregated comparison
        comp_path = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comp_path, "w") as f:
            json.dump(all_model_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Full comparison saved: {comp_path}")

    print(f"\n{'='*70}")
    print(f"  Total evaluation time: {total_time/60:.1f} minutes")
    print(f"  Results directory: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
