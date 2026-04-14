#!/usr/bin/env python3
"""TextQA benchmark with training-aligned tool-use format.

Uses the same prompt format as GRPO training:
- System prompt with tool instructions
- apply_chat_template(messages, tools=openai_tools)
- Model responds with submit_answer tool call
- Parse answer from tool call JSON

This gives a fair evaluation of RL-trained models that learned to use tools.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/eval_textqa_tooluse_format.py \
        --model_path /path/to/merged_hf \
        --output-dir results/benchmarks_tooluse/v6_step80
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

# ── Tool definitions matching training format ──
SUBMIT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit your final answer to the medical question.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer letter (A, B, C, or D)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief reasoning for your answer choice.",
                },
            },
            "required": ["answer"],
        },
    },
}

SEARCH_EVIDENCE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_evidence",
        "description": "Search for medical evidence related to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for medical evidence.",
                },
            },
            "required": ["query"],
        },
    },
}

THINK_TOOL = {
    "type": "function",
    "function": {
        "name": "think",
        "description": "Use this tool to think through the problem step by step.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your reasoning and analysis.",
                },
            },
            "required": ["thought"],
        },
    },
}

ANALYZE_OPTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_answer_options",
        "description": "Analyze the answer options for a multiple choice question.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question text.",
                },
                "options": {
                    "type": "string",
                    "description": "The answer options.",
                },
            },
            "required": ["question"],
        },
    },
}

# Tools passed to apply_chat_template — only submit_answer for direct evaluation
OPENAI_TOOLS = [SUBMIT_ANSWER_TOOL]

# System prompt — direct answer version using training tool format
SYSTEM_PROMPT = (
    "You are a medical AI assistant. Answer the following medical question "
    "by calling the submit_answer tool with the correct answer letter and your reasoning.\n\n"
    "Respond with ONLY the tool call JSON, no other text:\n"
    "{\"name\": \"submit_answer\", \"arguments\": {\"answer\": \"A\", \"reasoning\": \"...\"}}"
)


def extract_answer_from_response(response: str) -> str:
    """Extract answer letter from model response — handles tool call format and direct answer."""
    # 1. Try to parse submit_answer tool call JSON
    # Look for {"name": "submit_answer", "arguments": {"answer": "X"}}
    try:
        # Find JSON-like patterns
        json_patterns = re.findall(r'\{[^{}]*"name"\s*:\s*"submit_answer"[^{}]*\}', response, re.DOTALL)
        if not json_patterns:
            # Try nested JSON
            json_patterns = re.findall(r'\{.*?"submit_answer".*?\}', response, re.DOTALL)
        for pat in json_patterns:
            try:
                obj = json.loads(pat)
                if "arguments" in obj:
                    ans = obj["arguments"].get("answer", "")
                    for ch in ans:
                        if ch.upper() in "ABCDE":
                            return ch.upper()
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    # 2. Try to find "answer": "X" pattern
    m = re.search(r'"answer"\s*:\s*"([A-Ea-e])"', response)
    if m:
        return m.group(1).upper()

    # 3. Try <answer>X</answer> format
    m = re.search(r'<answer>\s*([A-Ea-e])\s*</answer>', response)
    if m:
        return m.group(1).upper()

    # 4. Fallback: first letter A-E in response (after stripping think tags)
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split('</think>')[-1].strip()
    for ch in clean:
        if ch in "ABCDE":
            return ch

    return ""


def _text_answer_to_letter(question: str, answer_text: str) -> str:
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


def main():
    import argparse
    import torch
    from transformers import AutoConfig, AutoTokenizer

    parser = argparse.ArgumentParser(description="TextQA with training-aligned tool-use format")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/benchmarks_tooluse")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size (0=auto)")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking (append </think>)")
    args = parser.parse_args()

    # Only set CUDA_VISIBLE_DEVICES if not already set by shell (e.g., parallel launch)
    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["PYTHONUNBUFFERED"] = "1"

    model_path = args.model_path
    model_name = Path(model_path).name

    print(f"\n{'#'*70}", flush=True)
    print(f"  TextQA (Tool-Use Format) — Training-Aligned Evaluation", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"  Format: apply_chat_template(tools=openai_tools)", flush=True)
    print(f"{'#'*70}\n", flush=True)

    # Load model
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(model_config, "model_type", "")
    is_qwen3_5 = model_type == "qwen3_5"

    # device_map={"": 0} ensures model loads on the single GPU visible via CUDA_VISIBLE_DEVICES
    load_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": 0})
    load_kwargs["attn_implementation"] = "eager"

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

    # Test that apply_chat_template works with tools
    test_messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Test"}]
    try:
        test_text = tokenizer.apply_chat_template(test_messages, tools=OPENAI_TOOLS,
                                                   tokenize=False, add_generation_prompt=True)
        print(f"  Chat template with tools: OK ({len(test_text)} chars)", flush=True)
        # Show first 500 chars to verify format
        print(f"  Preview: {test_text[:500]}...", flush=True)
    except Exception as e:
        print(f"  WARNING: apply_chat_template with tools failed: {e}", flush=True)
        print(f"  Falling back to tools-in-system-prompt format", flush=True)
        OPENAI_TOOLS.clear()  # Will use system prompt only

    # Benchmark files — matching run_full_benchmark_suite.py paths
    benchmark_files = {
        "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
        "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
        "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_test.jsonl",
    }

    all_results = {}
    BATCH_SIZE = args.batch_size if args.batch_size > 0 else (1 if is_qwen3_5 else 8)

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

        print(f"\n  [{model_name}] {bm_key}: {len(data)} samples (batch={BATCH_SIZE})", flush=True)

        correct = 0
        total = 0
        t0 = time.time()

        for batch_start in range(0, len(data), BATCH_SIZE):
            batch = data[batch_start:batch_start + BATCH_SIZE]
            batch_prompts = []
            batch_answers = []

            for item in batch:
                instances = item.get("instances", {})
                question = instances.get("input", "") if isinstance(instances, dict) else ""
                answer = instances.get("output", "") if isinstance(instances, dict) else ""
                if not question:
                    continue

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]

                try:
                    if OPENAI_TOOLS:
                        text = tokenizer.apply_chat_template(
                            messages, tools=OPENAI_TOOLS,
                            tokenize=False, add_generation_prompt=True,
                        )
                    else:
                        text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                        )
                except Exception:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )

                # Disable thinking: append </think> to skip thinking phase
                if args.no_think:
                    if text.rstrip().endswith("<think>"):
                        text = text.rstrip() + "\n</think>\n"
                    elif "<think>" in text and "</think>" not in text:
                        text += "</think>\n"

                batch_prompts.append(text)
                answer_letter = _text_answer_to_letter(question, answer.strip())
                batch_answers.append(answer_letter)

            if not batch_prompts:
                continue

            inputs = tokenizer(batch_prompts, return_tensors="pt", truncation=True,
                             max_length=4096, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            print(f"    batch {batch_start}: input_shape={inputs['input_ids'].shape} generating...", flush=True)

            import time as _t
            _t0 = _t.time()
            with torch.no_grad():
                # Direct submit_answer call — thinking + tool call JSON
                # Use proper EOS + tool_call end tokens for early stopping
                eos_ids = [tokenizer.eos_token_id]
                for special in ["</tool_call>", "<|im_end|>"]:
                    tid = tokenizer.convert_tokens_to_ids(special)
                    if tid is not None and tid != tokenizer.unk_token_id:
                        eos_ids.append(tid)
                max_tokens = 512 if args.no_think else 1024
                outputs = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )
            print(f"    batch {batch_start}: done in {_t.time()-_t0:.1f}s output={outputs.shape}", flush=True)

            for j in range(len(batch_prompts)):
                input_len = inputs["input_ids"].shape[-1]
                generated = outputs[j][input_len:]
                response = tokenizer.decode(generated, skip_special_tokens=True).strip()

                pred = extract_answer_from_response(response)
                ref = batch_answers[j]

                total += 1
                if pred and ref and pred == ref:
                    correct += 1

            done = min(batch_start + BATCH_SIZE, len(data))
            if done % 100 < BATCH_SIZE or done == len(data):
                elapsed = time.time() - t0
                acc = correct / max(total, 1)
                print(f"    {bm_key}: {done}/{len(data)} acc={acc:.3f} {elapsed:.0f}s", flush=True)

        acc = correct / max(total, 1)
        all_results[bm_key] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  [{model_name}] {bm_key}: accuracy={acc:.4f} ({correct}/{total})", flush=True)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"textqa_tooluse_{ts}.json"
    report = {
        "model_name": model_name,
        "model_path": model_path,
        "format": "tool-use (training-aligned)",
        "system_prompt": SYSTEM_PROMPT,
        "tools": [t["function"]["name"] for t in OPENAI_TOOLS],
        "timestamp": datetime.now().isoformat(),
        "category": "textqa_tooluse",
        "benchmarks": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"  TextQA (Tool-Use Format) RESULTS: {model_name}", flush=True)
    print(f"  {'Benchmark':<30} {'Accuracy':>10} {'Correct':>8} {'Total':>8}", flush=True)
    print(f"  {'-'*56}", flush=True)
    for key, r in all_results.items():
        print(f"  {key:<30} {r['accuracy']:>10.4f} {r['correct']:>8} {r['total']:>8}", flush=True)
    print(f"\n  Saved: {out_path}", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
