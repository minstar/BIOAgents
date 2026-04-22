#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-turn benchmark evaluation using AgentRunner.

Runs standard medical benchmarks (MedQA, MedMCQA, MMLU, VQA-RAD, SLAKE, PathVQA)
through the full multi-turn agent loop matching the v16 training format:
  think → search_evidence / analyze_options → submit_answer

This properly evaluates RL-trained models that learned to use tools in multi-turn mode,
unlike single-turn eval which truncates before submit_answer.

Usage:
    # TextQA benchmarks (MedQA + MedMCQA + MMLU) on GPU 0
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmark_multiturn.py \
        --model_path /path/to/merged_hf \
        --benchmarks medqa medmcqa mmlu \
        --domain medical_qa \
        --output-dir results/benchmarks_multiturn/v16_step60 \
        --max-turns 5

    # VQA benchmarks on GPU 4
    CUDA_VISIBLE_DEVICES=4 python scripts/eval_benchmark_multiturn.py \
        --model_path /path/to/merged_hf \
        --benchmarks vqa_rad slake pathvqa \
        --domain visual_diagnosis \
        --output-dir results/benchmarks_multiturn/v16_step60 \
        --max-turns 5

    # Limit samples for testing
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmark_multiturn.py \
        --model_path /path/to/merged_hf \
        --benchmarks medqa \
        --max-samples 50
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False  # Qwen3.5-VL Conv3D workaround

from rouge_score import rouge_scorer as _rouge_module
_rouge_scorer = _rouge_module.RougeScorer(["rougeL"], use_stemmer=True)

# ── Lazy-loaded biobert-nli for LFQA hallucination/comprehensiveness ──
_nli_model = None
_nli_tokenizer = None
_nli_device = None


def _ensure_nli_model():
    """Load biobert-nli model on first use (lazy init)."""
    global _nli_model, _nli_tokenizer, _nli_device
    if _nli_model is not None:
        return
    from transformers import AutoModel, AutoTokenizer
    _nli_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading biobert-nli for hallucination/comprehensiveness scoring...")
    _nli_model = AutoModel.from_pretrained("gsarti/biobert-nli").to(_nli_device)
    _nli_tokenizer = AutoTokenizer.from_pretrained("gsarti/biobert-nli")
    _nli_model.eval()
    logger.info(f"biobert-nli loaded on {_nli_device}")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from loguru import logger

# ── Benchmark file registry ──
BENCHMARK_FILES = {
    # TextQA (MC)
    "medqa": "evaluations/self-biorag/data/benchmark/med_qa_test.jsonl",
    "medmcqa": "evaluations/self-biorag/data/benchmark/medmc_qa_test.jsonl",
    "mmlu": "evaluations/self-biorag/data/benchmark/mmlu_test.jsonl",
    # MMLU subtypes
    "mmlu_anatomy": "evaluations/self-biorag/data/benchmark/mmlu_anatomy_test.jsonl",
    "mmlu_clinical": "evaluations/self-biorag/data/benchmark/mmlu_clinical_knowledge_test.jsonl",
    "mmlu_professional": "evaluations/self-biorag/data/benchmark/mmlu_professional_medicine_test.jsonl",
    "mmlu_genetics": "evaluations/self-biorag/data/benchmark/mmlu_medical_genetics_test.jsonl",
    "mmlu_biology": "evaluations/self-biorag/data/benchmark/mmlu_college_biology_test.jsonl",
    "mmlu_college_med": "evaluations/self-biorag/data/benchmark/mmlu_college_medicine_test.jsonl",
    # MedLFQA (long-form)
    "kqa_golden": "evaluations/OLAPH/MedLFQA/kqa_golden_test_MedLFQA.jsonl",
    "live_qa": "evaluations/OLAPH/MedLFQA/live_qa_test_MedLFQA.jsonl",
    "medication_qa": "evaluations/OLAPH/MedLFQA/medication_qa_test_MedLFQA.jsonl",
    "healthsearch_qa": "evaluations/OLAPH/MedLFQA/healthsearch_qa_test_MedLFQA.jsonl",
    "kqa_silver": "evaluations/OLAPH/MedLFQA/kqa_silver_wogold_test_MedLFQA.jsonl",
    # VQA (loaded differently - via datasets)
    "vqa_rad": "datasets/vqa/vqa_rad",
    "slake": "datasets/vqa/slake",
    "pathvqa": "datasets/vqa/pathvqa",
    "pmc_vqa": "datasets/vqa/pmc_vqa",
    "vqa_med_2021": "datasets/vqa/vqa_med_2021",
    "quilt_vqa": "datasets/vqa/quilt_vqa",
    # EHR (loaded differently - from JSON with tasks array)
    "mimic_iii": "data/ehr_benchmarks/mimic_iii_bench.json",
    "eicu": "data/ehr_benchmarks/eicu_bench.json",
}

# Domain mapping for benchmarks
BENCHMARK_DOMAIN = {
    "medqa": "medical_qa",
    "medmcqa": "medical_qa",
    "mmlu": "medical_qa",
    "mmlu_anatomy": "medical_qa",
    "mmlu_clinical": "medical_qa",
    "mmlu_professional": "medical_qa",
    "mmlu_genetics": "medical_qa",
    "mmlu_biology": "medical_qa",
    "mmlu_college_med": "medical_qa",
    "kqa_golden": "medical_qa",
    "live_qa": "medical_qa",
    "medication_qa": "medical_qa",
    "healthsearch_qa": "medical_qa",
    "kqa_silver": "medical_qa",
    "vqa_rad": "visual_diagnosis",
    "slake": "visual_diagnosis",
    "pathvqa": "visual_diagnosis",
    "pmc_vqa": "visual_diagnosis",
    "vqa_med_2021": "visual_diagnosis",
    "quilt_vqa": "visual_diagnosis",
    "mimic_iii": "ehr_management",
    "eicu": "ehr_management",
}


def load_textqa_benchmark(name: str) -> list[dict]:
    """Load a TextQA benchmark from JSONL and convert to task format."""
    filepath = PROJECT_ROOT / BENCHMARK_FILES[name]
    if not filepath.exists():
        logger.error(f"Benchmark file not found: {filepath}")
        return []

    # MedLFQA benchmarks
    MEDLFQA_BENCHMARKS = {"kqa_golden", "live_qa", "medication_qa", "healthsearch_qa", "kqa_silver"}

    tasks = []
    with open(filepath) as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)

            # Handle MedLFQA format (Question / Free_form_answer)
            must_have = []
            nice_to_have = []
            if name in MEDLFQA_BENCHMARKS:
                question = item.get("Question", "")
                answer = item.get("Free_form_answer", "").strip()
                must_have = item.get("Must_have", [])
                nice_to_have = item.get("Nice_to_have", [])
            else:
                instances = item.get("instances", {})
                question = instances.get("input", "")
                answer = instances.get("output", "").strip()

            # Extract options from question text
            options = {}
            for letter in "ABCDE":
                pat = rf"Option {letter}:\s*(.+?)(?=Option [A-E]:|$)"
                m = re.search(pat, question, re.DOTALL)
                if m:
                    options[letter] = m.group(1).strip()

            task = {
                "id": f"{name}_{idx}",
                "description": {
                    "purpose": f"Answer {name} question",
                    "difficulty": "medium",
                    "category": name,
                },
                "ticket": question,
                "correct_answer": answer,
                "answer": answer,
                "options": options,
                "must_have": must_have,
                "nice_to_have": nice_to_have,
                "evaluation_criteria": {
                    "actions": [
                        {"name": "submit_answer", "arguments": {"answer": answer}}
                    ],
                    "nl_assertions": [],
                    "reward_basis": ["ACTION"],
                },
            }
            tasks.append(task)

    logger.info(f"Loaded {len(tasks)} tasks from {name}")
    return tasks


def load_vqa_benchmark(name: str) -> list[dict]:
    """Load a VQA benchmark and convert to task format."""
    data_dir = PROJECT_ROOT / BENCHMARK_FILES[name]

    # Try to load test split from HuggingFace datasets cache or local JSON
    test_file = data_dir / "test.json"
    if not test_file.exists():
        test_file = data_dir / "test.jsonl"
    if not test_file.exists():
        # Try loading via datasets library
        try:
            return _load_vqa_from_datasets(name, data_dir)
        except Exception as e:
            logger.error(f"Cannot load VQA benchmark {name}: {e}")
            return []

    tasks = []
    with open(test_file) as f:
        if test_file.suffix == ".jsonl":
            items = [json.loads(l) for l in f if l.strip()]
        else:
            items = json.load(f)

    for idx, item in enumerate(items):
        question = item.get("question", item.get("input", ""))
        answer = str(item.get("answer", item.get("output", ""))).strip()
        image_path = item.get("image_path", item.get("image", ""))

        # Make image path absolute if relative
        if image_path and not Path(image_path).is_absolute():
            image_path = str(data_dir / "images" / image_path)

        task = {
            "id": f"{name}_{idx}",
            "description": {
                "purpose": f"Answer {name} visual question",
                "difficulty": "medium",
                "category": name,
            },
            "ticket": question,
            "correct_answer": answer,
            "answer": answer,
            "_image_path": image_path if image_path else None,
            "evaluation_criteria": {
                "actions": [
                    {"name": "submit_answer", "arguments": {"answer": answer}}
                ],
                "nl_assertions": [],
                "reward_basis": ["ACTION"],
            },
        }
        tasks.append(task)

    logger.info(f"Loaded {len(tasks)} tasks from {name}")
    return tasks


def load_ehr_benchmark(name: str) -> list[dict]:
    """Load an EHR benchmark from JSON (already in task format)."""
    filepath = PROJECT_ROOT / BENCHMARK_FILES[name]
    if not filepath.exists():
        logger.error(f"EHR benchmark file not found: {filepath}")
        return []

    with open(filepath) as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    # EHR tasks are already in AgentRunner task format
    # Add correct_answer field if missing
    for task in tasks:
        if "correct_answer" not in task:
            task["correct_answer"] = ""
        if "answer" not in task:
            task["answer"] = ""

    logger.info(f"Loaded {len(tasks)} tasks from {name}")
    return tasks


def _load_vqa_from_datasets(name: str, data_dir: Path) -> list[dict]:
    """Load VQA benchmark using the datasets library or raw files."""
    # Check for commonly used file patterns
    for pattern in ["*test*.json", "*test*.jsonl", "*test*.csv"]:
        matches = list(data_dir.glob(pattern))
        if matches:
            logger.info(f"Found {matches[0]} for {name}")
            with open(matches[0]) as f:
                if matches[0].suffix == ".jsonl":
                    items = [json.loads(l) for l in f if l.strip()]
                else:
                    items = json.load(f)
            tasks = []
            for idx, item in enumerate(items):
                q = item.get("question", item.get("input", ""))
                a = str(item.get("answer", item.get("output", ""))).strip()
                tasks.append({
                    "id": f"{name}_{idx}",
                    "ticket": q,
                    "correct_answer": a,
                    "answer": a,
                    "description": {"purpose": f"{name} VQA", "category": name},
                    "evaluation_criteria": {
                        "actions": [{"name": "submit_answer"}],
                        "nl_assertions": [],
                        "reward_basis": ["ACTION"],
                    },
                })
            return tasks
    raise FileNotFoundError(f"No test files found in {data_dir}")


def _run_single_task_multiturn(runner, task, env, max_turns):
    """Run a single task with multi-turn loop + forced submit on last turn.

    Unlike AgentRunner.run_task(), this injects a nudge message on the
    penultimate turn telling the model it MUST submit now, preventing
    the model from exhausting all turns on search/think without submitting.
    """
    from bioagents.evaluation.agent_runner import (
        TurnRecord, TaskResult, build_system_prompt, parse_tool_call,
    )

    task_id = task["id"]

    # Reset environment
    obs, info = env.reset(options={"task_id": task_id})

    # Build conversation
    tools_for_prompt = info["tools"]
    if runner.config.no_think and tools_for_prompt:
        tools_for_prompt = [
            t for t in tools_for_prompt
            if t.get("function", {}).get("name") != "think"
        ]

    system_prompt = build_system_prompt(
        info["policy"], tools_for_prompt,
        domain=runner.config.domain, task=task,
    )
    # Build user message — include image for VQA tasks
    image_path = task.get("_image_path")
    if image_path and os.path.exists(image_path):
        user_content = [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": obs},
        ]
    else:
        user_content = obs

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    openai_tools = tools_for_prompt

    turns = []
    submitted_answer = ""
    t_total = 0.0

    for turn_idx in range(max_turns):
        # On penultimate turn, inject submit nudge
        if turn_idx == max_turns - 1 and not submitted_answer:
            messages.append({
                "role": "user",
                "content": (
                    "IMPORTANT: This is your LAST turn. You MUST call submit_answer now "
                    "with your best answer based on the information you have gathered. "
                    "Do NOT call any other tool. Call submit_answer immediately."
                ),
            })

        t0 = time.time()
        raw_output = runner.generate(messages, tools=openai_tools)
        latency = time.time() - t0
        t_total += latency

        turn = TurnRecord(turn_idx=turn_idx, raw_output=raw_output, latency_seconds=latency)

        # Parse tool call
        tool_call = parse_tool_call(raw_output)

        if tool_call is not None:
            turn.parsed_tool_call = tool_call
            tool_name = tool_call.get("name", "")

            # Execute tool via environment
            action = json.dumps(tool_call)
            observation, reward, terminated, truncated, step_info = env.step(action)

            if isinstance(observation, (dict, list)):
                observation_str = json.dumps(observation, indent=2, ensure_ascii=False)
            else:
                observation_str = str(observation) if observation is not None else ""

            turn.tool_response = observation_str
            messages.append({"role": "assistant", "content": json.dumps(tool_call)})
            messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}:\n{observation_str}",
            })
            turns.append(turn)

            if tool_name == "submit_answer":
                submitted_answer = tool_call.get("arguments", {}).get("answer", "")
                break

            if terminated or truncated:
                break

            # Detect repetition
            if len(turns) >= 3:
                recent_names = [
                    t.parsed_tool_call.get("name", "") if t.parsed_tool_call else ""
                    for t in turns[-3:]
                ]
                if len(set(recent_names)) == 1 and recent_names[0] not in ("submit_answer", ""):
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have called '{recent_names[0]}' multiple times. "
                            "Please submit your final answer now using submit_answer."
                        ),
                    })
        else:
            turn.is_final_answer = True
            messages.append({"role": "assistant", "content": raw_output})
            turns.append(turn)
            break

    return turns, submitted_answer, t_total, env._tool_call_log


def run_benchmark_multiturn(
    benchmark_name: str,
    tasks: list[dict],
    runner,
    domain: str,
    max_turns: int,
    output_dir: Path,
    resume_from: int = 0,
):
    """Run a benchmark through multi-turn AgentRunner loop.

    Returns:
        dict with accuracy, avg_turns, avg_reward, per-sample results
    """
    from bioagents.gym.agent_env import BioAgentGymEnv

    LFQA_BENCHMARKS = {"kqa_golden", "live_qa", "medication_qa", "healthsearch_qa", "kqa_silver"}
    is_lfqa = benchmark_name in LFQA_BENCHMARKS

    results = []
    correct = 0
    total = 0
    rouge_l_sum = 0.0
    hall_sum = 0.0
    comp_sum = 0.0
    t_start = time.time()

    for i, task in enumerate(tasks):
        if i < resume_from:
            continue

        # Create fresh env for each task
        env = BioAgentGymEnv(domain=domain, max_turns=max_turns)

        # Inject task into env's task map
        env._task_map[task["id"]] = task
        env._tasks.append(task)

        try:
            turns, submitted, latency, tool_log = _run_single_task_multiturn(
                runner, task, env, max_turns
            )

            # If no submit_answer was called, extract from last turn output
            if not submitted and turns:
                submitted = _extract_answer_fallback(turns[-1].raw_output)

            gold = task["correct_answer"].strip()
            options = task.get("options", {})

            if is_lfqa:
                # LFQA: compute ROUGE-L on submitted answer vs gold
                rouge_l = _compute_rouge_l(submitted, gold)
                rouge_l_sum += rouge_l
                is_correct = rouge_l >= 0.3  # threshold for binary correct
                # Hallucination & comprehensiveness via biobert-nli
                must_have = task.get("must_have", [])
                nice_to_have = task.get("nice_to_have", [])
                hall = _compute_hallucination(submitted, must_have, nice_to_have)
                comp = _compute_comprehensiveness(submitted, must_have)
                hall_sum += hall
                comp_sum += comp
            else:
                # MC/VQA: exact/letter match
                is_correct = _check_answer(submitted, gold, options)
                rouge_l = None

            if is_correct:
                correct += 1
            total += 1

            result_entry = {
                "task_id": task["id"],
                "gold": gold,
                "submitted": submitted,
                "correct": is_correct,
                "turns": len(turns),
                "latency": latency,
            }
            if is_lfqa:
                result_entry["rouge_l"] = round(rouge_l, 4)
                result_entry["hallucination"] = round(hall, 2)
                result_entry["comprehensiveness"] = round(comp, 2)
            results.append(result_entry)

            # Progress logging every 10 samples
            if total % 10 == 0:
                elapsed = time.time() - t_start
                rate = total / elapsed * 60
                eta = (len(tasks) - total) / max(rate, 0.01)
                if is_lfqa:
                    avg_rl = rouge_l_sum / total
                    avg_h = hall_sum / total
                    avg_c = comp_sum / total
                    logger.info(
                        f"  [{benchmark_name}] {total}/{len(tasks)} "
                        f"rouge_l={avg_rl:.3f} hall={avg_h:.1f}% comp={avg_c:.1f}% "
                        f"rate={rate:.1f}/min ETA={eta:.0f}min"
                    )
                else:
                    acc = correct / total
                    logger.info(
                        f"  [{benchmark_name}] {total}/{len(tasks)} "
                        f"acc={acc:.3f} rate={rate:.1f}/min ETA={eta:.0f}min"
                    )

            # Periodic save every 100 samples
            if total % 100 == 0:
                _save_partial(benchmark_name, results, correct, total, output_dir)

        except Exception as e:
            logger.error(f"Error on task {task['id']}: {e}")
            total += 1
            result_entry = {
                "task_id": task["id"],
                "gold": task["correct_answer"],
                "submitted": "",
                "correct": False,
                "turns": 0,
                "error": str(e),
            }
            if is_lfqa:
                result_entry["rouge_l"] = 0.0
                result_entry["hallucination"] = 100.0
                result_entry["comprehensiveness"] = 0.0
                hall_sum += 100.0
            results.append(result_entry)

    elapsed = time.time() - t_start
    accuracy = correct / max(total, 1)

    summary = {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_turns": sum(r.get("turns", 0) for r in results) / max(len(results), 1),
        "avg_reward": sum(r.get("reward", 0) for r in results) / max(len(results), 1),
        "avg_latency": sum(r.get("latency", 0) for r in results) / max(len(results), 1),
        "total_time_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    if is_lfqa:
        summary["avg_rouge_l"] = rouge_l_sum / max(total, 1)
        summary["avg_hallucination"] = hall_sum / max(total, 1)
        summary["avg_comprehensiveness"] = comp_sum / max(total, 1)
        summary["metric"] = "rouge_l + hallucination + comprehensiveness"

    # Save final results
    out_path = output_dir / f"{benchmark_name}_multiturn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if is_lfqa:
        logger.info(
            f"\n{'='*60}\n"
            f"  {benchmark_name}: rouge_l={summary['avg_rouge_l']:.3f} "
            f"hall={summary['avg_hallucination']:.1f}% comp={summary['avg_comprehensiveness']:.1f}%\n"
            f"  (correct@0.3={correct}/{total}, acc={accuracy:.3f})\n"
            f"  avg_turns={summary['avg_turns']:.1f}  avg_reward={summary['avg_reward']:.3f}\n"
            f"  time={elapsed:.0f}s  saved={out_path}\n"
            f"{'='*60}"
        )
    else:
        logger.info(
            f"\n{'='*60}\n"
            f"  {benchmark_name}: accuracy={accuracy:.3f} ({correct}/{total})\n"
            f"  avg_turns={summary['avg_turns']:.1f}  avg_reward={summary['avg_reward']:.3f}\n"
            f"  time={elapsed:.0f}s  saved={out_path}\n"
            f"{'='*60}"
        )

    return summary


def _check_answer(submitted: str, gold: str, options: dict) -> bool:
    """Check if submitted answer matches gold, handling letter/text mismatches.

    Cases:
    - Gold is letter "D", submitted is "D" → exact match
    - Gold is text "Cross-linking of DNA", submitted is "D" → check if options["D"] matches gold
    - Gold is text, submitted is text → case-insensitive match
    - Submitted is letter, gold is text → find which letter has gold text, compare
    """
    submitted = submitted.strip()
    gold = gold.strip()

    if not submitted:
        return False

    # Direct match (case-insensitive)
    if submitted.lower() == gold.lower():
        return True

    # Gold is a short letter (A-E)
    if len(gold) <= 2 and gold.upper() in "ABCDE":
        # Check if submitted starts with the gold letter
        if submitted.upper().startswith(gold.upper()):
            return True
        # Check if submitted text matches the option text for the gold letter
        gold_text = options.get(gold.upper(), "")
        if gold_text and submitted.lower() == gold_text.lower():
            return True
        return False

    # Gold is full text — find which letter it corresponds to
    gold_letter = None
    for letter, text in options.items():
        if text.strip().lower() == gold.lower():
            gold_letter = letter
            break

    if gold_letter:
        # Submitted is a letter
        first_char = submitted[0].upper() if submitted else ""
        if first_char == gold_letter.upper():
            return True
        # Submitted starts with "X." or "X)" pattern
        m = re.match(r'^([A-E])[.\):\s]', submitted.upper())
        if m and m.group(1) == gold_letter.upper():
            return True

    # Substring match for free-text answers
    if gold.lower() in submitted.lower():
        return True

    return False


def _compute_rouge_l(submitted: str, gold: str) -> float:
    """Compute ROUGE-L F1 between submitted answer and gold reference."""
    if not submitted or not gold:
        return 0.0
    scores = _rouge_scorer.score(gold, submitted)
    return scores["rougeL"].fmeasure


def _nli_cosine(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using biobert-nli."""
    _ensure_nli_model()
    encoded = _nli_tokenizer(
        [text_a, text_b], padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(_nli_device)
    with torch.no_grad():
        output = _nli_model(**encoded)
    # Mean pooling
    embs = output[0]
    mask = encoded["attention_mask"].unsqueeze(-1).expand(embs.size()).float()
    pooled = (embs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.cosine_similarity(pooled[0:1], pooled[1:2]).item()


def _compute_hallucination(submitted: str, must_have: list, nice_to_have: list) -> float:
    """Hallucination rate: % of statements with cosine < 0.5."""
    all_stmts = must_have + nice_to_have
    if not all_stmts or not submitted:
        return 100.0
    hall = sum(1 for s in all_stmts if _nli_cosine(submitted, s) < 0.5)
    return hall / len(all_stmts) * 100


def _compute_comprehensiveness(submitted: str, must_have: list) -> float:
    """Comprehensiveness: % of must-have statements with cosine >= 0.5."""
    if not must_have or not submitted:
        return 0.0
    comp = sum(1 for s in must_have if _nli_cosine(submitted, s) >= 0.5)
    return comp / len(must_have) * 100


def _extract_answer_fallback(text: str) -> str:
    """Extract answer from raw text when no submit_answer was called."""
    # Try Qwen3.5 XML format
    m = re.search(r'<parameter=answer>\s*(.*?)\s*</parameter>', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try JSON format
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    if m:
        return m.group(1).strip()

    # Try "The answer is X" pattern
    m = re.search(r'(?:the answer is|answer:)\s*([A-E])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return text[:100].strip()


def _save_partial(benchmark_name, results, correct, total, output_dir):
    """Save partial results for resumability."""
    partial = {
        "benchmark": benchmark_name,
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "results": results,
    }
    path = output_dir / f"{benchmark_name}_partial.json"
    with open(path, "w") as f:
        json.dump(partial, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Multi-turn benchmark evaluation")
    parser.add_argument("--model_path", required=True, help="Path to merged HF checkpoint")
    parser.add_argument("--benchmarks", nargs="+", default=["medqa"],
                        choices=list(BENCHMARK_FILES.keys()),
                        help="Benchmarks to evaluate")
    parser.add_argument("--domain", default=None,
                        help="Override domain (default: auto from benchmark)")
    parser.add_argument("--output-dir", default="results/benchmarks_multiturn",
                        help="Output directory")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max turns per task (default 10: think/search cycles + submit)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per benchmark (0=all)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max new tokens per turn")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume from sample index")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable think() tool (ablation)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once, reuse across benchmarks
    from bioagents.evaluation.agent_runner import AgentRunner, RunConfig

    # Use the first benchmark's domain for model loading
    first_domain = args.domain or BENCHMARK_DOMAIN[args.benchmarks[0]]

    config = RunConfig(
        model_name_or_path=args.model_path,
        backend="transformers",
        domain=first_domain,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        log_dir=str(output_dir / "logs"),
        no_think=args.no_think,
    )

    logger.info(f"Loading model: {args.model_path}")
    runner = AgentRunner(config)
    runner.load_model()
    logger.info("Model loaded successfully")

    all_summaries = {}

    for bench_name in args.benchmarks:
        domain = args.domain or BENCHMARK_DOMAIN[bench_name]

        # Update runner's domain config for this benchmark
        runner.config.domain = domain

        # Load benchmark data
        VQA_BENCHMARKS = {"vqa_rad", "slake", "pathvqa", "pmc_vqa", "vqa_med_2021", "quilt_vqa"}
        EHR_BENCHMARKS = {"mimic_iii", "eicu"}
        if bench_name in VQA_BENCHMARKS:
            tasks = load_vqa_benchmark(bench_name)
        elif bench_name in EHR_BENCHMARKS:
            tasks = load_ehr_benchmark(bench_name)
        else:
            tasks = load_textqa_benchmark(bench_name)

        if not tasks:
            logger.warning(f"No tasks loaded for {bench_name}, skipping")
            continue

        # Apply offset (resume-from) first, then max_samples limit
        if args.resume_from > 0:
            tasks = tasks[args.resume_from:]
        if args.max_samples > 0:
            tasks = tasks[:args.max_samples]

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {bench_name}: {len(tasks)} samples (offset={args.resume_from}), domain={domain}, max_turns={args.max_turns}")
        logger.info(f"{'='*60}")

        summary = run_benchmark_multiturn(
            benchmark_name=bench_name,
            tasks=tasks,
            runner=runner,
            domain=domain,
            max_turns=args.max_turns,
            output_dir=output_dir,
            resume_from=0,  # Already applied above
        )
        all_summaries[bench_name] = {
            "accuracy": summary["accuracy"],
            "correct": summary["correct"],
            "total": summary["total"],
            "avg_turns": summary["avg_turns"],
            "time": summary["total_time_seconds"],
        }

    # Print final comparison table
    logger.info(f"\n{'='*60}")
    logger.info("MULTI-TURN BENCHMARK RESULTS")
    logger.info(f"Model: {Path(args.model_path).name}")
    logger.info(f"{'Benchmark':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'Turns':>8} {'Time':>10}")
    logger.info("-" * 65)
    for name, s in all_summaries.items():
        logger.info(
            f"{name:<15} {s['accuracy']:>9.3f} {s['correct']:>9d} "
            f"{s['total']:>7d} {s['avg_turns']:>7.1f} {s['time']:>9.0f}s"
        )
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
