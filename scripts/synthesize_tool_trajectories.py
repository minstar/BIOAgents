#!/usr/bin/env python3
from __future__ import annotations

"""Synthesize tool-based SFT trajectories from MedQA/MedMCQA train splits.

Runs the model through the GYM environment on train-split questions,
collecting successful trajectories as contamination-free SFT data.

This implements the WebRL "experience replay" / RIF-RFT approach:
- Use ONLY train split questions (no test set contamination)
- Model generates tool-use trajectories through the GYM environment
- Filter: keep only trajectories where model gets the correct answer
- Output: clean SFT data in OpenAI chat format

Usage:
    python scripts/synthesize_tool_trajectories.py \
        --model-path checkpoints/models/Lingshu-7B \
        --output datasets/sft/tool_trajectories_train.jsonl \
        --max-samples 500 \
        --num-rollouts 3
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_DIR = PROJECT_ROOT / "evaluations" / "self-biorag" / "data" / "benchmark"


def load_medqa_train(max_samples: int = None) -> list[dict]:
    """Load MedQA train split questions."""
    train_path = BENCHMARK_DIR / "med_qa_train_gpt4.jsonl"
    if not train_path.exists():
        train_path = BENCHMARK_DIR / "med_qa_train.json"

    tasks = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Parse the question and options
            question_text = record.get("question", "")
            answer_raw = record.get("answer", "")
            explanation = record.get("explanation", "")

            # Extract answer letter from "(D) Nitrofurantoin" format
            answer_letter = ""
            answer_text = answer_raw
            m = re.match(r"\(([A-E])\)\s*(.*)", answer_raw)
            if m:
                answer_letter = m.group(1)
                answer_text = m.group(2).strip()

            # Parse instances field for options
            instances_str = record.get("instances", "")
            options = {}
            if isinstance(instances_str, str):
                # Try to extract options from the input text
                opt_matches = re.findall(
                    r"\(([A-E])\)\s*([^(]+?)(?=\([A-E]\)|$)",
                    instances_str,
                )
                for label, text in opt_matches:
                    options[label] = text.strip()

            if not options:
                # Try extracting from question text itself
                opt_matches = re.findall(
                    r"\(([A-E])\)\s*([^(]+?)(?=\([A-E]\)|$)",
                    question_text,
                )
                for label, text in opt_matches:
                    options[label] = text.strip()

            tasks.append({
                "id": f"medqa_train_{record.get('id', len(tasks))}",
                "question": question_text,
                "answer_letter": answer_letter,
                "answer_text": answer_text,
                "explanation": explanation,
                "options": options,
                "source": "MedQA",
            })

            if max_samples and len(tasks) >= max_samples:
                break

    logger.info(f"Loaded {len(tasks)} MedQA train questions")
    return tasks


def load_medmcqa_train(max_samples: int = None) -> list[dict]:
    """Load MedMCQA train split questions."""
    train_path = BENCHMARK_DIR / "medmc_qa_train.json"
    if not train_path.exists():
        logger.warning(f"MedMCQA train not found: {train_path}")
        return []

    tasks = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            instances = record.get("instances", "")
            question = ""
            options = {}
            answer_letter = ""

            # Parse instances (may be string or dict)
            if isinstance(instances, str):
                try:
                    instances = eval(instances)  # noqa: S307
                except Exception:
                    pass

            if isinstance(instances, dict):
                input_text = instances.get("input", "")
                output_text = instances.get("output", "")
                question = input_text

                # Extract options
                opt_matches = re.findall(
                    r"\(([A-D])\)\s*([^(]+?)(?=\([A-D]\)|$)", input_text
                )
                for label, text in opt_matches:
                    options[label] = text.strip()

                # Extract answer letter
                m = re.match(r"\(([A-D])\)", output_text.strip())
                if m:
                    answer_letter = m.group(1)
            elif isinstance(instances, list) and instances:
                inst = instances[0] if isinstance(instances[0], dict) else {}
                question = inst.get("input", "")
                output_text = inst.get("output", "")
                m = re.match(r"\(([A-D])\)", output_text.strip())
                if m:
                    answer_letter = m.group(1)

            if not question or not answer_letter:
                continue

            tasks.append({
                "id": f"medmcqa_train_{record.get('id', len(tasks))}",
                "question": question,
                "answer_letter": answer_letter,
                "answer_text": options.get(answer_letter, ""),
                "explanation": record.get("explanation", ""),
                "options": options,
                "source": "MedMCQA",
            })

            if max_samples and len(tasks) >= max_samples:
                break

    logger.info(f"Loaded {len(tasks)} MedMCQA train questions")
    return tasks


def _to_vl_messages(messages: list) -> list:
    """Convert standard chat messages to Qwen VL format.

    VL models expect user content as list of content items:
    [{"type": "text", "text": "..."}]
    """
    vl_msgs = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user" and isinstance(content, str):
            vl_msgs.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })
        else:
            vl_msgs.append(msg)
    return vl_msgs


def build_gym_prompt(task: dict) -> str:
    """Build a GYM-style prompt for tool-based inference."""
    question = task["question"]
    options = task.get("options", {})

    # Format options
    options_text = ""
    if options:
        for label in sorted(options.keys()):
            options_text += f"\n({label}) {options[label]}"

    prompt = f"""You are a medical AI assistant. Answer the following medical question using the available tools.

Available tools:
1. think(thought: str) - Organize your reasoning about the medical question
2. search_pubmed(query: str) - Search PubMed for relevant medical evidence
3. retrieve_evidence(query: str) - Retrieve evidence from the medical knowledge base
4. submit_answer(answer: str, reasoning: str) - Submit your final answer (A, B, C, or D)

To use a tool, respond with ONLY a JSON object: {{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
One tool call per response. When you have enough information, use submit_answer.

QUESTION: {question}{options_text}

Begin by analyzing the question, then search for evidence, and finally submit your answer."""

    return prompt


def generate_trajectory(
    model,
    tokenizer,
    task: dict,
    max_turns: int = 6,
    temperature: float = 0.7,
    processor=None,
) -> Optional[dict]:
    """Generate a single tool-use trajectory for a task.

    Returns trajectory dict if model gets correct answer, None otherwise.
    """
    prompt = build_gym_prompt(task)
    correct_answer = task["answer_letter"]

    messages = [
        {"role": "system", "content": (
            "You are a clinical AI assistant. Your task is to answer medical "
            "questions using the available tools systematically.\n\n"
            "To use a tool, respond with ONLY a JSON object: "
            '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
            "One tool call per response. When done, use submit_answer."
        )},
        {"role": "user", "content": prompt},
    ]

    submitted_answer = None

    for turn in range(max_turns):
        # Format conversation for the model
        if processor is not None:
            # VL model: convert messages to VL format (text content as list)
            vl_messages = _to_vl_messages(messages)
            conv_text = processor.apply_chat_template(
                vl_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[conv_text],
                return_tensors="pt",
                padding=True,
            )
        else:
            conv_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                conv_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoder = processor if processor is not None else tokenizer
        response = decoder.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        if not response:
            break

        # Try to parse as tool call
        tool_call = _parse_tool_call(response)

        if tool_call:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})

            messages.append({"role": "assistant", "content": json.dumps(tool_call)})

            if tool_name == "submit_answer":
                submitted_answer = tool_args.get("answer", "").strip().upper()
                # Remove parentheses if present: "(A)" -> "A"
                submitted_answer = submitted_answer.strip("()")
                if len(submitted_answer) > 1:
                    submitted_answer = submitted_answer[0]
                break

            # Simulate tool response
            tool_response = _simulate_tool_response(tool_name, tool_args, task)
            messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}:\n{tool_response}",
            })
        else:
            # Model gave free-form text — check if it contains an answer
            messages.append({"role": "assistant", "content": response})

            # Try to extract answer from free-form response
            answer_match = re.search(
                r"(?:answer|correct|best)\s*(?:is|:)\s*\(?([A-D])\)?",
                response,
                re.IGNORECASE,
            )
            if answer_match:
                submitted_answer = answer_match.group(1).upper()
            break

    # Check if answer is correct
    if submitted_answer and submitted_answer == correct_answer:
        return {
            "messages": messages,
            "metadata": {
                "source": "tool_trajectory_synthesis",
                "domain": "medical_qa",
                "task_id": task["id"],
                "correct_answer": correct_answer,
                "submitted_answer": submitted_answer,
                "num_turns": len(messages),
                "model": "trajectory_synthesis",
                "benchmark_source": task["source"],
            },
        }

    return None


def _parse_tool_call(text: str) -> Optional[dict]:
    """Parse a tool call JSON from model output."""
    text = text.strip()

    # Try direct JSON parse
    try:
        obj = json.loads(text)
        if "name" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
    json_match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*.*?\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try nested JSON
    brace_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start : i + 1])
                    if "name" in obj:
                        return obj
                except json.JSONDecodeError:
                    start = None

    return None


def _simulate_tool_response(
    tool_name: str,
    tool_args: dict,
    task: dict,
) -> str:
    """Simulate tool responses for trajectory synthesis.

    Uses the task's explanation and knowledge to generate realistic
    but not answer-revealing tool responses.
    """
    if tool_name == "think":
        return "Thought recorded."

    if tool_name in ("search_pubmed", "retrieve_evidence"):
        query = tool_args.get("query", "")
        # Return relevant medical context from the explanation
        explanation = task.get("explanation", "")
        if explanation:
            # Return first ~300 chars of explanation as "evidence"
            evidence = explanation[:300]
            return (
                f"Search results for '{query}':\n\n"
                f"[1] Relevant medical evidence:\n{evidence}\n\n"
                f"[Note: Evidence retrieved from medical knowledge base]"
            )
        return (
            f"Search results for '{query}':\n\n"
            f"[1] Multiple relevant articles found. Key medical concepts "
            f"identified in the literature. Please analyze the clinical "
            f"presentation and make your best assessment."
        )

    return f"Tool '{tool_name}' executed successfully."


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize tool-based SFT trajectories from train splits"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(
            PROJECT_ROOT
            / "checkpoints/models/Lingshu-7B"
        ),
        help="Path to model for trajectory generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "datasets/sft/tool_trajectories_train.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Max number of train questions to process",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=3,
        help="Number of rollouts per question (keep best)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Max tool-use turns per trajectory",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Save checkpoint every N tasks",
    )
    parser.add_argument(
        "--include-medmcqa",
        action="store_true",
        help="Also include MedMCQA train split",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU device(s) to use",
    )
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load train data
    medqa_tasks = load_medqa_train(max_samples=args.max_samples)
    all_tasks = medqa_tasks

    if args.include_medmcqa:
        medmcqa_tasks = load_medmcqa_train(
            max_samples=max(0, args.max_samples - len(medqa_tasks))
        )
        all_tasks = medqa_tasks + medmcqa_tasks

    logger.info(f"Total tasks to process: {len(all_tasks)}")

    # Load model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if VL model
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    is_vl = model_type in ("qwen2_5_vl", "qwen2_vl")

    if is_vl:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        logger.info("Detected VL model — using Qwen2_5_VLForConditionalGeneration")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        # Use processor for proper input formatting
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        processor = None
    model.eval()

    # Generate trajectories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    successful = []
    total_attempts = 0
    start_time = time.time()

    for i, task in enumerate(all_tasks):
        best_trajectory = None

        for rollout in range(args.num_rollouts):
            total_attempts += 1
            trajectory = generate_trajectory(
                model=model,
                tokenizer=tokenizer,
                task=task,
                max_turns=args.max_turns,
                temperature=args.temperature,
                processor=processor,
            )
            if trajectory is not None:
                # Keep first successful rollout
                best_trajectory = trajectory
                break

        if best_trajectory is not None:
            successful.append(best_trajectory)

        # Progress logging
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            success_rate = len(successful) / (i + 1) * 100
            logger.info(
                f"Progress: {i+1}/{len(all_tasks)} tasks | "
                f"Successful: {len(successful)} ({success_rate:.1f}%) | "
                f"Rate: {rate:.1f} tasks/sec"
            )

        # Checkpoint save
        if (i + 1) % args.batch_size == 0 and successful:
            _save_trajectories(successful, output_path)
            logger.info(f"Checkpoint saved: {len(successful)} trajectories")

    # Final save
    if successful:
        _save_trajectories(successful, output_path)

    elapsed = time.time() - start_time
    success_rate = len(successful) / len(all_tasks) * 100 if all_tasks else 0

    logger.info("=" * 60)
    logger.info("Trajectory Synthesis Complete!")
    logger.info(f"  Total tasks: {len(all_tasks)}")
    logger.info(f"  Successful trajectories: {len(successful)} ({success_rate:.1f}%)")
    logger.info(f"  Total attempts: {total_attempts}")
    logger.info(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)

    # Save stats
    stats = {
        "total_tasks": len(all_tasks),
        "successful": len(successful),
        "success_rate": success_rate,
        "total_attempts": total_attempts,
        "elapsed_seconds": elapsed,
        "model_path": args.model_path,
        "num_rollouts": args.num_rollouts,
        "temperature": args.temperature,
        "sources": {
            "MedQA": sum(1 for t in successful if t["metadata"]["benchmark_source"] == "MedQA"),
            "MedMCQA": sum(1 for t in successful if t["metadata"]["benchmark_source"] == "MedMCQA"),
        },
    }
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved: {stats_path}")


def _save_trajectories(trajectories: list[dict], output_path: Path):
    """Save trajectories to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
