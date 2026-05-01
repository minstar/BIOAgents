#!/usr/bin/env python3
"""
HealthBench Professional evaluation with OpenAI LLM-judge rubric scoring.

Usage:
    # Text-only (direct generation, no tools):
    python scripts/eval_healthbench.py \
        --model_path checkpoints/models/Qwen3.5-9B \
        --output-dir results/healthbench/base_textonly \
        --max-new-tokens 2048

    # Multi-turn with AgentRunner:
    python scripts/eval_healthbench.py \
        --model_path checkpoints/models/Qwen3.5-9B \
        --output-dir results/healthbench/base_agentrunner \
        --mode multiturn --max-turns 10

    # Judge-only (score existing responses):
    python scripts/eval_healthbench.py \
        --judge-only results/healthbench/base_textonly/healthbench_responses.json \
        --output-dir results/healthbench/base_textonly
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

JUDGE_MODEL = "gpt-4.1-mini"
JUDGE_SYSTEM = (
    "You are an expert medical evaluator. You will be given a model's response "
    "to a medical question and a specific evaluation criterion. "
    "Determine whether the response satisfies the criterion. "
    "Answer with exactly YES or NO, followed by a brief justification."
)


def load_healthbench():
    """Load HealthBench Professional from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("openai/healthbench-professional", split="test")
    tasks = []
    for item in ds:
        messages = item["conversation"]["messages"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        question = user_msgs[-1]["content"] if user_msgs else ""

        tasks.append({
            "id": item["id"],
            "question": question,
            "conversation": messages,
            "physician_response": item.get("physician_response", ""),
            "rubric_items": item["rubric_items"],
            "use_case": item["use_case"],
            "type": item["type"],
            "difficulty": item["difficulty"],
            "specialty": item["specialty"],
        })

    logger.info(f"Loaded {len(tasks)} HealthBench Professional tasks")
    return tasks


def judge_rubric_item(response: str, criterion_text: str, client, model_name: str = JUDGE_MODEL) -> dict:
    """Judge a single rubric item using OpenAI."""
    user_prompt = (
        f"## Model Response\n{response}\n\n"
        f"## Criterion\n{criterion_text}\n\n"
        f"Does the model response satisfy this criterion? Answer YES or NO, "
        f"then briefly explain why."
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        verdict_text = completion.choices[0].message.content.strip()
        satisfied = verdict_text.upper().startswith("YES")
        return {
            "satisfied": satisfied,
            "verdict": verdict_text,
            "tokens_used": completion.usage.total_tokens,
        }
    except Exception as e:
        logger.warning(f"Judge API error: {e}")
        return {"satisfied": False, "verdict": f"ERROR: {e}", "tokens_used": 0}


def score_example(response: str, rubric_items: list, client, model_name: str = JUDGE_MODEL) -> dict:
    """Score a response against all rubric items."""
    total_positive = sum(item["points"] for item in rubric_items if item["points"] > 0)
    total_negative = sum(abs(item["points"]) for item in rubric_items if item["points"] < 0)

    earned = 0
    penalties = 0
    item_results = []
    total_tokens = 0

    for item in rubric_items:
        result = judge_rubric_item(response, item["criterion_text"], client, model_name)
        total_tokens += result["tokens_used"]

        if result["satisfied"]:
            if item["points"] > 0:
                earned += item["points"]
            else:
                # Negative criterion satisfied = penalty applies
                penalties += abs(item["points"])
        else:
            if item["points"] < 0:
                # Negative criterion NOT satisfied = no penalty (good)
                pass

        item_results.append({
            "criterion": item["criterion_text"],
            "points": item["points"],
            "satisfied": result["satisfied"],
            "verdict": result["verdict"],
        })

    # Normalize score: earned positive points / total possible positive points
    score = earned / total_positive if total_positive > 0 else 0.0
    # Penalty-adjusted score
    adjusted_score = (earned - penalties) / total_positive if total_positive > 0 else 0.0

    return {
        "score": score,
        "adjusted_score": max(0.0, adjusted_score),
        "earned": earned,
        "penalties": penalties,
        "total_positive": total_positive,
        "total_negative": total_negative,
        "item_results": item_results,
        "tokens_used": total_tokens,
    }


def generate_response_textonly(question: str, model, tokenizer, max_new_tokens: int = 2048) -> str:
    """Generate a text-only response (no tools)."""
    import torch

    system_prompt = (
        "You are a medical AI assistant. Provide a thorough, evidence-based "
        "response to the following medical question. Be comprehensive and "
        "include relevant clinical details, guidelines, and considerations."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def save_results(results: list, summary: dict, output_dir: Path, prefix: str = "healthbench"):
    """Save results and summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_path = output_dir / f"{prefix}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save summary
    summary_path = output_dir / f"{prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Summary saved to {summary_path}")


def save_partial(results: list, output_dir: Path):
    """Save partial results for resumability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "healthbench_partial.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_partial(output_dir: Path) -> list:
    """Load partial results if they exist."""
    path = output_dir / "healthbench_partial.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(description="HealthBench Professional Evaluation")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--mode", choices=["textonly", "multiturn"], default="textonly")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--judge-only", type=str, default=None,
                        help="Path to existing responses JSON (skip generation)")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from partial results")
    args = parser.parse_args()

    judge_model = args.judge_model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize OpenAI client
    import openai
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=os.environ.get("OPENAI_ORG_ID"),
        project=os.environ.get("OPENAI_PROJECT_ID"),
    )

    # Load tasks
    tasks = load_healthbench()
    if args.max_samples > 0:
        tasks = tasks[:args.max_samples]

    # Resume from partial
    existing_results = []
    start_idx = 0
    if args.resume:
        existing_results = load_partial(output_dir)
        start_idx = len(existing_results)
        logger.info(f"Resuming from {start_idx}/{len(tasks)}")

    if args.judge_only:
        # Load pre-generated responses
        with open(args.judge_only) as f:
            responses_data = json.load(f)
        responses = {r["id"]: r["response"] for r in responses_data}
        logger.info(f"Loaded {len(responses)} pre-generated responses")
    elif args.mode == "textonly":
        # Load model for text-only generation
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        logger.info("Model loaded successfully")
        responses = None
    else:
        logger.error("Multiturn mode not yet implemented — use textonly or judge-only")
        return

    # Run evaluation
    results = list(existing_results)
    total_tokens = 0
    t_start = time.time()

    for i, task in enumerate(tasks):
        if i < start_idx:
            continue

        # Generate or retrieve response
        if responses is not None:
            response = responses.get(task["id"], "")
        else:
            response = generate_response_textonly(
                task["question"], model, tokenizer, args.max_new_tokens
            )

        # Score with rubric
        score_result = score_example(response, task["rubric_items"], client, judge_model)
        total_tokens += score_result["tokens_used"]

        result = {
            "id": task["id"],
            "question": task["question"][:300],
            "response": response[:1000],
            "use_case": task["use_case"],
            "type": task["type"],
            "difficulty": task["difficulty"],
            "specialty": task["specialty"],
            "score": score_result["score"],
            "adjusted_score": score_result["adjusted_score"],
            "earned": score_result["earned"],
            "total_positive": score_result["total_positive"],
            "penalties": score_result["penalties"],
            "item_results": score_result["item_results"],
        }
        results.append(result)

        # Progress logging
        if (i + 1) % 10 == 0 or i == len(tasks) - 1:
            elapsed = time.time() - t_start
            rate = (i + 1 - start_idx) / (elapsed / 60) if elapsed > 0 else 0
            avg_score = sum(r["score"] for r in results) / len(results)
            avg_adj = sum(r["adjusted_score"] for r in results) / len(results)
            logger.info(
                f"  [{i+1}/{len(tasks)}] score={avg_score:.3f} adj={avg_adj:.3f} "
                f"rate={rate:.1f}/min tokens={total_tokens}"
            )
            save_partial(results, output_dir)

    # Compute summary
    scores = [r["score"] for r in results]
    adj_scores = [r["adjusted_score"] for r in results]

    summary = {
        "benchmark": "healthbench_professional",
        "judge_model": judge_model,
        "total": len(results),
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "avg_adjusted_score": sum(adj_scores) / len(adj_scores) if adj_scores else 0,
        "total_api_tokens": total_tokens,
        "breakdown": {},
    }

    # Breakdowns
    for key in ["use_case", "type", "difficulty", "specialty"]:
        groups = {}
        for r in results:
            val = r[key]
            if val not in groups:
                groups[val] = []
            groups[val].append(r["score"])
        summary["breakdown"][key] = {
            k: {"avg_score": sum(v) / len(v), "count": len(v)}
            for k, v in sorted(groups.items())
        }

    save_results(results, summary, output_dir)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"HealthBench Professional Results")
    logger.info(f"{'='*60}")
    logger.info(f"  Score:          {summary['avg_score']:.3f}")
    logger.info(f"  Adjusted Score: {summary['avg_adjusted_score']:.3f}")
    logger.info(f"  Total:          {summary['total']}")
    logger.info(f"  API Tokens:     {summary['total_api_tokens']}")
    logger.info(f"\n  By type:")
    for k, v in summary["breakdown"]["type"].items():
        logger.info(f"    {k}: {v['avg_score']:.3f} (n={v['count']})")
    logger.info(f"\n  By difficulty:")
    for k, v in summary["breakdown"]["difficulty"].items():
        logger.info(f"    {k}: {v['avg_score']:.3f} (n={v['count']})")
    logger.info(f"\n  By use_case:")
    for k, v in summary["breakdown"]["use_case"].items():
        logger.info(f"    {k}: {v['avg_score']:.3f} (n={v['count']})")


if __name__ == "__main__":
    main()
