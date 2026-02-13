#!/usr/bin/env python3
"""Generate multi-domain SFT training data from baseline trajectories.

Extracts successful agent trajectories across all 5 domains and
converts them into SFT training format. Uses high-quality runs from
Qwen3-8B-Base (highest Action Score) as expert demonstrations.

Strategy:
  1. Extract successful trajectories (action_score >= threshold)
  2. Convert to chat-format SFT examples (system + multi-turn tool calls)
  3. Generate ideal QA trajectories from training tasks
  4. Balance across domains
  5. Add domain-specific system prompts for each domain

Output: datasets/sft/multidomain_sft.jsonl

Usage:
    python scripts/generate_multidomain_sft.py
    python scripts/generate_multidomain_sft.py --min-score 0.5 --max-per-domain 100
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioagents.data_pipeline.sft_generator import (
    qa_tasks_to_sft,
    save_sft_dataset,
)
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_DIR = PROJECT_ROOT / "logs" / "baseline"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "sft"

# ── Domain-specific system prompts (richer than sft_generator defaults) ──────

DOMAIN_SYSTEM_PROMPTS = {
    "clinical_diagnosis": (
        "You are a clinical AI assistant. Your task is to assess patients, "
        "review their medical history, vital signs, and laboratory results, "
        "perform differential diagnosis, check drug interactions, and develop "
        "treatment plans. Use the available tools systematically.\n\n"
        "To use a tool, respond with ONLY a JSON object: "
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "One tool call per response. When done, provide your clinical assessment as text."
    ),
    "medical_qa": (
        "You are a medical AI assistant that answers medical questions using "
        "evidence-based reasoning. Search for evidence, think through the "
        "options, and submit your answer with clear reasoning.\n\n"
        "To use a tool, respond with ONLY a JSON object: "
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "When ready, use submit_answer to submit your final answer."
    ),
    "visual_diagnosis": (
        "You are a medical imaging AI assistant. Analyze medical images "
        "(X-ray, CT, MRI, pathology, dermoscopy, fundoscopy) to identify "
        "abnormalities and provide diagnostic assessments.\n\n"
        "To use a tool, respond with ONLY a JSON object: "
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "When done, use submit_answer with your visual diagnosis."
    ),
    "drug_interaction": (
        "You are a clinical pharmacology AI assistant. Review patient medication "
        "profiles, identify drug-drug interactions, assess risk levels, and "
        "recommend safer alternatives when needed.\n\n"
        "To use a tool, respond with ONLY a JSON object: "
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "When done, use submit_answer with your interaction assessment."
    ),
    "ehr_management": (
        "You are an EHR clinical AI assistant. Review electronic health records, "
        "analyze lab trends, calculate clinical scores, assess discharge readiness, "
        "and provide comprehensive clinical assessments.\n\n"
        "To use a tool, respond with ONLY a JSON object: "
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "When done, use submit_answer with your clinical recommendation."
    ),
}


# ── Trajectory extraction ────────────────────────────────────────────────────

def find_all_task_files(model_filter: str = None) -> list[tuple[str, str, Path]]:
    """Find all task result JSON files from baseline runs.
    
    Returns:
        List of (model_name, domain, task_file_path)
    """
    results = []
    for run_dir in sorted(BASELINE_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        # Parse directory name: {model}_{domain}_{timestamp}
        parts = run_dir.name.split("_")
        # Find domain by matching known domain names
        domain = None
        model_parts = []
        for i, p in enumerate(parts):
            candidate = "_".join(parts[i:])
            for d in DOMAIN_SYSTEM_PROMPTS:
                if candidate.startswith(d + "_"):
                    domain = d
                    model_parts = parts[:i]
                    break
            if domain:
                break
        
        if not domain:
            continue
        
        model_name = "-".join(model_parts) if model_parts else parts[0]
        
        if model_filter and model_filter not in model_name:
            continue
        
        # Find task files
        for task_file in sorted(run_dir.glob("task_*.json")):
            results.append((model_name, domain, task_file))
    
    return results


def trajectory_to_sft_example(
    task_file: Path,
    domain: str,
    min_action_score: float = 0.5,
) -> dict | None:
    """Convert a single task trajectory into an SFT training example.
    
    Args:
        task_file: Path to task_*.json file
        domain: Domain name
        min_action_score: Minimum action score threshold
    
    Returns:
        SFT example dict or None if below threshold
    """
    with open(task_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    action_score = data.get("action_score", 0.0)
    if action_score < min_action_score:
        return None
    
    turns = data.get("turns", [])
    if not turns:
        return None
    
    # Build chat messages
    messages = [{"role": "system", "content": DOMAIN_SYSTEM_PROMPTS.get(domain, "")}]
    
    # We need to reconstruct the conversation.
    # turns[0] usually doesn't have the initial observation in our format,
    # but the raw_output on turn 0 is the model's first response to the task.
    # We need to add a user message for the task ticket.
    
    # Add task ticket as initial user message
    task_id = data.get("task_id", "")
    messages.append({
        "role": "user",
        "content": f"[Task: {task_id}] Please complete this clinical task using the available tools.",
    })
    
    valid_turns = 0
    for turn in turns:
        raw_output = turn.get("raw_output", "")
        parsed_tool_call = turn.get("parsed_tool_call")
        tool_response = turn.get("tool_response", "")
        is_final = turn.get("is_final_answer", False)
        
        if parsed_tool_call:
            # Assistant makes a tool call (clean JSON only)
            messages.append({
                "role": "assistant",
                "content": json.dumps(parsed_tool_call, ensure_ascii=False),
            })
            valid_turns += 1
            
            # Tool response from environment
            if tool_response:
                tool_name = parsed_tool_call.get("name", "tool")
                # Truncate very long tool responses for training
                resp_text = tool_response[:3000]
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_name}:\n{resp_text}",
                })
        elif is_final and raw_output:
            # Final clinical assessment as text
            messages.append({"role": "assistant", "content": raw_output})
            valid_turns += 1
    
    if valid_turns < 1:
        return None
    
    return {
        "messages": messages,
        "metadata": {
            "source": "trajectory",
            "domain": domain,
            "task_id": task_id,
            "model": data.get("model_name", ""),
            "action_score": action_score,
            "final_reward": data.get("final_reward", 0.0),
            "total_turns": data.get("total_turns", 0),
        },
    }


# ── QA ideal trajectory generation ──────────────────────────────────────────

def generate_qa_sft_from_tasks(split: str = "train") -> list[dict]:
    """Generate ideal QA trajectories from medical_qa training tasks.
    
    Uses the train split tasks to create ideal search→think→submit sequences.
    """
    tasks_path = PROJECT_ROOT / "data" / "domains" / "medical_qa" / "tasks.json"
    split_path = PROJECT_ROOT / "data" / "domains" / "medical_qa" / "split_tasks.json"
    
    # Load all tasks
    if not tasks_path.exists():
        logger.warning("No medical_qa tasks.json found")
        return []
    
    with open(tasks_path) as f:
        all_tasks = json.load(f)
    
    # Build task map
    task_map = {t["id"]: t for t in all_tasks}
    
    # Filter by split if split file exists
    if split_path.exists():
        with open(split_path) as f:
            split_data = json.load(f)
        task_ids = split_data.get(split, [])
        # task_ids can be strings (IDs) or dicts (full tasks)
        tasks = []
        for item in task_ids:
            if isinstance(item, str):
                if item in task_map:
                    tasks.append(task_map[item])
            elif isinstance(item, dict):
                tasks.append(item)
    else:
        tasks = all_tasks
    
    if not tasks:
        logger.warning(f"No tasks found for split '{split}'")
        return []
    
    return qa_tasks_to_sft(tasks, include_reasoning=True, domain="medical_qa")


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate multi-domain SFT data")
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="Min action score for trajectory inclusion (default: 0.5)")
    parser.add_argument("--max-per-domain", type=int, default=200,
                        help="Max examples per domain (default: 200)")
    parser.add_argument("--model-filter", default=None,
                        help="Only include trajectories from this model (default: all)")
    parser.add_argument("--include-qa-train", action="store_true", default=True,
                        help="Include synthetic QA training examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Output path (default: datasets/sft/multidomain_sft.jsonl)")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("\n" + "=" * 80)
    print("  BIOAgents Multi-Domain SFT Data Generation")
    print("=" * 80)
    
    # ── Step 1: Extract trajectories ─────────────────────────────────
    print("\n  Step 1: Extracting baseline trajectories...")
    
    task_files = find_all_task_files(model_filter=args.model_filter)
    print(f"  Found {len(task_files)} task result files")
    
    trajectory_examples = []
    domain_counts = Counter()
    model_counts = Counter()
    skipped = 0
    
    for model_name, domain, task_file in task_files:
        example = trajectory_to_sft_example(
            task_file, domain, min_action_score=args.min_score
        )
        if example:
            trajectory_examples.append(example)
            domain_counts[domain] += 1
            model_counts[model_name] += 1
        else:
            skipped += 1
    
    print(f"  Extracted: {len(trajectory_examples)} examples (skipped {skipped})")
    print(f"  By domain: {dict(domain_counts)}")
    print(f"  By model:  {dict(model_counts)}")
    
    # ── Step 2: Generate QA training examples ────────────────────────
    qa_examples = []
    if args.include_qa_train:
        print("\n  Step 2: Generating QA training examples...")
        qa_examples = generate_qa_sft_from_tasks(split="train")
        print(f"  Generated: {len(qa_examples)} QA examples")
    
    # ── Step 3: Balance and combine ──────────────────────────────────
    print("\n  Step 3: Balancing across domains...")
    
    # Group trajectory examples by domain
    by_domain = defaultdict(list)
    for ex in trajectory_examples:
        domain = ex["metadata"]["domain"]
        by_domain[domain].append(ex)
    
    # Add QA examples
    for ex in qa_examples:
        by_domain["medical_qa"].append(ex)
    
    # Cap per domain
    balanced = []
    for domain, examples in by_domain.items():
        if len(examples) > args.max_per_domain:
            # Prioritize high action-score trajectories
            examples.sort(
                key=lambda x: x["metadata"].get("action_score", 0), reverse=True
            )
            examples = examples[:args.max_per_domain]
        balanced.extend(examples)
    
    # Shuffle
    random.shuffle(balanced)
    
    print(f"  Total: {len(balanced)} examples")
    final_domain_counts = Counter(ex["metadata"]["domain"] for ex in balanced)
    for domain, count in sorted(final_domain_counts.items()):
        print(f"    {domain}: {count}")
    
    # ── Step 4: Compute dataset statistics ───────────────────────────
    print("\n  Step 4: Dataset statistics...")
    
    total_messages = sum(len(ex["messages"]) for ex in balanced)
    avg_messages = total_messages / max(len(balanced), 1)
    
    source_counts = Counter(ex["metadata"].get("source", "unknown") for ex in balanced)
    
    total_chars = sum(
        sum(len(m.get("content", "")) for m in ex["messages"])
        for ex in balanced
    )
    
    print(f"  Total examples: {len(balanced)}")
    print(f"  Avg messages per example: {avg_messages:.1f}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Sources: {dict(source_counts)}")
    
    # ── Step 5: Save ─────────────────────────────────────────────────
    output_path = args.output or str(OUTPUT_DIR / "multidomain_sft.jsonl")
    
    print(f"\n  Step 5: Saving to {output_path}...")
    save_sft_dataset(balanced, output_path, format="jsonl")
    
    # Also save a stats file
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_examples": len(balanced),
        "domain_counts": dict(final_domain_counts),
        "source_counts": dict(source_counts),
        "model_counts": dict(model_counts),
        "avg_messages_per_example": round(avg_messages, 1),
        "total_characters": total_chars,
        "min_action_score": args.min_score,
        "max_per_domain": args.max_per_domain,
        "model_filter": args.model_filter,
    }
    stats_path = Path(output_path).with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"  Stats: {stats_path}")
    print(f"\n  ✅ Multi-domain SFT dataset generated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
