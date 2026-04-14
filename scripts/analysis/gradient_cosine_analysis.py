#!/usr/bin/env python3
"""
Gradient Cosine Similarity Analysis Across Domains for GRPO Training

Analyzes gradient alignment/conflict between domains by computing per-domain
gradient vectors from a saved LoRA checkpoint. This is a post-hoc analysis
that loads a checkpoint, samples tasks per domain, computes gradients, and
measures pairwise cosine similarity.

Hypothesis: Gradients from text_qa vs. agentic domains (clinical_diagnosis,
drug_interaction, etc.) may conflict, explaining why agentic competence
improves but text QA performance does not transfer.

Usage:
    # Activate the project venv first:
    source /data/project/private/minstar/workspace/BIOAgents/.venv/bin/activate

    # Run with defaults (uses latest checkpoint version):
    python gradient_cosine_analysis.py

    # Specify checkpoint and GPU:
    python gradient_cosine_analysis.py \
        --checkpoint-dir /data/project/private/minstar/workspace/BIOAgents/checkpoints/full_4modality_grpo_qwen35_9b_fast/_vllm_lora/v24 \
        --num-samples-per-domain 10 \
        --gpu 0

    # The script outputs:
    #   1. 8x8 cosine similarity matrix (ASCII + saved to JSON)
    #   2. Gradient magnitude per domain
    #   3. Conflicting domain pairs (negative cosine similarity)
    #   4. JSON results at logs/analysis/gradient_cosine.json

Requirements:
    - PyTorch, transformers, peft (all in project .venv)
    - A saved LoRA checkpoint (produced by save_steps=25 during GRPO training)
    - Task data at data/domains/full_4modality_combined/tasks.json
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
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/data/project/private/minstar/workspace/BIOAgents")
DEFAULT_CHECKPOINT_BASE = PROJECT_ROOT / "checkpoints" / "full_4modality_grpo_qwen35_9b_fast" / "_vllm_lora"
DEFAULT_TASKS_PATH = PROJECT_ROOT / "data" / "domains" / "full_4modality_combined" / "tasks.json"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "analysis"
OUTPUT_PATH = OUTPUT_DIR / "gradient_cosine.json"

ALL_DOMAINS = [
    "text_qa",
    "multimodal_vqa",
    "clinical_diagnosis",
    "drug_interaction",
    "ehr_management",
    "triage_emergency",
    "psychiatry",
    "obstetrics",
]

# Domain-specific system prompts (must match training format in grpo_trainer.py)
DOMAIN_SYSTEM_PROMPTS = {
    "text_qa": (
        "You are a medical AI assistant that answers medical questions using "
        "evidence-based reasoning. Use tools to search for evidence, then "
        "submit your answer with clear reasoning.\n\n"
        "Available tools: search_pubmed, browse_article, search_medical_wiki, "
        "browse_wiki_entry, retrieve_evidence, analyze_answer_options, think, submit_answer.\n\n"
        'To call a tool, respond with JSON: {"name": "tool_name", "arguments": {...}}\n'
        "When ready, use submit_answer to provide your final answer."
    ),
    "drug_interaction": (
        "You are a clinical pharmacology AI assistant specializing in drug-drug "
        "interaction assessment. Review medication profiles, check interactions, "
        "and provide management recommendations.\n\n"
        "Available tools: get_patient_medications, get_drug_info, check_interaction, "
        "check_all_interactions, search_alternatives, check_dosage, "
        "search_drugs_by_class, think, submit_answer.\n\n"
        'To call a tool, respond with JSON: {"name": "tool_name", "arguments": {...}}\n'
        "When done, use submit_answer to provide your recommendation."
    ),
    "multimodal_vqa": (
        "You are a medical AI assistant specializing in visual diagnosis. "
        "Analyze medical images, interpret reports, and answer visual questions.\n\n"
        "Available tools: get_image_metadata, get_image_report, analyze_image, "
        "compare_images, search_similar_cases, answer_visual_question, think.\n\n"
        'To call a tool, respond with JSON: {"name": "tool_name", "arguments": {...}}'
    ),
    "clinical_diagnosis": (
        "You are a medical AI assistant for clinical diagnosis. Use tools to "
        "review patient records, order tests, and make clinical recommendations.\n\n"
        'To call a tool, respond with JSON: {"name": "tool_name", "arguments": {...}}'
    ),
}

DEFAULT_SYSTEM_PROMPT = "You are a medical AI assistant."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(base_dir: Path) -> Optional[Path]:
    """Find the latest versioned checkpoint directory (v0, v1, ..., vN)."""
    if not base_dir.exists():
        return None
    versions = []
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.startswith("v"):
            match = re.match(r"v(\d+)", d.name)
            if match:
                versions.append((int(match.group(1)), d))
    if not versions:
        return None
    versions.sort(key=lambda x: x[0], reverse=True)
    return versions[0][1]


def verify_checkpoint(checkpoint_dir: Path) -> bool:
    """Verify that the checkpoint directory contains required files."""
    required = ["adapter_config.json", "adapter_model.safetensors"]
    for f in required:
        if not (checkpoint_dir / f).exists():
            return False
    return True


def load_tasks(tasks_path: Path) -> dict[str, list[dict]]:
    """Load tasks grouped by _source_domain."""
    with open(tasks_path) as f:
        all_tasks = json.load(f)
    grouped: dict[str, list[dict]] = {}
    for task in all_tasks:
        domain = task.get("_source_domain", "unknown")
        grouped.setdefault(domain, []).append(task)
    return grouped


def get_system_prompt(domain: str) -> str:
    """Get the system prompt for a domain, matching training format."""
    return DOMAIN_SYSTEM_PROMPTS.get(domain, DEFAULT_SYSTEM_PROMPT)


def build_prompt_messages(task: dict, domain: str) -> list[dict]:
    """Build chat-format messages from a task, matching grpo_trainer.py format."""
    ticket = task.get("ticket", task.get("question", ""))
    system_msg = get_system_prompt(domain)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": ticket},
    ]


def compute_domain_gradient(
    model,
    tokenizer,
    tasks: list[dict],
    domain: str,
    device: torch.device,
    max_length: int = 512,
) -> Optional[torch.Tensor]:
    """Compute the average gradient vector over LoRA parameters for a domain batch.

    Returns a flattened 1-D gradient vector (on CPU) or None if no valid samples.
    """
    model.zero_grad()
    valid_count = 0

    for task in tasks:
        messages = build_prompt_messages(task, domain)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Skip degenerate inputs
        if input_ids.shape[1] < 2:
            continue

        # Causal LM loss: predict next token
        labels = input_ids.clone()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        if loss is None or torch.isnan(loss):
            continue

        loss.backward()
        valid_count += 1

    if valid_count == 0:
        return None

    # Extract and flatten LoRA gradients only
    grad_parts = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.grad is not None:
            grad_parts.append(param.grad.detach().flatten().cpu())

    if not grad_parts:
        return None

    grad_vector = torch.cat(grad_parts) / valid_count
    return grad_vector


def compute_cosine_matrix(
    grad_vectors: dict[str, torch.Tensor],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Compute pairwise cosine similarity and per-domain gradient magnitudes."""
    domains = sorted(grad_vectors.keys())
    cosine_matrix: dict[str, dict[str, float]] = {}
    magnitudes: dict[str, float] = {}

    for d in domains:
        magnitudes[d] = float(torch.norm(grad_vectors[d], p=2).item())

    for d1 in domains:
        cosine_matrix[d1] = {}
        for d2 in domains:
            sim = F.cosine_similarity(
                grad_vectors[d1].unsqueeze(0),
                grad_vectors[d2].unsqueeze(0),
            ).item()
            cosine_matrix[d1][d2] = round(sim, 4)

    return cosine_matrix, magnitudes


def print_ascii_matrix(
    cosine_matrix: dict[str, dict[str, float]],
    domains: list[str],
) -> str:
    """Format cosine similarity matrix as an aligned ASCII table."""
    # Abbreviate domain names for display
    abbrevs = {
        "text_qa": "TxtQA",
        "multimodal_vqa": "MVQA",
        "clinical_diagnosis": "ClinDx",
        "drug_interaction": "DrugIx",
        "ehr_management": "EHR",
        "triage_emergency": "Triage",
        "psychiatry": "Psych",
        "obstetrics": "OB",
    }
    col_width = 8
    header = " " * 10 + "".join(abbrevs.get(d, d[:6]).rjust(col_width) for d in domains)
    lines = [header, "-" * len(header)]

    for d1 in domains:
        row_label = abbrevs.get(d1, d1[:6]).ljust(10)
        row_values = ""
        for d2 in domains:
            val = cosine_matrix[d1][d2]
            row_values += f"{val:+.4f}".rjust(col_width)
        lines.append(row_label + row_values)

    table = "\n".join(lines)
    return table


def find_conflicts(
    cosine_matrix: dict[str, dict[str, float]],
    domains: list[str],
) -> list[dict]:
    """Identify domain pairs with negative cosine similarity (conflicting gradients)."""
    conflicts = []
    seen = set()
    for d1 in domains:
        for d2 in domains:
            if d1 >= d2:
                continue
            pair_key = (d1, d2)
            if pair_key in seen:
                continue
            seen.add(pair_key)
            sim = cosine_matrix[d1][d2]
            if sim < 0:
                conflicts.append({
                    "domain_a": d1,
                    "domain_b": d2,
                    "cosine_similarity": sim,
                })
    conflicts.sort(key=lambda x: x["cosine_similarity"])
    return conflicts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gradient cosine similarity analysis across GRPO training domains"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Path to a specific LoRA checkpoint directory (e.g., .../v24). "
            "If not provided, the latest version under _vllm_lora/ is used."
        ),
    )
    parser.add_argument(
        "--num-samples-per-domain",
        type=int,
        default=5,
        help="Number of task samples per domain (default: 5)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (default: 0)",
    )
    parser.add_argument(
        "--tasks-path",
        type=str,
        default=str(DEFAULT_TASKS_PATH),
        help="Path to tasks.json",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length for input truncation (default: 512)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PATH),
        help="Output JSON path",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Resolve checkpoint
    # -----------------------------------------------------------------------
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = find_latest_checkpoint(DEFAULT_CHECKPOINT_BASE)
        if checkpoint_dir is None:
            print("ERROR: No checkpoint found under", DEFAULT_CHECKPOINT_BASE)
            print()
            print("This script requires a LoRA checkpoint produced by GRPO training.")
            print("Checkpoints are saved every save_steps=25 training steps.")
            print()
            print("Expected directory structure:")
            print(f"  {DEFAULT_CHECKPOINT_BASE}/v0/")
            print(f"  {DEFAULT_CHECKPOINT_BASE}/v1/")
            print("  ...")
            print()
            print("Each version directory should contain:")
            print("  adapter_config.json")
            print("  adapter_model.safetensors")
            sys.exit(1)

    if not verify_checkpoint(checkpoint_dir):
        print(f"ERROR: Checkpoint at {checkpoint_dir} is missing required files.")
        print("Expected: adapter_config.json, adapter_model.safetensors")
        sys.exit(1)

    print(f"Using checkpoint: {checkpoint_dir}")

    # -----------------------------------------------------------------------
    # 2. Load adapter config to find base model
    # -----------------------------------------------------------------------
    with open(checkpoint_dir / "adapter_config.json") as f:
        adapter_config = json.load(f)
    base_model_path = adapter_config["base_model_name_or_path"]
    print(f"Base model: {base_model_path}")

    # -----------------------------------------------------------------------
    # 3. Load tasks
    # -----------------------------------------------------------------------
    tasks_path = Path(args.tasks_path)
    if not tasks_path.exists():
        print(f"ERROR: Tasks file not found at {tasks_path}")
        sys.exit(1)

    grouped_tasks = load_tasks(tasks_path)
    available_domains = sorted(grouped_tasks.keys())
    print(f"Available domains: {available_domains}")
    print(f"Sampling {args.num_samples_per_domain} tasks per domain")

    # Filter to domains that actually exist in the data
    domains_to_analyze = [d for d in ALL_DOMAINS if d in grouped_tasks]
    if not domains_to_analyze:
        print("ERROR: No matching domains found in tasks data.")
        sys.exit(1)

    # Sample tasks per domain (deterministic seed for reproducibility)
    import random
    rng = random.Random(42)
    sampled: dict[str, list[dict]] = {}
    for domain in domains_to_analyze:
        pool = grouped_tasks[domain]
        n = min(args.num_samples_per_domain, len(pool))
        sampled[domain] = rng.sample(pool, n)
        print(f"  {domain}: {n} samples (from {len(pool)} total)")

    # -----------------------------------------------------------------------
    # 4. Load model + LoRA
    # -----------------------------------------------------------------------
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (this may take a minute)...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
        attn_implementation="flash_attention_2",
    )
    print(f"  Base model loaded in {time.time() - t0:.1f}s")

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    model.train()  # Need gradients

    # Freeze base model, only LoRA params require grad
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    lora_param_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  LoRA trainable parameters: {lora_param_count:,}")

    # -----------------------------------------------------------------------
    # 5. Compute per-domain gradient vectors
    # -----------------------------------------------------------------------
    print("\nComputing per-domain gradients...")
    grad_vectors: dict[str, torch.Tensor] = {}
    for domain in domains_to_analyze:
        t0 = time.time()
        print(f"  [{domain}] computing gradients over {len(sampled[domain])} samples...", end="", flush=True)
        gv = compute_domain_gradient(
            model, tokenizer, sampled[domain], domain, device, args.max_length
        )
        if gv is not None:
            grad_vectors[domain] = gv
            elapsed = time.time() - t0
            print(f" done ({elapsed:.1f}s, dim={gv.shape[0]:,})")
        else:
            print(" SKIPPED (no valid gradients)")

    if len(grad_vectors) < 2:
        print("ERROR: Need at least 2 domains with valid gradients for comparison.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 6. Compute cosine similarity matrix
    # -----------------------------------------------------------------------
    analyzed_domains = sorted(grad_vectors.keys())
    cosine_matrix, magnitudes = compute_cosine_matrix(grad_vectors)

    # -----------------------------------------------------------------------
    # 7. Output results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GRADIENT COSINE SIMILARITY MATRIX")
    print("=" * 80)
    table = print_ascii_matrix(cosine_matrix, analyzed_domains)
    print(table)

    print("\n" + "-" * 80)
    print("GRADIENT MAGNITUDES (L2 norm)")
    print("-" * 80)
    for d in analyzed_domains:
        print(f"  {d:25s}  {magnitudes[d]:.6f}")

    conflicts = find_conflicts(cosine_matrix, analyzed_domains)
    print("\n" + "-" * 80)
    if conflicts:
        print(f"CONFLICTING DOMAIN PAIRS ({len(conflicts)} found)")
        print("-" * 80)
        for c in conflicts:
            print(f"  {c['domain_a']:25s} vs {c['domain_b']:25s}  cos={c['cosine_similarity']:+.4f}")
    else:
        print("No conflicting domain pairs (all cosine similarities >= 0)")
        print("-" * 80)

    # Highlight text_qa vs agentic domains specifically
    agentic_domains = [
        d for d in analyzed_domains
        if d not in ("text_qa", "multimodal_vqa")
    ]
    print("\n" + "-" * 80)
    print("TEXT_QA vs AGENTIC DOMAINS (key hypothesis)")
    print("-" * 80)
    if "text_qa" in cosine_matrix:
        for d in agentic_domains:
            if d in cosine_matrix["text_qa"]:
                sim = cosine_matrix["text_qa"][d]
                label = "CONFLICT" if sim < 0 else "ALIGNED" if sim > 0.3 else "WEAK"
                print(f"  text_qa vs {d:25s}  cos={sim:+.4f}  [{label}]")
    else:
        print("  text_qa not in analyzed domains")

    # -----------------------------------------------------------------------
    # 8. Save JSON
    # -----------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": str(checkpoint_dir),
        "base_model": base_model_path,
        "num_samples_per_domain": args.num_samples_per_domain,
        "max_length": args.max_length,
        "domains_analyzed": analyzed_domains,
        "lora_param_count": lora_param_count,
        "gradient_dim": grad_vectors[analyzed_domains[0]].shape[0],
        "cosine_similarity_matrix": cosine_matrix,
        "gradient_magnitudes": {k: round(v, 6) for k, v in magnitudes.items()},
        "conflicting_pairs": conflicts,
        "num_conflicts": len(conflicts),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
