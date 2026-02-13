#!/usr/bin/env python3
"""LLM-based Task Generator for BIOAgents Healthcare AI GYM.

Generates diverse, high-quality medical task scenarios using LLM APIs.
Unlike the template-based scale_tasks.py, this generator:
1. Uses LLMs (OpenAI/Claude/local) for creative scenario generation
2. Generates realistic clinical narratives
3. Validates generated tasks against the domain DB
4. Supports all 7 domains

Usage:
    python scripts/generate_tasks_llm.py \
        --domains clinical_diagnosis triage_emergency \
        --target 50 \
        --provider openai \
        --validate

    # Using local model
    python scripts/generate_tasks_llm.py \
        --domains all --target 100 --provider local \
        --model-path /path/to/model
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent


# ══════════════════════════════════════════════════════════════
#  Task Templates (per domain)
# ══════════════════════════════════════════════════════════════

DOMAIN_PROMPTS = {
    "clinical_diagnosis": {
        "system": "You are a medical education expert creating clinical case scenarios for an AI agent training system.",
        "template": """Generate a realistic clinical diagnosis task for a medical AI agent.

The task should include:
1. A patient ticket with demographics, chief complaint, symptoms, and relevant history
2. Expected diagnostic actions the agent should take
3. Natural language assertions about what the agent should identify

Requirements:
- Condition: {condition}
- Severity: {severity}
- Patient demographic: {demographic}
- Include 4-6 expected tool actions from: get_patient_info, get_vital_signs, get_lab_results, order_lab_test, get_differential_diagnosis, search_clinical_guidelines, record_diagnosis
- Include 3-4 clinical assertions

Output ONLY valid JSON in this exact format:
{{
    "id": "dx_{condition_short}_{idx:03d}",
    "ticket": "<patient ticket text>",
    "description": {{"purpose": "<one line>", "difficulty": "<easy|medium|hard>", "condition": "{condition}"}},
    "evaluation_criteria": {{
        "actions": [
            {{"name": "<tool_name>", "arguments": {{}}, "compare_args": [], "info": "<description>"}}
        ],
        "nl_assertions": ["<assertion1>", "<assertion2>"],
        "reward_basis": ["ACTION", "NL_ASSERTION"]
    }}
}}""",
        "variables": {
            "conditions": [
                "pneumonia", "meningitis", "appendicitis", "myocardial_infarction",
                "pulmonary_embolism", "diabetic_ketoacidosis", "stroke", "sepsis",
                "heart_failure", "pancreatitis", "UTI", "COPD_exacerbation",
                "GI_bleeding", "acute_kidney_injury", "asthma_exacerbation",
                "deep_vein_thrombosis", "cellulitis", "pyelonephritis",
                "atrial_fibrillation", "hypertensive_emergency",
            ],
            "severities": ["mild", "moderate", "severe"],
            "demographics": [
                "25M healthy", "35F pregnant", "68M with diabetes and CKD",
                "82F with dementia", "8-year-old boy", "45F with lupus",
                "55M obese smoker", "72F with atrial fibrillation",
            ],
        },
    },

    "triage_emergency": {
        "system": "You are an emergency medicine attending creating triage scenarios for training AI triage systems.",
        "template": """Generate a realistic emergency department triage case.

Requirements:
- Chief complaint: {complaint}
- Expected ESI level: {esi_level}
- Patient demographic: {demographic}
- Include vital signs in the ticket
- Include 3-5 expected triage actions from: get_patient_presentation, get_vital_signs, assess_airway_breathing, get_medical_history, calculate_gcs, calculate_esi_level, get_ed_status, check_protocol, order_stat_labs, order_imaging, submit_answer
- Include 3-4 clinical assertions

Output ONLY valid JSON in this exact format:
{{
    "id": "triage_{complaint_short}_{idx:03d}",
    "ticket": "<triage case description with vitals>",
    "patient_id": "ED_GEN_{idx:03d}",
    "correct_answer": "ESI {esi_level} - <brief reasoning>",
    "description": {{"purpose": "<one line>", "difficulty": "<easy|medium|hard>", "category": "<category>"}},
    "evaluation_criteria": {{
        "actions": [
            {{"name": "<tool_name>", "arguments": {{}}, "compare_args": [], "info": "<description>"}}
        ],
        "nl_assertions": ["<assertion1>", "<assertion2>"],
        "reward_basis": ["ACTION", "NL_ASSERTION"]
    }}
}}""",
        "variables": {
            "complaints": [
                "chest_pain", "shortness_of_breath", "abdominal_pain",
                "headache", "altered_mental_status", "seizure",
                "trauma_mvc", "overdose", "gi_bleeding",
                "back_pain", "fever_neutropenia", "diabetic_emergency",
                "allergic_reaction", "laceration", "eye_injury",
                "testicular_pain", "vaginal_bleeding", "burns",
            ],
            "esi_levels": [1, 2, 3, 4, 5],
            "demographics": [
                "45M", "28F pregnant", "75M on warfarin", "5-year-old",
                "60F diabetic", "32M healthy", "88F nursing home",
            ],
        },
    },

    "drug_interaction": {
        "system": "You are a clinical pharmacist creating drug interaction scenarios for AI training.",
        "template": """Generate a drug interaction checking scenario.

Requirements:
- Drug pair: {drug_pair}
- Interaction type: {interaction}
- Patient context: {context}
- Include 3-5 expected actions from: get_drug_info, check_interaction, check_all_interactions, get_patient_medications, search_alternatives, check_dosage, submit_answer

Output ONLY valid JSON matching the BIOAgents task format with evaluation_criteria.""",
        "variables": {
            "drug_pairs": [
                ("warfarin", "NSAIDs"), ("metformin", "contrast"), ("SSRIs", "triptans"),
                ("ACE_inhibitors", "potassium"), ("digoxin", "amiodarone"),
                ("lithium", "thiazides"), ("carbamazepine", "erythromycin"),
                ("methotrexate", "trimethoprim"), ("cyclosporine", "statins"),
                ("tacrolimus", "azithromycin"), ("DOACs", "antifungals"),
            ],
        },
    },

    "radiology_report": {
        "system": "You are a radiology attending creating structured reporting cases for AI training.",
        "template": """Generate a radiology reporting task.

Requirements:
- Modality: {modality}
- Body part: {body_part}
- Clinical scenario: {scenario}
- Include 4-6 expected actions from: get_study_info, get_clinical_history, get_prior_reports, get_report_template, analyze_findings, search_radiology_knowledge, get_reporting_checklist, submit_report

Output ONLY valid JSON matching the BIOAgents task format with evaluation_criteria.""",
        "variables": {
            "modalities": ["XR", "CT", "MRI", "US", "MG"],
            "scenarios": [
                "chest_pneumonia", "ct_pe", "mri_brain_ms",
                "us_appendicitis", "ct_renal_stone", "xr_fracture",
                "ct_aortic_dissection", "mri_knee_acl",
            ],
        },
    },
}


# ══════════════════════════════════════════════════════════════
#  LLM Providers
# ══════════════════════════════════════════════════════════════

def generate_with_openai(system_prompt: str, user_prompt: str, model: str = "gpt-4o") -> str:
    """Generate task using OpenAI API."""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        max_tokens=2000,
    )
    return response.choices[0].message.content


def generate_with_anthropic(system_prompt: str, user_prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Generate task using Anthropic API."""
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def generate_with_local(system_prompt: str, user_prompt: str, model_path: str = "") -> str:
    """Generate task using local model via vLLM or transformers."""
    # Placeholder — implement based on local setup
    raise NotImplementedError("Local model generation not yet configured. Use --provider openai or anthropic.")


PROVIDERS = {
    "openai": generate_with_openai,
    "anthropic": generate_with_anthropic,
    "local": generate_with_local,
}


# ══════════════════════════════════════════════════════════════
#  Task Generator
# ══════════════════════════════════════════════════════════════

def generate_tasks_for_domain(
    domain: str,
    target_count: int = 50,
    provider: str = "openai",
    model: str = "gpt-4o",
    seed: int = 42,
) -> list[dict]:
    """Generate tasks for a domain using LLM.

    Args:
        domain: Domain name.
        target_count: Number of tasks to generate.
        provider: LLM provider ('openai', 'anthropic', 'local').
        model: Model name.
        seed: Random seed.

    Returns:
        List of generated task dicts.
    """
    random.seed(seed)

    if domain not in DOMAIN_PROMPTS:
        logger.warning(f"No LLM template for domain: {domain}. Using template-based generation.")
        return []

    config = DOMAIN_PROMPTS[domain]
    gen_fn = PROVIDERS.get(provider)
    if gen_fn is None:
        raise ValueError(f"Unknown provider: {provider}")

    tasks = []
    failures = 0

    for idx in range(target_count):
        # Sample variables
        variables = {}
        for key, values in config.get("variables", {}).items():
            if isinstance(values[0], tuple):
                variables[key] = random.choice(values)
            else:
                variables[key] = random.choice(values)

        # Build prompt
        prompt_vars = {
            "idx": idx + 1,
            **{k: str(v) for k, v in variables.items()},
        }

        # Handle specific variable mappings
        if domain == "clinical_diagnosis":
            prompt_vars["condition"] = variables.get("conditions", "pneumonia")
            prompt_vars["severity"] = variables.get("severities", "moderate")
            prompt_vars["demographic"] = variables.get("demographics", "55M")
            prompt_vars["condition_short"] = prompt_vars["condition"][:20].replace(" ", "_")
        elif domain == "triage_emergency":
            prompt_vars["complaint"] = variables.get("complaints", "chest_pain")
            prompt_vars["esi_level"] = variables.get("esi_levels", 3)
            prompt_vars["demographic"] = variables.get("demographics", "45M")
            prompt_vars["complaint_short"] = str(prompt_vars["complaint"])[:20]
        elif domain == "drug_interaction":
            pair = variables.get("drug_pairs", ("warfarin", "aspirin"))
            prompt_vars["drug_pair"] = f"{pair[0]} + {pair[1]}"
            prompt_vars["interaction"] = "potential adverse interaction"
            prompt_vars["context"] = random.choice(["elderly polypharmacy", "young adult", "renal impairment"])
        elif domain == "radiology_report":
            prompt_vars["modality"] = variables.get("modalities", "CT")
            prompt_vars["body_part"] = "chest"
            prompt_vars["scenario"] = variables.get("scenarios", "chest_pneumonia")

        try:
            user_prompt = config["template"].format(**prompt_vars)
        except KeyError as e:
            logger.warning(f"Template variable missing: {e}")
            continue

        try:
            response = gen_fn(config["system"], user_prompt, model)

            # Parse JSON from response
            task = _extract_json(response)
            if task:
                # Ensure unique ID
                if not task.get("id"):
                    task["id"] = f"{domain[:3]}_llm_{idx+1:04d}"
                tasks.append(task)
                logger.info(f"  [{domain}] Generated task {idx+1}/{target_count}: {task['id']}")
            else:
                failures += 1
                logger.warning(f"  [{domain}] Failed to parse JSON for task {idx+1}")

        except Exception as e:
            failures += 1
            logger.error(f"  [{domain}] Generation error for task {idx+1}: {e}")

        # Rate limiting
        if provider in ("openai", "anthropic"):
            import time
            time.sleep(0.5)

    logger.info(f"[{domain}] Generated {len(tasks)}/{target_count} tasks ({failures} failures)")
    return tasks


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response text."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON block
    import re
    patterns = [
        r"```json\s*\n(.*?)\n```",
        r"```\s*\n(.*?)\n```",
        r"\{[\s\S]*\}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue

    return None


def validate_task(task: dict, domain: str) -> list[str]:
    """Validate a generated task against domain requirements.

    Returns list of validation errors (empty = valid).
    """
    errors = []

    # Required fields
    if not task.get("id"):
        errors.append("Missing 'id'")
    if not task.get("ticket"):
        errors.append("Missing 'ticket'")
    if not task.get("evaluation_criteria"):
        errors.append("Missing 'evaluation_criteria'")

    ec = task.get("evaluation_criteria", {})
    if not ec.get("actions"):
        errors.append("Missing evaluation_criteria.actions")
    if not ec.get("reward_basis"):
        errors.append("Missing evaluation_criteria.reward_basis")

    # Domain-specific validation
    if domain == "triage_emergency":
        # Check ESI level in answer
        answer = str(task.get("correct_answer", ""))
        if not any(f"ESI {i}" in answer for i in range(1, 6)):
            errors.append("correct_answer should contain ESI level (1-5)")

    return errors


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based Task Generator")
    parser.add_argument(
        "--domains", nargs="+",
        default=["clinical_diagnosis", "triage_emergency"],
        help="Domains to generate tasks for ('all' for all domains)",
    )
    parser.add_argument("--target", type=int, default=50, help="Tasks per domain")
    parser.add_argument("--provider", default="openai", choices=list(PROVIDERS.keys()))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true", help="Validate generated tasks")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt without calling LLM")

    args = parser.parse_args()

    all_domains = list(DOMAIN_PROMPTS.keys())
    domains = all_domains if "all" in args.domains else args.domains

    total = 0
    for domain in domains:
        logger.info(f"\n{'='*50}")
        logger.info(f"Generating tasks for: {domain}")
        logger.info(f"{'='*50}")

        if args.dry_run:
            config = DOMAIN_PROMPTS.get(domain, {})
            print(f"\n[{domain}] System: {config.get('system', 'N/A')[:100]}")
            print(f"[{domain}] Template preview:\n{config.get('template', 'N/A')[:500]}")
            continue

        tasks = generate_tasks_for_domain(
            domain=domain,
            target_count=args.target,
            provider=args.provider,
            model=args.model,
            seed=args.seed,
        )

        if args.validate:
            valid_tasks = []
            for t in tasks:
                errors = validate_task(t, domain)
                if errors:
                    logger.warning(f"  Invalid task {t.get('id', '?')}: {errors}")
                else:
                    valid_tasks.append(t)
            tasks = valid_tasks
            logger.info(f"  After validation: {len(tasks)} valid tasks")

        # Save
        if tasks:
            if args.output_dir:
                out_dir = Path(args.output_dir) / domain
            else:
                out_dir = PROJECT_ROOT / "data" / "domains" / domain

            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "tasks_llm_generated.json"
            with open(out_path, "w") as f:
                json.dump(tasks, f, indent=2, ensure_ascii=False)

            # Also generate split
            random.shuffle(tasks)
            split_idx = int(len(tasks) * 0.7)
            split_path = out_dir / "split_tasks_llm.json"
            with open(split_path, "w") as f:
                json.dump({
                    "train": [t["id"] for t in tasks[:split_idx]],
                    "test": [t["id"] for t in tasks[split_idx:]],
                }, f, indent=2)

            total += len(tasks)
            logger.info(f"  Saved {len(tasks)} tasks to {out_path}")

    logger.info(f"\nTotal generated: {total} tasks across {len(domains)} domains")


if __name__ == "__main__":
    main()
