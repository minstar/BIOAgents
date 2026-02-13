"""Cross-domain Clinical Pathway environment.

This environment orchestrates tools from multiple domains to simulate
realistic multi-phase clinical workflows.
"""

import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "domains" / "cross_domain"


def get_environment(task: dict, **kwargs) -> dict:
    """Build a cross-domain environment for the given task.

    Since cross-domain tasks reference tools from their phase's domain,
    this function delegates tool setup to the appropriate domain environment.

    Args:
        task: Task dict with 'domain' field indicating which domain's tools to use

    Returns:
        Environment dict with tools, db, and policy
    """
    phase_domain = task.get("domain", "clinical_diagnosis")

    # Try to load the phase domain's environment
    try:
        import importlib
        mod = importlib.import_module(f"bioagents.domains.{phase_domain}.environment")
        base_env = mod.get_environment(task, **kwargs)
    except Exception as e:
        logger.warning(f"Could not load domain '{phase_domain}' environment: {e}")
        base_env = {"tools": [], "db": {}, "policy": ""}

    # Inject cross-domain context into the policy
    patient_data = task.get("patient_data", {})
    phase_info = task.get("description", {})

    cross_domain_header = (
        f"=== CROSS-DOMAIN CLINICAL PATHWAY ===\n"
        f"Pathway: {phase_info.get('pathway', 'N/A')}\n"
        f"Current Phase: {phase_info.get('phase', 'N/A')}\n"
        f"Difficulty: {phase_info.get('difficulty', 'N/A')}\n"
        f"Time-critical: {phase_info.get('time_critical', False)}\n"
        f"Safety-critical: {phase_info.get('safety_critical', False)}\n\n"
        f"Patient: {patient_data.get('name', 'N/A')}, "
        f"{patient_data.get('age', 'N/A')}{patient_data.get('sex', '')}\n"
        f"Allergies: {', '.join(patient_data.get('allergies', [])) or 'NKDA'}\n"
        f"Conditions: {', '.join(patient_data.get('conditions', [])) or 'None'}\n"
        f"Medications: {', '.join(patient_data.get('medications', [])) or 'None'}\n\n"
        f"IMPORTANT: This is a multi-phase clinical pathway. Maintain clinical "
        f"continuity from previous phases. Your decisions here will affect "
        f"subsequent phases.\n\n"
    )

    base_env["policy"] = cross_domain_header + base_env.get("policy", "")
    base_env["cross_domain"] = True
    base_env["patient_data"] = patient_data

    return base_env


def get_tasks(split: Optional[str] = None) -> list[dict]:
    """Load cross-domain tasks.

    Args:
        split: 'train', 'test', or None for all.

    Returns:
        List of task dicts.
    """
    tasks_path = DATA_DIR / "tasks.json"
    if not tasks_path.exists():
        logger.warning(f"No tasks.json found at {tasks_path}")
        return []

    with open(tasks_path) as f:
        all_tasks = json.load(f)

    if split is None:
        return all_tasks

    split_path = DATA_DIR / "split_tasks.json"
    if not split_path.exists():
        logger.warning(f"No split_tasks.json found, returning all tasks")
        return all_tasks

    with open(split_path) as f:
        splits = json.load(f)

    allowed_ids = set(splits.get(split, []))
    return [t for t in all_tasks if t["id"] in allowed_ids]
