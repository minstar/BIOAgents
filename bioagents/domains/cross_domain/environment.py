"""Cross-domain Clinical Pathway environment.

This environment orchestrates tools from multiple domains to simulate
realistic multi-phase clinical workflows.

Each cross-domain task has a ``domain`` field that specifies which
domain's tools should be loaded for that phase (e.g. ``triage_emergency``
for the triage phase).  At ``reset()`` time the GYM passes the selected
task so we can load the correct toolkit.
"""

import importlib
import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from bioagents.environment.environment import Environment

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "domains" / "cross_domain"
POLICY_PATH = DATA_DIR / "policy.md"


def _build_cross_domain_header(task: dict) -> str:
    """Build a context header from the cross-domain task metadata."""
    patient_data = task.get("patient_data", {})
    phase_info = task.get("description", {})
    return (
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


def get_environment(
    db=None,
    max_turns: int = 20,
    task: Optional[dict] = None,
    **kwargs,
) -> Environment:
    """Create a cross-domain environment.

    When *task* is supplied the function loads the phase-specific domain's
    toolkit (e.g. ``triage_emergency`` tools for the triage phase) and
    prepends a cross-domain patient-context header to the policy.

    When *task* is ``None`` (e.g. called from ``get_gym_stats``),
    a lightweight fallback environment with only the cross-domain
    policy is returned.

    Args:
        db: Unused — kept for interface compatibility.
        max_turns: Maximum interaction turns.
        task: The cross-domain task dict (has ``domain`` field).

    Returns:
        A fully configured :class:`Environment`.
    """
    # Load the cross-domain base policy
    policy_text = ""
    if POLICY_PATH.exists():
        with open(POLICY_PATH, "r", encoding="utf-8") as f:
            policy_text = f.read()

    # If no task provided, return a minimal environment (for stats / probing)
    if task is None:
        return Environment(
            domain_name="cross_domain",
            policy=policy_text,
            tools=None,
            max_turns=max_turns,
        )

    # ---- Task-aware path: load the phase domain's tools ----
    phase_domain = task.get("domain", "clinical_diagnosis")
    tools = None

    try:
        mod = importlib.import_module(
            f"bioagents.domains.{phase_domain}.environment"
        )
        phase_env: Environment = mod.get_environment(max_turns=max_turns)
        tools = phase_env.tools
        # Append phase domain's policy under the cross-domain header
        phase_policy = phase_env.policy or ""
    except Exception as e:
        logger.warning(
            f"Could not load phase domain '{phase_domain}' tools: {e}. "
            f"Cross-domain environment will have no tools."
        )
        phase_policy = ""

    # Compose final policy: cross-domain header + base policy + phase policy
    header = _build_cross_domain_header(task)
    full_policy = header + policy_text + "\n\n" + phase_policy

    return Environment(
        domain_name="cross_domain",
        policy=full_policy,
        tools=tools,
        max_turns=max_turns,
    )


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
