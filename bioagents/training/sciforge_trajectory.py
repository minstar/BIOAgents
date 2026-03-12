"""SciForge-inspired Clinical Trajectory Synthesis via Dependency Graphs.

Inspired by SciAgentGYM's SciForge (2026) which models the tool action space
as a dependency graph to generate logic-aware training trajectories.

This module adapts the approach for healthcare:
1. Clinical Tool DAGs -- each medical domain has a dependency graph that
   encodes which tools must be called before others.
2. Topological Trajectory Generation -- given a task's expected_actions,
   generates valid tool call orderings via topological sort with randomised
   tie-breaking for diversity.
3. Multi-turn Trajectory Formatting -- produces complete conversations
   ready for SFT or GRPO training.

Usage::

    from bioagents.training.sciforge_trajectory import (
        ClinicalToolDAG, generate_trajectories,
    )
    dag = ClinicalToolDAG.for_domain("ehr_management")
    trajectories = generate_trajectories(tasks=tasks, dag=dag, num_variants=3)
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# ============================================================
# Clinical Tool Dependency Graphs (per domain)
# ============================================================

CLINICAL_TOOL_DAGS: Dict[str, Dict[str, List[str]]] = {
    "clinical_diagnosis": {
        "get_patient_info": [],
        "get_vital_signs": ["get_patient_info"],
        "get_lab_results": ["get_patient_info"],
        "get_clinical_notes": ["get_patient_info"],
        "order_lab_test": ["get_lab_results"],
        "get_differential_diagnosis": ["get_vital_signs", "get_lab_results"],
        "search": [], "search_guidelines": [], "search_evidence": [],
        "think": [],
        "record_diagnosis": ["get_differential_diagnosis"],
        "submit_answer": ["think"],
    },
    "ehr_management": {
        "get_patient_summary": [],
        "get_admission_info": ["get_patient_summary"],
        "get_admission_history": ["get_patient_summary"],
        "get_lab_results": ["get_patient_summary"],
        "get_lab_trend": ["get_lab_results"],
        "get_vital_signs": ["get_patient_summary"],
        "detect_vital_alerts": ["get_vital_signs"],
        "get_medication_orders": ["get_patient_summary"],
        "get_clinical_scores": ["get_vital_signs", "get_lab_results"],
        "get_procedures": ["get_patient_summary"],
        "get_discharge_summary": ["get_patient_summary"],
        "get_quality_indicators": ["get_patient_summary"],
        "lookup_icd_code": [], "search": [], "think": [],
        "submit_answer": ["think"],
    },
    "drug_interaction": {
        "get_drug_info": [], "get_patient_medications": [],
        "check_interaction": ["get_patient_medications"],
        "check_all_interactions": ["get_patient_medications"],
        "check_dosage": ["get_drug_info"],
        "search_alternatives": ["check_interaction"],
        "search": [], "search_guidelines": [], "think": [],
        "submit_answer": ["think"],
    },
    "triage_emergency": {
        "get_patient_presentation": [],
        "assess_airway_breathing": ["get_patient_presentation"],
        "get_vital_signs": ["get_patient_presentation"],
        "get_medical_history": ["get_patient_presentation"],
        "calculate_gcs": ["get_patient_presentation"],
        "calculate_esi_level": ["get_vital_signs", "assess_airway_breathing"],
        "get_ed_status": [], "check_protocol": [],
        "order_stat_labs": ["calculate_esi_level"],
        "order_imaging": ["calculate_esi_level"],
        "search": [], "think": [],
        "submit_answer": ["calculate_esi_level", "think"],
    },
    "medical_qa": {
        "search": [], "search_pubmed": [], "search_medical_wiki": [],
        "search_evidence": [], "search_guidelines": [],
        "browse": ["search"],
        "browse_article": ["search_pubmed"],
        "browse_wiki_entry": ["search_medical_wiki"],
        "analyze_answer_options": [], "think": [],
        "submit_answer": ["think"],
    },
    "psychiatry": {
        "get_patient_presentation": [],
        "get_psychiatric_history": ["get_patient_presentation"],
        "perform_mental_status_exam": ["get_patient_presentation"],
        "administer_phq9": ["get_patient_presentation"],
        "administer_gad7": ["get_patient_presentation"],
        "assess_suicide_risk": ["get_patient_presentation"],
        "screen_substance_use": ["get_patient_presentation"],
        "administer_mmse": ["get_patient_presentation"],
        "get_current_medications": ["get_patient_presentation"],
        "search": [], "think": [],
        "submit_answer": ["think"],
    },
    "obstetrics": {
        "get_patient_presentation": [],
        "get_obstetric_history": ["get_patient_presentation"],
        "get_prenatal_labs": ["get_patient_presentation"],
        "assess_fetal_status": ["get_patient_presentation"],
        "assess_labor_progress": ["get_patient_presentation"],
        "calculate_bishop_score": ["assess_labor_progress"],
        "get_biophysical_profile": ["assess_fetal_status"],
        "check_medication_safety": [],
        "get_risk_assessment": ["get_obstetric_history", "get_prenatal_labs"],
        "check_ob_protocol": [], "search": [], "think": [],
        "submit_answer": ["think"],
    },
    "visual_diagnosis": {
        "analyze_medical_image": [],
        "get_image_report": ["analyze_medical_image"],
        "get_patient_context": [],
        "search_similar_cases": ["analyze_medical_image"],
        "compare_with_prior": ["analyze_medical_image"],
        "search": [], "think": [],
        "record_visual_diagnosis": ["think"],
        "submit_answer": ["think"],
    },
    "radiology_report": {
        "get_study_info": [], "get_clinical_history": [],
        "get_prior_reports": ["get_study_info"],
        "analyze_findings": ["get_study_info", "get_clinical_history"],
        "get_report_template": [], "get_reporting_checklist": [],
        "search": [], "think": [],
        "submit_report": ["analyze_findings", "think"],
        "submit_answer": ["think"],
    },
}


@dataclass
class ClinicalToolDAG:
    """Dependency graph for clinical tool calling in a specific domain."""
    domain: str
    edges: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def for_domain(cls, domain: str) -> "ClinicalToolDAG":
        edges = CLINICAL_TOOL_DAGS.get(domain, {})
        if not edges:
            logger.warning(f"No DAG for domain '{domain}', using empty graph")
        return cls(domain=domain, edges=dict(edges))

    def get_prerequisites(self, tool_name: str) -> List[str]:
        return list(self.edges.get(tool_name, []))

    def topological_sort(
        self, required_tools: List[str], seed: Optional[int] = None,
    ) -> List[str]:
        """Kahn's algorithm with randomised tie-breaking."""
        rng = random.Random(seed)
        required = set(required_tools)
        to_process = list(required)
        while to_process:
            tool = to_process.pop()
            for prereq in self.edges.get(tool, []):
                if prereq not in required:
                    required.add(prereq)
                    to_process.append(prereq)

        in_degree: Dict[str, int] = {t: 0 for t in required}
        adj: Dict[str, List[str]] = defaultdict(list)
        for tool in required:
            for prereq in self.edges.get(tool, []):
                if prereq in required:
                    adj[prereq].append(tool)
                    in_degree[tool] += 1

        queue = [t for t in required if in_degree[t] == 0]
        rng.shuffle(queue)
        ordered: List[str] = []
        while queue:
            node = queue.pop(0)
            ordered.append(node)
            for nb in adj[node]:
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)
            rng.shuffle(queue)

        if len(ordered) != len(required):
            logger.warning(f"Cycle in DAG for {self.domain}, fallback")
            return list(required_tools)
        return ordered

    def validate_ordering(self, ordering: List[str]) -> bool:
        seen: set = set()
        for tool in ordering:
            for prereq in self.edges.get(tool, []):
                if prereq not in seen and prereq in set(ordering):
                    return False
            seen.add(tool)
        return True


# ============================================================
# Trajectory Generation
# ============================================================

@dataclass
class SyntheticTurn:
    role: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    content: Optional[str] = None


@dataclass
class SyntheticTrajectory:
    task_id: str
    domain: str
    system_prompt: str
    user_message: str
    turns: List[SyntheticTurn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_messages(self) -> List[Dict[str, str]]:
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_message},
        ]
        for t in self.turns:
            if t.role == "assistant" and t.tool_name:
                call = json.dumps(
                    {"name": t.tool_name, "arguments": t.tool_args or {}},
                    ensure_ascii=False,
                )
                msgs.append({"role": "assistant", "content": call})
            elif t.role == "tool":
                msgs.append({"role": "tool", "content": t.content or ""})
            elif t.role == "assistant":
                msgs.append({"role": "assistant", "content": t.content or ""})
        return msgs


def _extract_tool_list(task: dict) -> List[Tuple[str, dict]]:
    actions: List[Tuple[str, dict]] = []
    for key in ("expected_actions", "evaluation_criteria"):
        src = task.get(key)
        if src is None:
            continue
        if isinstance(src, dict):
            src = src.get("actions", [])
        if isinstance(src, list):
            for a in src:
                if isinstance(a, dict):
                    name = a.get("tool") or a.get("name", "")
                    args = a.get("args") or a.get("arguments", {})
                    if name:
                        actions.append((name, args))
    return actions


def generate_trajectories(
    tasks: List[dict],
    dag: ClinicalToolDAG,
    num_variants: int = 3,
    system_prompt: str = "",
    include_think: bool = True,
    include_submit: bool = True,
    seed: int = 42,
) -> List[SyntheticTrajectory]:
    """Generate SciForge-style trajectories from tasks using the DAG."""
    all_trajectories: List[SyntheticTrajectory] = []

    for task in tasks:
        task_id = task.get("id", "unknown")
        ticket = task.get("ticket", "")
        tool_actions = _extract_tool_list(task)
        if not tool_actions:
            continue

        tool_names = [name for name, _ in tool_actions]
        tool_args_map = {name: args for name, args in tool_actions}

        if include_think and "think" not in tool_names:
            tool_names.append("think")
            tool_args_map["think"] = {}
        if include_submit and "submit_answer" not in tool_names:
            tool_names.append("submit_answer")
            tool_args_map["submit_answer"] = {}

        for vi in range(num_variants):
            vs = seed + hash(task_id) + vi
            ordering = dag.topological_sort(tool_names, seed=vs)
            turns: List[SyntheticTurn] = []
            for tn in ordering:
                args = tool_args_map.get(tn, {})
                turns.append(SyntheticTurn(
                    role="assistant", tool_name=tn, tool_args=args,
                ))
                turns.append(SyntheticTurn(
                    role="tool",
                    content=f"[{tn} result -- fill via environment replay]",
                ))
            traj = SyntheticTrajectory(
                task_id=f"{task_id}_v{vi}",
                domain=dag.domain,
                system_prompt=(
                    system_prompt
                    or f"You are a medical AI in the {dag.domain} domain."
                ),
                user_message=ticket,
                turns=turns,
                metadata={
                    "variant": vi, "ordering": ordering,
                    "dag_valid": dag.validate_ordering(ordering),
                    "source": "sciforge_synthesis",
                },
            )
            all_trajectories.append(traj)

    logger.info(
        f"Generated {len(all_trajectories)} trajectories "
        f"for {len(tasks)} tasks in {dag.domain}"
    )
    return all_trajectories


def trajectories_to_jsonl(
    trajectories: List[SyntheticTrajectory], output_path: str,
) -> int:
    """Save trajectories as JSONL for SFT training."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            record = {
                "task_id": traj.task_id,
                "domain": traj.domain,
                "messages": traj.to_messages(),
                "metadata": traj.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    logger.info(f"Wrote {count} trajectories to {output_path}")
    return count
