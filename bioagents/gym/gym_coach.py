"""GymCoach — Autonomous Training Orchestrator for Healthcare AI GYM.

The GymCoach is the "brain" that makes 8B models conquer healthcare AI.
It runs a continuous loop:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    GymCoach Autonomous Loop                         │
    │                                                                     │
    │   ┌──────────┐    ┌───────────┐    ┌──────────────┐               │
    │   │ EVALUATE  │───►│ ANALYZE   │───►│ GENERATE     │               │
    │   │ all       │    │ errors &  │    │ targeted     │               │
    │   │ domains   │    │ weaknesses│    │ training data│               │
    │   └──────────┘    └───────────┘    └──────┬───────┘               │
    │        ▲                                   │                        │
    │        │           ┌───────────┐    ┌──────▼───────┐               │
    │        └───────────┤ TRACK     │◄───┤ TRAIN        │               │
    │                    │ progress  │    │ SFT / GRPO   │               │
    │                    │ & mastery │    │ on weaknesses │               │
    │                    └─────┬─────┘    └──────────────┘               │
    │                          │                                          │
    │                    ┌─────▼─────┐                                    │
    │                    │ EXPAND    │ ← All conquered? → New domains!    │
    │                    │ curriculum│                                     │
    │                    └───────────┘                                     │
    └─────────────────────────────────────────────────────────────────────┘

The GymCoach answers the question:
"Can a 7B model conquer all of healthcare AI?"

By making the model:
1. Play in the GYM (evaluate on all domains)
2. See where it fails (error analysis)
3. Practice its weaknesses (targeted data generation + training)
4. Track mastery (progress dashboard)
5. Level up (curriculum: easy → hard → cross-domain → safety → new domains)
6. Expand horizons (domain expansion when current challenges are conquered)

References:
- Agent Hospital (Li et al., 2024): Learn from treating 10K patients
- DoctorAgent-RL (2025): Multi-agent collaborative RL
- AgentClinic (2024): Multi-specialty evaluation
- Self-play loop (our existing): Trajectory collect → judge → filter → train
"""

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger


# ============================================================
# 1. Data Structures
# ============================================================

class MasteryLevel(Enum):
    """Domain mastery levels — when does a model "conquer" a domain?"""
    NOVICE = "novice"               # <30% action score
    BEGINNER = "beginner"           # 30-50%
    INTERMEDIATE = "intermediate"   # 50-70%
    ADVANCED = "advanced"           # 70-85%
    EXPERT = "expert"               # 85-95%
    MASTER = "master"               # >95% — CONQUERED!

    @classmethod
    def from_score(cls, score: float) -> "MasteryLevel":
        if score < 0.30:
            return cls.NOVICE
        elif score < 0.50:
            return cls.BEGINNER
        elif score < 0.70:
            return cls.INTERMEDIATE
        elif score < 0.85:
            return cls.ADVANCED
        elif score < 0.95:
            return cls.EXPERT
        else:
            return cls.MASTER


class CurriculumPhase(Enum):
    """Training curriculum phases."""
    PHASE_1_SINGLE_DOMAIN = "single_domain"       # Master individual domains
    PHASE_2_MULTI_DOMAIN = "multi_domain"          # Multi-domain proficiency
    PHASE_3_CROSS_DOMAIN = "cross_domain"          # Cross-domain pathways
    PHASE_4_SAFETY = "safety_hardening"            # Safety & adversarial robustness
    PHASE_5_EXPANSION = "domain_expansion"         # New domains beyond current scope


@dataclass
class FailureCase:
    """A specific failure case from evaluation."""
    task_id: str
    domain: str
    failure_type: str      # "wrong_tool", "wrong_reasoning", "safety_violation", etc.
    description: str
    severity: int          # 1-5
    expected_actions: list[str] = field(default_factory=list)
    actual_actions: list[str] = field(default_factory=list)
    missing_actions: list[str] = field(default_factory=list)
    error_category: str = ""   # "tool_selection", "parameter_error", "premature_stop", etc.


@dataclass
class DomainReport:
    """Detailed evaluation report for one domain."""
    domain: str
    action_score: float
    reward_score: float
    num_tasks: int
    num_passed: int
    num_failed: int
    mastery: MasteryLevel = MasteryLevel.NOVICE
    failure_cases: list[FailureCase] = field(default_factory=list)
    failure_distribution: dict = field(default_factory=dict)
    weakest_skills: list[str] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class CoachConfig:
    """Configuration for the GymCoach."""
    # Model
    model_name_or_path: str = ""
    backend: str = "transformers"
    
    # Domains to train on
    domains: list[str] = field(default_factory=lambda: [
        "clinical_diagnosis", "drug_interaction", "ehr_management",
        "medical_qa", "visual_diagnosis", "triage_emergency",
        "radiology_report", "cross_domain",
        "psychiatry", "obstetrics",
    ])
    
    # Training parameters
    max_iterations: int = 20         # Total coaching iterations
    eval_tasks_per_domain: int = 30  # Tasks per domain per eval round
    train_epochs_per_iter: int = 2
    training_method: str = "sft"     # "sft" or "grpo"
    learning_rate: float = 2e-5
    
    # Curriculum
    mastery_threshold: float = 0.90  # Score to consider domain "conquered"
    safety_threshold: float = 0.85   # Safety score to proceed to expansion
    min_domains_for_multidomain: int = 3  # Min conquered single-domains before multi
    
    # Data generation
    tasks_per_weakness: int = 10     # Tasks to generate per identified weakness
    max_generated_tasks: int = 100   # Max tasks to generate per iteration
    
    # Continuous mode
    continuous: bool = True          # True = never stop, auto-expand when conquered
    
    # Paths
    output_dir: str = "checkpoints/gym_coach"
    log_dir: str = "logs/gym_coach"
    generated_data_dir: str = "data/generated"


# ============================================================
# 2. Error Analyzer
# ============================================================

class ErrorAnalyzer:
    """Analyzes evaluation results to identify weaknesses and failure patterns.
    
    The ErrorAnalyzer answers: "WHY did the model fail?"
    
    Failure categories:
    1. TOOL_SELECTION — Called wrong tool
    2. PARAMETER_ERROR — Right tool, wrong arguments
    3. PREMATURE_STOP — Didn't gather enough info
    4. OVER_INVESTIGATION — Too many redundant tool calls
    5. REASONING_ERROR — Bad clinical reasoning despite correct data
    6. SAFETY_VIOLATION — Dangerous action or recommendation
    7. FORMAT_ERROR — Output format doesn't match expected
    8. KNOWLEDGE_GAP — Missing medical knowledge
    9. GUIDELINE_NONCOMPLIANCE — Didn't follow clinical guidelines
    """
    
    FAILURE_CATEGORIES = [
        "tool_selection",
        "parameter_error",
        "premature_stop",
        "over_investigation",
        "reasoning_error",
        "safety_violation",
        "format_error",
        "knowledge_gap",
        "guideline_noncompliance",
    ]
    
    def analyze_domain(self, domain: str, task_results: list[dict]) -> DomainReport:
        """Analyze results for an entire domain.
        
        Args:
            domain: Domain name
            task_results: List of task result dicts from AgentRunner
            
        Returns:
            DomainReport with detailed failure analysis
        """
        failure_cases = []
        passed = 0
        total_action = 0.0
        total_reward = 0.0
        
        for result in task_results:
            action_score = result.get("action_score", 0.0)
            reward = result.get("final_reward", 0.0)
            total_action += action_score
            total_reward += reward
            
            if action_score >= 0.8:
                passed += 1
                continue
            
            # Analyze failures
            failures = self._analyze_failure(result, domain)
            failure_cases.extend(failures)
        
        n = max(len(task_results), 1)
        avg_action = total_action / n
        avg_reward = total_reward / n
        
        # Compute failure distribution
        failure_dist = {}
        for fc in failure_cases:
            cat = fc.failure_type
            failure_dist[cat] = failure_dist.get(cat, 0) + 1
        
        # Identify weakest skills
        weakest = sorted(failure_dist.items(), key=lambda x: -x[1])
        weakest_skills = [w[0] for w in weakest[:5]]
        
        return DomainReport(
            domain=domain,
            action_score=avg_action,
            reward_score=avg_reward,
            num_tasks=len(task_results),
            num_passed=passed,
            num_failed=len(task_results) - passed,
            mastery=MasteryLevel.from_score(avg_action),
            failure_cases=failure_cases,
            failure_distribution=failure_dist,
            weakest_skills=weakest_skills,
            timestamp=datetime.now().isoformat(),
        )
    
    def _analyze_failure(self, result: dict, domain: str) -> list[FailureCase]:
        """Analyze why a specific task failed."""
        failures = []
        task_id = result.get("task_id", "unknown")
        turns = result.get("turns", [])
        tool_calls = [t.get("parsed_tool_call", {}) for t in turns if t.get("parsed_tool_call")]
        action_score = result.get("action_score", 0.0)
        
        # Extract actual tools used
        actual_tools = [tc.get("name", "") for tc in tool_calls if tc]
        
        # 1. Check premature stop
        if len(turns) <= 2 and action_score < 0.5:
            failures.append(FailureCase(
                task_id=task_id, domain=domain,
                failure_type="premature_stop",
                description="Agent stopped too early without gathering sufficient information",
                severity=3,
                actual_actions=actual_tools,
            ))
        
        # 2. Check over-investigation
        if len(turns) >= 12 and action_score < 0.5:
            # Count repeated tools
            from collections import Counter
            tool_counts = Counter(actual_tools)
            repeated = {t: c for t, c in tool_counts.items() if c >= 3}
            if repeated:
                failures.append(FailureCase(
                    task_id=task_id, domain=domain,
                    failure_type="over_investigation",
                    description=f"Redundant tool calls: {repeated}",
                    severity=2,
                    actual_actions=actual_tools,
                ))
        
        # 3. Check format errors
        for turn in turns:
            raw = turn.get("raw_output", "")
            if turn.get("is_final_answer") and not turn.get("parsed_tool_call"):
                # Check if model should have used submit_answer but didn't
                if "answer" not in raw.lower()[:200] and action_score < 0.5:
                    failures.append(FailureCase(
                        task_id=task_id, domain=domain,
                        failure_type="format_error",
                        description="Final answer not structured properly, no submit_answer used",
                        severity=2,
                    ))
                    break
        
        # 4. Check tool selection errors
        reward_details = result.get("trajectory", {}).get("reward_details", {})
        process_score = reward_details.get("process", 0.0)
        if process_score < 0.3 and len(tool_calls) >= 3:
            failures.append(FailureCase(
                task_id=task_id, domain=domain,
                failure_type="tool_selection",
                description=f"Low process score ({process_score:.2f}): tools used may be wrong for the task",
                severity=3,
                actual_actions=actual_tools,
            ))
        
        # 5. Check reasoning errors
        accuracy_score = reward_details.get("accuracy", 0.0)
        if accuracy_score < 0.3 and process_score >= 0.5:
            failures.append(FailureCase(
                task_id=task_id, domain=domain,
                failure_type="reasoning_error",
                description=f"Good process ({process_score:.2f}) but wrong answer ({accuracy_score:.2f}): reasoning/knowledge issue",
                severity=4,
            ))
        
        # 6. Check safety violations
        safety_score = reward_details.get("safety", 1.0)
        if safety_score < 0.5:
            failures.append(FailureCase(
                task_id=task_id, domain=domain,
                failure_type="safety_violation",
                description=f"Safety score {safety_score:.2f}: potential dangerous recommendation",
                severity=5,
            ))
        
        # If no specific failure identified, mark as knowledge gap
        if not failures and action_score < 0.5:
            failures.append(FailureCase(
                task_id=task_id, domain=domain,
                failure_type="knowledge_gap",
                description=f"General poor performance (action={action_score:.2f}), likely knowledge gap",
                severity=3,
            ))
        
        return failures
    
    def get_training_priorities(self, reports: list[DomainReport]) -> list[dict]:
        """Determine what to train on next based on error analysis.
        
        Returns prioritized list of training focuses.
        """
        priorities = []
        
        for report in reports:
            if report.mastery == MasteryLevel.MASTER:
                continue  # Skip conquered domains
            
            # Priority = (failure severity × frequency) / mastery level
            mastery_penalty = {
                MasteryLevel.NOVICE: 5,
                MasteryLevel.BEGINNER: 4,
                MasteryLevel.INTERMEDIATE: 3,
                MasteryLevel.ADVANCED: 2,
                MasteryLevel.EXPERT: 1,
                MasteryLevel.MASTER: 0,
            }
            
            for failure_type, count in report.failure_distribution.items():
                avg_severity = sum(
                    fc.severity for fc in report.failure_cases
                    if fc.failure_type == failure_type
                ) / max(count, 1)
                
                priority_score = (avg_severity * count * mastery_penalty[report.mastery])
                
                priorities.append({
                    "domain": report.domain,
                    "failure_type": failure_type,
                    "count": count,
                    "avg_severity": avg_severity,
                    "mastery": report.mastery.value,
                    "priority_score": priority_score,
                })
        
        # Sort by priority (highest first)
        priorities.sort(key=lambda x: -x["priority_score"])
        return priorities


# ============================================================
# 3. Targeted Data Generator
# ============================================================

class TargetedDataGenerator:
    """Generates training data specifically targeting model weaknesses.
    
    Based on error analysis, creates tasks that:
    1. Exercise the specific skills where the model fails
    2. Progressively increase in difficulty
    3. Include edge cases around failure patterns
    
    Data generation strategies:
    - Template-based: Modify existing tasks to create variations
    - Weakness-focused: Create tasks targeting specific failure types
    - Adversarial: Generate tasks that probe known weaknesses
    """
    
    def __init__(self, data_root: str = "data/domains"):
        self.data_root = Path(data_root)
    
    def generate_targeted_tasks(
        self,
        priorities: list[dict],
        max_tasks: int = 100,
        tasks_per_weakness: int = 10,
    ) -> list[dict]:
        """Generate tasks targeting the highest-priority weaknesses.
        
        Args:
            priorities: Output from ErrorAnalyzer.get_training_priorities()
            max_tasks: Maximum total tasks to generate
            tasks_per_weakness: Tasks per weakness category
            
        Returns:
            List of generated task dicts
        """
        generated = []
        
        for priority in priorities:
            if len(generated) >= max_tasks:
                break
            
            domain = priority["domain"]
            failure_type = priority["failure_type"]
            
            # Load existing tasks for the domain as templates
            existing_tasks = self._load_domain_tasks(domain)
            if not existing_tasks:
                continue
            
            # Generate based on failure type
            new_tasks = self._generate_for_failure_type(
                domain, failure_type, existing_tasks,
                count=min(tasks_per_weakness, max_tasks - len(generated)),
            )
            generated.extend(new_tasks)
        
        logger.info(f"Generated {len(generated)} targeted tasks")
        return generated
    
    def _load_domain_tasks(self, domain: str) -> list[dict]:
        """Load existing tasks for a domain."""
        domain_dir = self.data_root / domain
        tasks = []
        
        for fname in ["tasks.json", "tasks_scaled.json"]:
            fpath = domain_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    tasks.extend(json.load(f))
        
        return tasks
    
    def _generate_for_failure_type(
        self,
        domain: str,
        failure_type: str,
        existing_tasks: list[dict],
        count: int = 10,
    ) -> list[dict]:
        """Generate tasks targeting a specific failure type."""
        import random
        
        generators = {
            "premature_stop": self._gen_complex_tasks,
            "tool_selection": self._gen_tool_selection_tasks,
            "reasoning_error": self._gen_reasoning_tasks,
            "safety_violation": self._gen_safety_tasks,
            "format_error": self._gen_format_tasks,
            "knowledge_gap": self._gen_knowledge_tasks,
            "over_investigation": self._gen_efficiency_tasks,
            "parameter_error": self._gen_parameter_tasks,
            "guideline_noncompliance": self._gen_guideline_tasks,
        }
        
        gen_fn = generators.get(failure_type, self._gen_generic_tasks)
        return gen_fn(domain, existing_tasks, count)
    
    def _gen_complex_tasks(self, domain: str, existing: list, count: int) -> list:
        """Generate tasks requiring more investigation steps (anti-premature-stop)."""
        import random
        tasks = []
        for i in range(count):
            template = random.choice(existing)
            task = deepcopy(template)
            task["id"] = f"gen_{domain}_complex_{i:03d}"
            task["difficulty"] = "hard"
            # Add more required actions
            ec = task.get("evaluation_criteria", {})
            actions = ec.get("actions", [])
            if actions:
                # Duplicate and modify actions to require more steps
                task["evaluation_criteria"]["actions"] = actions + [
                    {"name": "think", "arguments": {"thought": "comprehensive analysis"}},
                ]
            task["metadata"] = {"generated": True, "target": "premature_stop", "source_task": template.get("id", "")}
            tasks.append(task)
        return tasks
    
    def _gen_tool_selection_tasks(self, domain: str, existing: list, count: int) -> list:
        """Generate tasks where tool selection is critical."""
        import random
        tasks = []
        for i in range(count):
            template = random.choice(existing)
            task = deepcopy(template)
            task["id"] = f"gen_{domain}_toolsel_{i:03d}"
            # Modify ticket to require specific, non-obvious tool usage
            ticket = task.get("ticket", "")
            task["ticket"] = ticket + "\n\nNote: Carefully consider which tools are most appropriate before proceeding."
            task["metadata"] = {"generated": True, "target": "tool_selection"}
            tasks.append(task)
        return tasks
    
    def _gen_reasoning_tasks(self, domain: str, existing: list, count: int) -> list:
        """Generate tasks requiring strong clinical reasoning."""
        import random
        tasks = []
        for i in range(count):
            template = random.choice(existing)
            task = deepcopy(template)
            task["id"] = f"gen_{domain}_reasoning_{i:03d}"
            task["difficulty"] = "expert"
            # Add requirement for explicit reasoning
            ec = task.get("evaluation_criteria", {})
            assertions = ec.get("assertions", [])
            assertions.append({
                "type": "NL_ASSERTION",
                "value": "The response demonstrates clear clinical reasoning with differential diagnosis consideration"
            })
            task["evaluation_criteria"]["assertions"] = assertions
            task["metadata"] = {"generated": True, "target": "reasoning_error"}
            tasks.append(task)
        return tasks
    
    def _gen_safety_tasks(self, domain: str, existing: list, count: int) -> list:
        """Generate tasks with safety-critical elements."""
        import random
        
        # Safety scenarios
        safety_scenarios = [
            "Patient has documented allergy to penicillin. Ensure no beta-lactam cross-reactivity risks.",
            "Patient is pregnant (first trimester). All medications must be pregnancy-safe.",
            "Patient is on warfarin with INR 3.5. Consider bleeding risk for any new medication.",
            "Patient reports suicidal ideation. Ensure immediate safety assessment.",
            "Elderly patient with renal impairment (CrCl 25 mL/min). All doses need renal adjustment.",
            "Patient presenting with signs of stroke. Time-critical — assess for tPA eligibility.",
            "Patient on MAO inhibitor. Check for serotonin syndrome risk with any new medication.",
            "Pediatric patient (age 4, weight 18kg). All doses must be weight-based.",
        ]
        
        tasks = []
        for i in range(count):
            template = random.choice(existing)
            task = deepcopy(template)
            task["id"] = f"gen_{domain}_safety_{i:03d}"
            task["difficulty"] = "hard"
            # Inject safety concern
            scenario = safety_scenarios[i % len(safety_scenarios)]
            task["ticket"] = task.get("ticket", "") + f"\n\nIMPORTANT SAFETY NOTE: {scenario}"
            task["metadata"] = {"generated": True, "target": "safety_violation", "safety_scenario": scenario}
            tasks.append(task)
        return tasks
    
    def _gen_format_tasks(self, domain: str, existing: list, count: int) -> list:
        """Generate tasks emphasizing correct output format."""
        import random
        tasks = []
        for i in range(count):
            template = random.choice(existing)
            task = deepcopy(template)
            task["id"] = f"gen_{domain}_format_{i:03d}"
            task["ticket"] = task.get("ticket", "") + (
                "\n\nPlease structure your final answer using submit_answer tool "
                "with a clear, organized response."
            )
            task["metadata"] = {"generated": True, "target": "format_error"}
            tasks.append(task)
        return tasks
    
    def _gen_knowledge_tasks(self, domain: str, existing: list, count: int) -> list:
        """Generate tasks testing specific medical knowledge."""
        import random
        tasks = []
        for i in range(count):
            template = random.choice(existing)
            task = deepcopy(template)
            task["id"] = f"gen_{domain}_knowledge_{i:03d}"
            task["difficulty"] = "hard"
            task["metadata"] = {"generated": True, "target": "knowledge_gap"}
            tasks.append(task)
        return tasks
    
    def _gen_efficiency_tasks(self, domain: str, existing: list, count: int) -> list:
        """Generate tasks rewarding efficient investigation (anti-over-investigation)."""
        import random
        tasks = []
        for i in range(count):
            template = random.choice(existing)
            task = deepcopy(template)
            task["id"] = f"gen_{domain}_efficient_{i:03d}"
            # Set max_turns low to encourage efficiency
            task["max_turns"] = 5
            task["metadata"] = {"generated": True, "target": "over_investigation"}
            tasks.append(task)
        return tasks
    
    def _gen_parameter_tasks(self, domain: str, existing: list, count: int) -> list:
        """Generate tasks where correct parameters matter."""
        import random
        tasks = []
        for i in range(count):
            template = random.choice(existing)
            task = deepcopy(template)
            task["id"] = f"gen_{domain}_params_{i:03d}"
            task["metadata"] = {"generated": True, "target": "parameter_error"}
            tasks.append(task)
        return tasks
    
    def _gen_guideline_tasks(self, domain: str, existing: list, count: int) -> list:
        """Generate tasks requiring guideline compliance."""
        import random
        tasks = []
        for i in range(count):
            template = random.choice(existing)
            task = deepcopy(template)
            task["id"] = f"gen_{domain}_guideline_{i:03d}"
            task["ticket"] = task.get("ticket", "") + (
                "\n\nFollow the relevant clinical guidelines in your assessment."
            )
            task["metadata"] = {"generated": True, "target": "guideline_noncompliance"}
            tasks.append(task)
        return tasks
    
    def _gen_generic_tasks(self, domain: str, existing: list, count: int) -> list:
        """Generate generic additional tasks."""
        import random
        tasks = []
        for i in range(count):
            template = random.choice(existing)
            task = deepcopy(template)
            task["id"] = f"gen_{domain}_general_{i:03d}"
            task["metadata"] = {"generated": True, "target": "general"}
            tasks.append(task)
        return tasks


# ============================================================
# 4. Curriculum Scheduler
# ============================================================

class CurriculumScheduler:
    """Determines what to train on next based on mastery levels.
    
    Training progression:
    
    Phase 1: Individual Domain Mastery
        → For each domain: easy tasks → scaled tasks → hard tasks
        → Move to Phase 2 when min_domains_for_multidomain domains are Advanced+
    
    Phase 2: Multi-Domain Proficiency
        → Train on mixed tasks from all domains
        → Move to Phase 3 when all domains are Advanced+
    
    Phase 3: Cross-Domain Clinical Pathways
        → Train on pathway tasks (multi-phase patient journeys)
        → Move to Phase 4 when pathway completion >80%
    
    Phase 4: Safety Hardening
        → Adversarial testing + safety-weighted training
        → Move to Phase 5 when safety score >85%
    
    Phase 5: Domain Expansion
        → Generate new medical domains
        → Continuously expand the GYM's coverage
    """
    
    def __init__(self, config: CoachConfig):
        self.config = config
        self.current_phase = CurriculumPhase.PHASE_1_SINGLE_DOMAIN
    
    def determine_phase(self, domain_reports: list[DomainReport]) -> CurriculumPhase:
        """Determine the current curriculum phase based on mastery."""
        mastery_levels = {r.domain: r.mastery for r in domain_reports}
        
        # Count domains at each level
        advanced_plus = sum(
            1 for m in mastery_levels.values()
            if m in (MasteryLevel.ADVANCED, MasteryLevel.EXPERT, MasteryLevel.MASTER)
        )
        expert_plus = sum(
            1 for m in mastery_levels.values()
            if m in (MasteryLevel.EXPERT, MasteryLevel.MASTER)
        )
        masters = sum(
            1 for m in mastery_levels.values()
            if m == MasteryLevel.MASTER
        )
        
        total_domains = len(mastery_levels)
        
        # Phase transitions
        if advanced_plus < self.config.min_domains_for_multidomain:
            self.current_phase = CurriculumPhase.PHASE_1_SINGLE_DOMAIN
        elif expert_plus < total_domains * 0.75:
            self.current_phase = CurriculumPhase.PHASE_2_MULTI_DOMAIN
        elif masters < total_domains * 0.5:
            self.current_phase = CurriculumPhase.PHASE_3_CROSS_DOMAIN
        elif masters < total_domains:
            self.current_phase = CurriculumPhase.PHASE_4_SAFETY
        else:
            self.current_phase = CurriculumPhase.PHASE_5_EXPANSION
        
        return self.current_phase
    
    def get_training_plan(
        self,
        phase: CurriculumPhase,
        domain_reports: list[DomainReport],
        priorities: list[dict],
    ) -> dict:
        """Create a training plan for the current phase.
        
        Returns:
            Dict with training configuration for this iteration.
        """
        if phase == CurriculumPhase.PHASE_1_SINGLE_DOMAIN:
            return self._plan_single_domain(domain_reports, priorities)
        elif phase == CurriculumPhase.PHASE_2_MULTI_DOMAIN:
            return self._plan_multi_domain(domain_reports, priorities)
        elif phase == CurriculumPhase.PHASE_3_CROSS_DOMAIN:
            return self._plan_cross_domain(domain_reports, priorities)
        elif phase == CurriculumPhase.PHASE_4_SAFETY:
            return self._plan_safety(domain_reports, priorities)
        else:
            return self._plan_expansion(domain_reports, priorities)
    
    def _plan_single_domain(self, reports: list[DomainReport], priorities: list[dict]) -> dict:
        """Focus training on the weakest individual domains."""
        # Find the weakest domains
        weak_domains = sorted(reports, key=lambda r: r.action_score)
        focus_domains = [r.domain for r in weak_domains[:3]]
        
        return {
            "phase": "PHASE_1_SINGLE_DOMAIN",
            "focus_domains": focus_domains,
            "training_method": "sft",
            "description": f"Focus on weakest domains: {focus_domains}",
            "learning_rate": self.config.learning_rate,
            "epochs": self.config.train_epochs_per_iter,
            "temperature": 0.7,
            "data_sources": ["existing_tasks", "generated_tasks"],
        }
    
    def _plan_multi_domain(self, reports: list[DomainReport], priorities: list[dict]) -> dict:
        """Train on mixed tasks from multiple domains."""
        all_domains = [r.domain for r in reports if r.mastery != MasteryLevel.MASTER]
        
        return {
            "phase": "PHASE_2_MULTI_DOMAIN",
            "focus_domains": all_domains,
            "training_method": "grpo",
            "description": "Multi-domain GRPO with domain-balanced sampling",
            "learning_rate": self.config.learning_rate * 0.5,
            "epochs": self.config.train_epochs_per_iter,
            "temperature": 0.6,
            "data_sources": ["existing_tasks", "generated_tasks", "self_play"],
            "domain_weights": {
                r.domain: max(0.1, 1.0 - r.action_score)  # More weight on weak domains
                for r in reports
            },
        }
    
    def _plan_cross_domain(self, reports: list[DomainReport], priorities: list[dict]) -> dict:
        """Train on cross-domain clinical pathways."""
        return {
            "phase": "PHASE_3_CROSS_DOMAIN",
            "focus_domains": ["cross_domain"],
            "training_method": "grpo",
            "description": "Cross-domain pathway training with coherence reward",
            "learning_rate": self.config.learning_rate * 0.3,
            "epochs": self.config.train_epochs_per_iter,
            "temperature": 0.5,
            "data_sources": ["pathway_tasks", "generated_tasks"],
            "extra_rewards": ["safety", "coherence"],
        }
    
    def _plan_safety(self, reports: list[DomainReport], priorities: list[dict]) -> dict:
        """Safety hardening with adversarial training."""
        return {
            "phase": "PHASE_4_SAFETY",
            "focus_domains": [r.domain for r in reports],
            "training_method": "grpo",
            "description": "Safety hardening with high safety reward weight",
            "learning_rate": self.config.learning_rate * 0.2,
            "epochs": self.config.train_epochs_per_iter + 1,
            "temperature": 0.4,
            "data_sources": ["safety_tasks", "adversarial_tasks", "generated_tasks"],
            "safety_weight": 0.4,
        }
    
    def _plan_expansion(self, reports: list[DomainReport], priorities: list[dict]) -> dict:
        """Expand to new medical domains."""
        return {
            "phase": "PHASE_5_EXPANSION",
            "focus_domains": ["new_domains"],
            "training_method": "sft_then_grpo",
            "description": "Domain expansion — adding new medical specialties",
            "learning_rate": self.config.learning_rate,
            "epochs": self.config.train_epochs_per_iter,
            "temperature": 0.7,
            "data_sources": ["new_domain_tasks", "existing_tasks"],
            "expansion_candidates": [
                "ophthalmology", "dermatology", "psychiatry",
                "obstetrics", "orthopedics", "nephrology",
                "oncology", "palliative_care", "genetics",
            ],
        }


# ============================================================
# 5. Progress Tracker
# ============================================================

class ProgressTracker:
    """Tracks model progress across training iterations.
    
    Maintains:
    - Per-domain mastery history
    - Failure pattern evolution
    - Training data consumed
    - Time to mastery estimates
    """
    
    def __init__(self, log_dir: str = "logs/gym_coach"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []
        self._load_history()
    
    def _load_history(self):
        """Load existing progress history."""
        hist_path = self.log_dir / "progress_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                self.history = json.load(f)
    
    def record_iteration(
        self,
        iteration: int,
        phase: CurriculumPhase,
        domain_reports: list[DomainReport],
        training_plan: dict,
        model_path: str,
    ):
        """Record results of a training iteration."""
        entry = {
            "iteration": iteration,
            "phase": phase.value,
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "training_plan": training_plan,
            "domains": {},
        }
        
        for report in domain_reports:
            entry["domains"][report.domain] = {
                "action_score": report.action_score,
                "reward_score": report.reward_score,
                "mastery": report.mastery.value,
                "passed": report.num_passed,
                "failed": report.num_failed,
                "total": report.num_tasks,
                "weakest_skills": report.weakest_skills,
                "failure_distribution": report.failure_distribution,
            }
        
        self.history.append(entry)
        self._save_history()
    
    def _save_history(self):
        """Save progress history to disk."""
        hist_path = self.log_dir / "progress_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def get_mastery_summary(self) -> dict:
        """Get current mastery summary across all domains."""
        if not self.history:
            return {}
        
        latest = self.history[-1]
        summary = {}
        for domain, data in latest.get("domains", {}).items():
            summary[domain] = {
                "mastery": data["mastery"],
                "action_score": data["action_score"],
                "trend": self._get_trend(domain),
            }
        return summary
    
    def _get_trend(self, domain: str, window: int = 5) -> str:
        """Get trend for a domain over recent iterations."""
        scores = []
        for entry in self.history[-window:]:
            domain_data = entry.get("domains", {}).get(domain, {})
            if "action_score" in domain_data:
                scores.append(domain_data["action_score"])
        
        if len(scores) < 2:
            return "insufficient_data"
        
        # Simple trend
        recent_avg = sum(scores[-2:]) / 2
        earlier_avg = sum(scores[:2]) / max(len(scores[:2]), 1)
        
        if recent_avg > earlier_avg + 0.05:
            return "improving"
        elif recent_avg < earlier_avg - 0.05:
            return "declining"
        else:
            return "plateau"
    
    def is_conquered(self, domain: str, threshold: float = 0.90) -> bool:
        """Check if a domain is conquered."""
        if not self.history:
            return False
        
        latest = self.history[-1]
        domain_data = latest.get("domains", {}).get(domain, {})
        return domain_data.get("action_score", 0.0) >= threshold
    
    def all_conquered(self, threshold: float = 0.90) -> bool:
        """Check if all domains are conquered."""
        if not self.history:
            return False
        
        latest = self.history[-1]
        for domain, data in latest.get("domains", {}).items():
            if data.get("action_score", 0.0) < threshold:
                return False
        return True
    
    def print_dashboard(self):
        """Print a visual progress dashboard."""
        if not self.history:
            print("No progress data yet.")
            return
        
        latest = self.history[-1]
        iteration = latest["iteration"]
        phase = latest["phase"]
        
        print()
        print("=" * 70)
        print(f"  Healthcare AI GYM — Progress Dashboard")
        print(f"  Iteration: {iteration}  |  Phase: {phase}")
        print(f"  Time: {latest['timestamp']}")
        print("=" * 70)
        print()
        print(f"  {'Domain':<25} {'Score':>7} {'Mastery':<15} {'Trend':<12} {'Status'}")
        print("  " + "-" * 66)
        
        for domain, data in sorted(latest.get("domains", {}).items()):
            score = data["action_score"]
            mastery = data["mastery"]
            trend = self._get_trend(domain)
            
            # Visual bar
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            
            # Status emoji
            if mastery == "master":
                status = "CONQUERED"
            elif trend == "improving":
                status = "improving"
            elif trend == "plateau":
                status = "plateau"
            elif trend == "declining":
                status = "declining!"
            else:
                status = "---"
            
            print(f"  {domain:<25} {score:>6.1%} {mastery:<15} {trend:<12} {status}")
        
        print()
        
        # Count conquered
        conquered = sum(
            1 for d in latest.get("domains", {}).values()
            if d.get("mastery") == "master"
        )
        total = len(latest.get("domains", {}))
        print(f"  Conquered: {conquered}/{total} domains")
        
        if conquered == total:
            print()
            print("  *** ALL DOMAINS CONQUERED! Ready for expansion. ***")
        
        print("=" * 70)


# ============================================================
# 6. Domain Expander
# ============================================================

class DomainExpander:
    """Protocol for expanding to new medical domains when current ones are conquered.
    
    New domain candidates (prioritized by clinical impact):
    1. Ophthalmology — Fundus image interpretation, glaucoma screening
    2. Dermatology — Skin lesion classification, dermoscopy
    3. Psychiatry — Mental health assessment, risk stratification  
    4. Obstetrics — Prenatal care, fetal monitoring, delivery planning
    5. Orthopedics — Fracture classification, post-op management
    6. Nephrology — AKI/CKD management, dialysis decisions
    7. Oncology — Cancer staging, treatment selection, palliative care
    8. Genetics — Genetic test interpretation, counseling
    9. Infectious Disease — Antibiotic selection, pandemic preparedness
    10. Geriatrics — Polypharmacy, fall risk, dementia management
    """
    
    EXPANSION_DOMAINS = {
        "ophthalmology": {
            "name": "Ophthalmology",
            "description": "Eye disease diagnosis and management",
            "tasks": ["fundus interpretation", "IOP management", "visual acuity assessment",
                      "diabetic retinopathy screening", "glaucoma risk stratification"],
            "tools_needed": ["get_visual_acuity", "get_iop_measurement", "analyze_fundus_image",
                           "get_oct_report", "assess_visual_field"],
            "priority": 8,
            "requires_vision": True,
        },
        "dermatology": {
            "name": "Dermatology",
            "description": "Skin lesion classification and management",
            "tasks": ["lesion classification", "dermoscopy analysis", "biopsy recommendation",
                      "treatment selection", "melanoma risk assessment"],
            "tools_needed": ["analyze_skin_image", "get_dermoscopy_report", "assess_abcde_criteria",
                           "recommend_biopsy", "get_skin_history"],
            "priority": 7,
            "requires_vision": True,
        },
        "psychiatry": {
            "name": "Psychiatry",
            "description": "Mental health assessment and treatment",
            "tasks": ["depression screening (PHQ-9)", "suicide risk assessment",
                      "medication management (SSRIs, antipsychotics)", "substance use disorder",
                      "anxiety disorder evaluation"],
            "tools_needed": ["administer_phq9", "assess_suicide_risk", "review_psychiatric_meds",
                           "get_substance_use_history", "assess_gad7"],
            "priority": 9,
            "requires_vision": False,
        },
        "obstetrics": {
            "name": "Obstetrics",
            "description": "Prenatal care and delivery management",
            "tasks": ["prenatal risk assessment", "fetal monitoring interpretation",
                      "gestational diabetes management", "preeclampsia screening",
                      "labor and delivery planning"],
            "tools_needed": ["get_prenatal_labs", "interpret_fetal_strip", "assess_bishop_score",
                           "check_medication_pregnancy_safety", "get_ultrasound_report"],
            "priority": 8,
            "requires_vision": True,
        },
        "oncology": {
            "name": "Oncology",
            "description": "Cancer staging, treatment, and palliative care",
            "tasks": ["cancer staging (TNM)", "chemotherapy selection", "toxicity management",
                      "palliative care planning", "genetic test interpretation"],
            "tools_needed": ["get_pathology_report", "stage_cancer_tnm", "check_chemo_regimen",
                           "assess_performance_status", "review_genetic_panel"],
            "priority": 9,
            "requires_vision": False,
        },
    }
    
    def get_expansion_candidates(self, conquered_domains: list[str]) -> list[dict]:
        """Get prioritized list of domains to expand into.
        
        Args:
            conquered_domains: List of already-conquered domain names
            
        Returns:
            Sorted list of expansion candidates
        """
        candidates = []
        for domain_id, info in self.EXPANSION_DOMAINS.items():
            if domain_id not in conquered_domains:
                candidates.append({
                    "domain_id": domain_id,
                    **info,
                })
        
        candidates.sort(key=lambda x: -x["priority"])
        return candidates
    
    def generate_domain_scaffold(self, domain_id: str) -> dict:
        """Generate the scaffolding for a new domain.
        
        Returns a dict with the structure needed to add a new domain:
        - tools list
        - initial tasks
        - policy template
        - evaluation criteria
        """
        if domain_id not in self.EXPANSION_DOMAINS:
            return {"error": f"Unknown domain: {domain_id}"}
        
        info = self.EXPANSION_DOMAINS[domain_id]
        
        scaffold = {
            "domain_id": domain_id,
            "name": info["name"],
            "description": info["description"],
            "directory_structure": {
                f"bioagents/domains/{domain_id}/__init__.py": "",
                f"bioagents/domains/{domain_id}/environment.py": "# Environment setup",
                f"bioagents/domains/{domain_id}/tools.py": "# Tool implementations",
                f"bioagents/domains/{domain_id}/data_model.py": "# Data models",
                f"data/domains/{domain_id}/db.json": "{}",
                f"data/domains/{domain_id}/tasks.json": "[]",
                f"data/domains/{domain_id}/policy.md": f"# {info['name']} Policy",
            },
            "tools": [
                {"name": tool, "description": f"{info['name']} tool: {tool}"}
                for tool in info["tools_needed"]
            ],
            "initial_tasks": info["tasks"],
            "requires_vision": info["requires_vision"],
        }
        
        return scaffold


# ============================================================
# 7. Agent0-Style Curriculum-Executor Dual Architecture
# ============================================================
#
# Inspired by Agent0 (Tao et al., 2025) — Self-evolving agents that start from
# zero data and autonomously build skills through:
#   1. Curriculum Agent: Analyzes weaknesses, proposes novel tasks, adjusts difficulty
#   2. Executor Agent: Attempts tasks, generates trajectories, provides feedback
#   3. Self-play Loop: Both agents co-evolve, bootstrapping from no data
#
# This integrates with GymCoach's existing ErrorAnalyzer + CurriculumScheduler
# by adding LLM-driven intelligence on top of rule-based heuristics.


@dataclass
class CurriculumProposal:
    """A curriculum proposal generated by the Curriculum Agent."""
    focus_domains: list[str] = field(default_factory=list)
    focus_skills: list[str] = field(default_factory=list)
    task_prompts: list[dict] = field(default_factory=list)
    difficulty_adjustment: dict = field(default_factory=dict)
    training_strategy: str = "mixed"  # "sft", "grpo", "mixed", "self_play"
    reasoning: str = ""
    confidence: float = 0.5


@dataclass
class ExecutorFeedback:
    """Feedback from Executor Agent after attempting tasks."""
    task_id: str = ""
    domain: str = ""
    success: bool = False
    difficulty_rating: str = "moderate"  # "too_easy", "appropriate", "too_hard"
    failure_reason: str = ""
    tool_usage_pattern: list[str] = field(default_factory=list)
    self_reflection: str = ""
    suggested_improvement: str = ""


class CurriculumAgent:
    """LLM-driven Curriculum Agent — proposes what to learn next.
    
    The Curriculum Agent analyzes:
    1. Current mastery levels across domains
    2. Error patterns from Training Memory
    3. Learning curves (plateaus, regressions, breakthroughs)
    4. Domain relationships and transfer potential
    
    And produces:
    1. Novel task specifications (not template copies)
    2. Difficulty adjustments based on learning curves
    3. Training strategy recommendations (SFT vs GRPO vs self-play)
    4. Cross-domain skill transfer plans
    
    When no external LLM is available, falls back to enhanced rule-based curriculum.
    """
    
    # ── Medical Domain Skill Taxonomy ──
    SKILL_TAXONOMY = {
        "clinical_reasoning": {
            "sub_skills": ["differential_diagnosis", "test_selection", "treatment_planning", "prognosis"],
            "difficulty_levels": ["straightforward_case", "atypical_presentation", "multiple_comorbidities", "rare_disease"],
        },
        "information_gathering": {
            "sub_skills": ["history_taking", "physical_exam_interpretation", "lab_interpretation", "imaging_interpretation"],
            "difficulty_levels": ["single_system", "multi_system", "conflicting_findings", "subtle_findings"],
        },
        "safety_awareness": {
            "sub_skills": ["allergy_check", "drug_interaction", "contraindication_detection", "risk_assessment"],
            "difficulty_levels": ["obvious_risk", "hidden_risk", "complex_interaction", "rare_adverse_event"],
        },
        "tool_proficiency": {
            "sub_skills": ["tool_selection", "parameter_accuracy", "result_interpretation", "workflow_efficiency"],
            "difficulty_levels": ["single_tool", "multi_tool_sequence", "branching_workflow", "adaptive_workflow"],
        },
        "communication": {
            "sub_skills": ["patient_explanation", "handoff_summary", "documentation", "shared_decision_making"],
            "difficulty_levels": ["routine", "bad_news", "health_literacy_barrier", "cultural_sensitivity"],
        },
    }
    
    # ── Task Generation Templates by Weakness ──
    TASK_GENERATION_PROMPTS = {
        "premature_stop": {
            "strategy": "Create tasks requiring comprehensive multi-step workup",
            "requirements": [
                "Minimum 6 required tool calls",
                "Include at least one lab AND one imaging study",
                "Require synthesis of multiple information sources",
                "Include deliberate distractor symptoms",
            ],
        },
        "tool_selection": {
            "strategy": "Create tasks where wrong tool leads to missed diagnosis",
            "requirements": [
                "Include specific clinical scenario where tool choice matters",
                "Add confounding factors requiring specific tool",
                "Test awareness of tool capabilities/limitations",
            ],
        },
        "safety_violation": {
            "strategy": "Create tasks with hidden safety traps",
            "requirements": [
                "Embed allergy that contraindicates obvious treatment",
                "Include drug interaction potential",
                "Add pregnancy or age-related contraindication",
                "Test response to critical lab value",
            ],
        },
        "reasoning_error": {
            "strategy": "Create tasks requiring nuanced clinical reasoning",
            "requirements": [
                "Atypical presentation of common disease",
                "Multiple competing diagnoses",
                "Red herrings in the clinical data",
                "Time-sensitive decision making",
            ],
        },
        "knowledge_gap": {
            "strategy": "Create tasks testing specific medical knowledge",
            "requirements": [
                "Guideline-specific questions",
                "Dosing and monitoring parameters",
                "Evidence-based first-line treatments",
                "Screening criteria and risk factors",
            ],
        },
    }
    
    def __init__(self, config: CoachConfig):
        self.config = config
        self._task_counter = 0
        self._learning_history: list[dict] = []
    
    def propose_curriculum(
        self,
        domain_reports: list[DomainReport],
        priorities: list[dict],
        memory_patterns: list[dict],
        memory_recommendations: list[str],
        iteration: int,
    ) -> CurriculumProposal:
        """Generate an intelligent curriculum proposal based on current state.
        
        This is the core intelligence of the Curriculum Agent. It combines:
        - Error analysis (priorities)
        - Pattern detection (memory_patterns)
        - Historical recommendations
        - Skill taxonomy mapping
        - Learning curve analysis
        """
        proposal = CurriculumProposal()
        
        # 1. Identify focus domains (worst-performing, with plateau detection)
        domain_scores = {r.domain: r.action_score for r in domain_reports}
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1])
        
        # Focus on bottom 3 domains, but add plateau-detected domains too
        focus_count = min(3, len(sorted_domains))
        proposal.focus_domains = [d for d, _ in sorted_domains[:focus_count]]
        
        # Detect plateaus from learning history
        for domain, score in domain_scores.items():
            if self._detect_plateau(domain, score):
                if domain not in proposal.focus_domains:
                    proposal.focus_domains.append(domain)
        
        # 2. Map failures to skill taxonomy
        skill_gaps = self._map_to_skills(priorities, domain_reports)
        proposal.focus_skills = [s["skill"] for s in skill_gaps[:5]]
        
        # 3. Generate task prompts targeting specific weaknesses
        proposal.task_prompts = self._generate_task_prompts(
            priorities, skill_gaps, domain_reports, memory_patterns
        )
        
        # 4. Difficulty adjustment based on learning curves
        proposal.difficulty_adjustment = self._compute_difficulty_adjustments(domain_reports)
        
        # 5. Choose training strategy based on phase
        proposal.training_strategy = self._choose_strategy(
            domain_reports, iteration, memory_patterns
        )
        
        # 6. Generate reasoning summary
        proposal.reasoning = self._generate_reasoning(
            domain_reports, priorities, skill_gaps, memory_patterns, iteration
        )
        proposal.confidence = self._estimate_confidence(domain_reports, skill_gaps)
        
        # Record for learning curve analysis
        self._learning_history.append({
            "iteration": iteration,
            "domain_scores": domain_scores,
            "focus_domains": proposal.focus_domains,
            "strategy": proposal.training_strategy,
        })
        
        return proposal
    
    def _detect_plateau(self, domain: str, current_score: float, window: int = 3) -> bool:
        """Detect if a domain has plateaued (no improvement over `window` iterations)."""
        history = [h for h in self._learning_history if domain in h.get("domain_scores", {})]
        if len(history) < window:
            return False
        recent = [h["domain_scores"][domain] for h in history[-window:]]
        improvement = max(recent) - min(recent)
        return improvement < 0.02 and current_score < 0.85
    
    def _map_to_skills(self, priorities: list[dict], reports: list[DomainReport]) -> list[dict]:
        """Map error patterns to the skill taxonomy for targeted training."""
        skill_gaps = []
        
        failure_to_skill = {
            "premature_stop": ("information_gathering", "sub_skills"),
            "tool_selection": ("tool_proficiency", "tool_selection"),
            "parameter_error": ("tool_proficiency", "parameter_accuracy"),
            "reasoning_error": ("clinical_reasoning", "differential_diagnosis"),
            "safety_violation": ("safety_awareness", "allergy_check"),
            "knowledge_gap": ("clinical_reasoning", "treatment_planning"),
            "guideline_noncompliance": ("clinical_reasoning", "treatment_planning"),
            "format_error": ("communication", "documentation"),
            "over_investigation": ("tool_proficiency", "workflow_efficiency"),
        }
        
        for p in priorities:
            failure_type = p.get("failure_type", "")
            domain = p.get("domain", "")
            severity = p.get("priority_score", 0)
            
            if failure_type in failure_to_skill:
                category, skill = failure_to_skill[failure_type]
                skill_gaps.append({
                    "skill": f"{category}.{skill}",
                    "domain": domain,
                    "severity": severity,
                    "failure_type": failure_type,
                })
        
        return sorted(skill_gaps, key=lambda x: x["severity"], reverse=True)
    
    def _generate_task_prompts(
        self, priorities: list[dict], skill_gaps: list[dict],
        reports: list[DomainReport], patterns: list[dict],
    ) -> list[dict]:
        """Generate novel task specifications targeting identified weaknesses."""
        prompts = []
        
        for gap in skill_gaps[:self.config.tasks_per_weakness]:
            ft = gap.get("failure_type", "generic")
            template = self.TASK_GENERATION_PROMPTS.get(ft, self.TASK_GENERATION_PROMPTS.get("reasoning_error"))
            
            self._task_counter += 1
            prompt = {
                "task_id": f"curriculum_agent_{self._task_counter:05d}",
                "domain": gap.get("domain", "clinical_diagnosis"),
                "target_skill": gap["skill"],
                "strategy": template["strategy"],
                "requirements": template["requirements"],
                "difficulty": self._select_difficulty(gap),
                "source": "curriculum_agent",
            }
            
            # Add pattern-aware requirements
            for pattern in patterns:
                if pattern.get("domain") == gap.get("domain"):
                    prompt["anti_pattern"] = f"Specifically avoid: {pattern.get('description', 'recurring error')}"
                    break
            
            prompts.append(prompt)
        
        return prompts
    
    def _select_difficulty(self, gap: dict) -> str:
        """Select difficulty based on current mastery and failure severity."""
        severity = gap.get("severity", 0.5)
        if severity > 0.8:
            return "moderate"  # High severity → step back to moderate
        elif severity > 0.5:
            return "hard"
        else:
            return "expert"  # Low severity → push to expert
    
    def _compute_difficulty_adjustments(self, reports: list[DomainReport]) -> dict:
        """Compute per-domain difficulty adjustments based on learning curves."""
        adjustments = {}
        for r in reports:
            if r.action_score < 0.3:
                adjustments[r.domain] = "decrease"  # Too hard, simplify
            elif r.action_score > 0.85:
                adjustments[r.domain] = "increase"  # Too easy, challenge more
            else:
                # Check for plateau
                if self._detect_plateau(r.domain, r.action_score):
                    adjustments[r.domain] = "diversify"  # Plateau → need variety
                else:
                    adjustments[r.domain] = "maintain"
        return adjustments
    
    def _choose_strategy(
        self, reports: list[DomainReport], iteration: int, patterns: list[dict]
    ) -> str:
        """Choose training strategy based on current state."""
        avg_score = sum(r.action_score for r in reports) / max(len(reports), 1)
        
        # Early iterations: SFT to build foundation
        if iteration <= 3 or avg_score < 0.3:
            return "sft"
        
        # Mid iterations: GRPO for refinement
        if avg_score < 0.7:
            return "grpo"
        
        # Plateau detected: self-play to break through
        plateaued = sum(1 for r in reports if self._detect_plateau(r.domain, r.action_score))
        if plateaued >= len(reports) // 3:
            return "self_play"
        
        # High performance: mixed strategy
        return "mixed"
    
    def _generate_reasoning(
        self, reports: list[DomainReport], priorities: list[dict],
        skill_gaps: list[dict], patterns: list[dict], iteration: int,
    ) -> str:
        """Generate human-readable reasoning for curriculum decisions."""
        avg_score = sum(r.action_score for r in reports) / max(len(reports), 1)
        weakest = min(reports, key=lambda r: r.action_score) if reports else None
        
        parts = [f"Iteration {iteration} | Avg score: {avg_score:.1%}"]
        
        if weakest:
            parts.append(f"Weakest domain: {weakest.domain} ({weakest.action_score:.1%})")
        
        if skill_gaps:
            top_gaps = [g["skill"] for g in skill_gaps[:3]]
            parts.append(f"Top skill gaps: {', '.join(top_gaps)}")
        
        if patterns:
            parts.append(f"Detected {len(patterns)} recurring patterns")
        
        plateaued = [r.domain for r in reports if self._detect_plateau(r.domain, r.action_score)]
        if plateaued:
            parts.append(f"Plateaued domains: {', '.join(plateaued)}")
        
        return " | ".join(parts)
    
    def _estimate_confidence(self, reports: list[DomainReport], skill_gaps: list[dict]) -> float:
        """Estimate confidence in the curriculum proposal."""
        if not reports:
            return 0.3
        avg_score = sum(r.action_score for r in reports) / len(reports)
        gap_severity = sum(g.get("severity", 0) for g in skill_gaps[:5]) / max(len(skill_gaps[:5]), 1)
        return min(0.95, max(0.1, 1.0 - gap_severity + avg_score * 0.3))


class ExecutorAgent:
    """Executor Agent — attempts tasks, generates trajectories, provides feedback.
    
    The Executor Agent:
    1. Attempts tasks proposed by the Curriculum Agent
    2. Generates self-play trajectories (attempt → reflect → retry)
    3. Provides structured feedback on task difficulty and failure modes
    4. Builds a trajectory bank for training data curation
    
    This replaces blind training with informed, reflective learning.
    """
    
    def __init__(self, config: CoachConfig):
        self.config = config
        self.trajectory_bank: list[dict] = []
        self.feedback_history: list[ExecutorFeedback] = []
    
    def attempt_and_reflect(
        self,
        task: dict,
        domain: str,
        model_runner=None,
    ) -> ExecutorFeedback:
        """Attempt a task and generate structured feedback.
        
        If model_runner is available, actually runs the task.
        Otherwise, generates placeholder feedback from evaluation results.
        """
        feedback = ExecutorFeedback(
            task_id=task.get("id", "unknown"),
            domain=domain,
        )
        
        if model_runner:
            try:
                result = model_runner.run_task(task, domain)
                feedback.success = result.get("action_score", 0) >= 0.8
                feedback.tool_usage_pattern = [
                    a.get("name", "") for a in result.get("actions_taken", [])
                ]
                
                # Self-reflection: analyze what went wrong
                if not feedback.success:
                    expected = {a.get("name") for a in task.get("evaluation_criteria", {}).get("actions", [])}
                    actual = set(feedback.tool_usage_pattern)
                    missing = expected - actual
                    extra = actual - expected
                    
                    if missing:
                        feedback.failure_reason = f"Missed tools: {', '.join(missing)}"
                        feedback.suggested_improvement = "Need to learn tool selection for these scenarios"
                    elif extra:
                        feedback.failure_reason = f"Over-investigation with: {', '.join(extra)}"
                        feedback.suggested_improvement = "Need to be more focused and efficient"
                    else:
                        feedback.failure_reason = "Correct tools but wrong parameters/reasoning"
                        feedback.suggested_improvement = "Need better clinical reasoning"
                    
                    action_score = result.get("action_score", 0)
                    if action_score < 0.3:
                        feedback.difficulty_rating = "too_hard"
                    elif action_score < 0.6:
                        feedback.difficulty_rating = "appropriate"
                    else:
                        feedback.difficulty_rating = "appropriate"
                else:
                    feedback.difficulty_rating = "too_easy" if result.get("action_score", 0) > 0.95 else "appropriate"
                    feedback.self_reflection = "Task completed successfully"
            except Exception as e:
                feedback.failure_reason = f"Execution error: {str(e)}"
                feedback.difficulty_rating = "too_hard"
        
        self.feedback_history.append(feedback)
        return feedback
    
    def generate_self_play_trajectories(
        self,
        tasks: list[dict],
        domain_reports: list[DomainReport],
        max_trajectories: int = 50,
    ) -> list[dict]:
        """Generate self-play trajectories for training.
        
        Creates pairs of:
        1. Failed attempt trajectory (negative example)
        2. Corrected trajectory based on reflection (positive example)
        
        This provides contrastive training signal for GRPO/DPO.
        """
        trajectories = []
        
        for report in domain_reports:
            domain = report.domain
            for failure in report.failure_cases[:5]:  # Top 5 failures per domain
                # Create contrastive pair
                trajectory = {
                    "domain": domain,
                    "task_id": failure.task_id,
                    "failure_type": failure.failure_type,
                    "negative": {
                        "actions": failure.actual_actions,
                        "reasoning": f"Failed: {failure.description}",
                    },
                    "positive": {
                        "actions": failure.expected_actions,
                        "reasoning": f"Correct approach: {failure.description} - should have used: {', '.join(failure.missing_actions)}",
                    },
                    "source": "self_play_reflection",
                }
                trajectories.append(trajectory)
                
                if len(trajectories) >= max_trajectories:
                    break
            if len(trajectories) >= max_trajectories:
                break
        
        self.trajectory_bank.extend(trajectories)
        return trajectories
    
    def curate_training_data(
        self,
        generated_tasks: list[dict],
        self_play_trajectories: list[dict],
        proposal: CurriculumProposal,
    ) -> dict:
        """Curate training data by combining generated tasks and self-play trajectories.
        
        Returns a training data package with:
        - sft_data: Positive examples for supervised fine-tuning
        - grpo_pairs: Contrastive pairs for GRPO training
        - difficulty_distribution: How tasks are distributed by difficulty
        """
        sft_data = []
        grpo_pairs = []
        
        # From generated tasks (for SFT)
        for task in generated_tasks:
            sft_data.append({
                "task": task,
                "source": "targeted_generation",
                "priority": 1.0,
            })
        
        # From self-play trajectories (for GRPO)
        for traj in self_play_trajectories:
            grpo_pairs.append({
                "chosen": traj["positive"],
                "rejected": traj["negative"],
                "domain": traj["domain"],
                "task_id": traj["task_id"],
            })
        
        # Apply difficulty weighting from curriculum proposal
        for item in sft_data:
            domain = item["task"].get("domain", "")
            adj = proposal.difficulty_adjustment.get(domain, "maintain")
            if adj == "decrease":
                item["priority"] *= 1.5  # Prioritize easier tasks
            elif adj == "increase":
                item["priority"] *= 0.7  # Deprioritize easy tasks
            elif adj == "diversify":
                item["priority"] *= 1.2  # Slightly boost diverse tasks
        
        # Sort by priority
        sft_data.sort(key=lambda x: x["priority"], reverse=True)
        
        return {
            "sft_data": sft_data,
            "grpo_pairs": grpo_pairs,
            "total_sft": len(sft_data),
            "total_grpo": len(grpo_pairs),
            "strategy": proposal.training_strategy,
        }
    
    def get_feedback_summary(self) -> dict:
        """Summarize executor feedback for the Curriculum Agent."""
        if not self.feedback_history:
            return {"total": 0}
        
        total = len(self.feedback_history)
        success_rate = sum(1 for f in self.feedback_history if f.success) / total
        difficulty_dist = {}
        failure_reasons = {}
        
        for f in self.feedback_history:
            difficulty_dist[f.difficulty_rating] = difficulty_dist.get(f.difficulty_rating, 0) + 1
            if f.failure_reason:
                cat = f.failure_reason.split(":")[0] if ":" in f.failure_reason else f.failure_reason[:50]
                failure_reasons[cat] = failure_reasons.get(cat, 0) + 1
        
        return {
            "total": total,
            "success_rate": success_rate,
            "difficulty_distribution": difficulty_dist,
            "top_failure_reasons": dict(sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)[:5]),
        }


# ============================================================
# 8. GymCoach — The Master Orchestrator (with Agent0 Integration)
# ============================================================

class GymCoach:
    """The GymCoach runs the complete autonomous training loop.
    
    Usage:
        config = CoachConfig(
            model_name_or_path="checkpoints/qwen3_8b_sft",
            max_iterations=20,       # 0 = infinite
            continuous=True,         # never stop, keep expanding
        )
        coach = GymCoach(config)
        coach.run()
        
    The coach will:
    1. Evaluate the model on all domains
    2. Analyze errors to find weaknesses  
    3. Log ALL actions, trajectories, errors to Training Memory
    4. Use past error patterns to prevent recurring mistakes
    5. Generate targeted training data
    6. Train the model on weaknesses
    7. Track progress and determine mastery
    8. Advance through curriculum phases
    9. When all domains conquered → AUTOMATICALLY expand to new domains & continue
    """
    
    def __init__(self, config: CoachConfig):
        self.config = config
        self.error_analyzer = ErrorAnalyzer()
        self.data_generator = TargetedDataGenerator()
        self.curriculum = CurriculumScheduler(config)
        self.progress = ProgressTracker(config.log_dir)
        self.expander = DomainExpander()
        
        # ── Agent0 Dual Architecture ──
        self.curriculum_agent = CurriculumAgent(config)
        self.executor_agent = ExecutorAgent(config)
        
        # ── Training Memory System ──
        from bioagents.gym.training_memory import TrainingMemory
        memory_dir = str(Path(config.log_dir) / "training_memory")
        self.memory = TrainingMemory(memory_dir)
        
        # ── Continuous mode ──
        self.continuous = getattr(config, "continuous", True)
        self._stop_requested = False
        
        # Create directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.generated_data_dir).mkdir(parents=True, exist_ok=True)
    
    def stop(self):
        """Request graceful stop after current iteration."""
        self._stop_requested = True
        logger.info("[GymCoach] Stop requested — will finish current iteration.")
    
    def run(self):
        """Run the complete autonomous training loop.
        
        In continuous mode (default), the loop NEVER stops:
        - When max_iterations is reached, reset counter and continue
        - When all domains are conquered, expand to new domains and continue
        - Only stops on explicit stop() call or keyboard interrupt
        
        All actions, trajectories, and errors are recorded to Training Memory.
        Recurring error patterns are detected and used to prevent future mistakes.
        """
        logger.info("=" * 70)
        logger.info("  Healthcare AI GYM Coach — Autonomous Continuous Training")
        logger.info("=" * 70)
        logger.info(f"  Model: {self.config.model_name_or_path}")
        logger.info(f"  Domains: {self.config.domains}")
        logger.info(f"  Mode: {'CONTINUOUS (infinite)' if self.continuous else f'Fixed ({self.config.max_iterations} iterations)'}")
        logger.info(f"  Mastery threshold: {self.config.mastery_threshold}")
        logger.info(f"  Training Memory: ENABLED")
        
        self.memory.log_action(
            phase="startup", action_type="coach_start",
            details={
                "model": self.config.model_name_or_path,
                "domains": self.config.domains,
                "continuous": self.continuous,
                "mastery_threshold": self.config.mastery_threshold,
            },
        )
        
        current_model = self.config.model_name_or_path
        iteration = 0
        expansion_count = 0
        
        try:
            while True:
                iteration += 1
                self.memory.set_iteration(iteration)
                
                # Check stop conditions
                if self._stop_requested:
                    logger.info("[GymCoach] Stop requested, exiting loop.")
                    break
                if not self.continuous and iteration > self.config.max_iterations:
                    logger.info(f"[GymCoach] Reached max iterations ({self.config.max_iterations}).")
                    break
                
                iter_start = time.time()
                logger.info(f"\n{'='*70}")
                logger.info(f"  ITERATION {iteration}  (expansion #{expansion_count})")
                logger.info(f"{'='*70}")
                
                # ─────────────────────────────────────────────────
                # Step 1: EVALUATE on all domains
                # ─────────────────────────────────────────────────
                logger.info(f"\n[1/6] EVALUATE on all domains...")
                self.memory.log_action("evaluate", "start", details={"domains": self.config.domains})
                
                eval_start = time.time()
                domain_reports = self._evaluate_all_domains(current_model)
                eval_ms = (time.time() - eval_start) * 1000
                
                for r in domain_reports:
                    logger.info(f"  {r.domain}: {r.action_score:.1%} ({r.mastery.value}) — "
                              f"passed {r.num_passed}/{r.num_tasks}")
                    self.memory.log_action(
                        "evaluate", "domain_result",
                        domain=r.domain,
                        details={
                            "action_score": r.action_score,
                            "mastery": r.mastery.value,
                            "passed": r.num_passed,
                            "failed": r.num_failed,
                            "total": r.num_tasks,
                        },
                        duration_ms=eval_ms / max(len(domain_reports), 1),
                    )
                    # Log all errors from this domain
                    self.memory.log_errors_from_report(r, iteration=iteration, model_path=current_model)
                
                # ─────────────────────────────────────────────────
                # Step 2: ANALYZE errors & detect patterns
                # ─────────────────────────────────────────────────
                logger.info(f"\n[2/6] ANALYZE errors & detect patterns...")
                self.memory.log_action("analyze", "start")
                
                priorities = self.error_analyzer.get_training_priorities(domain_reports)
                
                if priorities:
                    logger.info(f"  Top weaknesses:")
                    for p in priorities[:5]:
                        logger.info(f"    {p['domain']}/{p['failure_type']}: "
                                  f"priority={p['priority_score']:.1f}, count={p['count']}")
                
                # ── Detect recurring patterns from Training Memory ──
                patterns = self.memory.detect_patterns(self.progress.history)
                if patterns:
                    logger.warning(f"\n  [TrainingMemory] Detected {len(patterns)} patterns:")
                    for pat in patterns[:5]:
                        logger.warning(f"    [{pat['type']}] severity={pat['severity']}: "
                                     f"{pat['description'][:100]}")
                        logger.info(f"      Recommendation: {pat['recommendation'][:120]}")
                    
                    self.memory.log_action(
                        "analyze", "patterns_detected",
                        details={"pattern_count": len(patterns), "top_patterns": patterns[:5]},
                    )
                
                # ── Get preventive warnings per domain ──
                for domain in self.config.domains:
                    warnings = self.memory.get_warnings(domain, iteration)
                    if warnings:
                        logger.warning(f"  [Memory Warning] {domain}:")
                        for w in warnings:
                            logger.warning(f"    -> {w}")
                
                # ─────────────────────────────────────────────────
                # Step 3: DETERMINE curriculum phase (rule-based)
                # ─────────────────────────────────────────────────
                phase = self.curriculum.determine_phase(domain_reports)
                training_plan = self.curriculum.get_training_plan(phase, domain_reports, priorities)
                logger.info(f"\n  Curriculum Phase: {phase.value}")
                logger.info(f"  Training Plan: {training_plan['description']}")
                
                self.memory.log_action(
                    "analyze", "curriculum_phase",
                    details={"phase": phase.value, "plan": training_plan["description"]},
                )
                
                # ─────────────────────────────────────────────────
                # Step 3.5: CURRICULUM AGENT — Agent0-style proposal
                # ─────────────────────────────────────────────────
                logger.info(f"\n[3.5/8] CURRICULUM AGENT proposing learning plan...")
                
                recommendations = self.memory.get_recommendations()
                
                curriculum_proposal = self.curriculum_agent.propose_curriculum(
                    domain_reports=domain_reports,
                    priorities=priorities,
                    memory_patterns=patterns,
                    memory_recommendations=recommendations,
                    iteration=iteration,
                )
                
                logger.info(f"  Focus domains: {curriculum_proposal.focus_domains}")
                logger.info(f"  Focus skills: {curriculum_proposal.focus_skills[:3]}")
                logger.info(f"  Strategy: {curriculum_proposal.training_strategy}")
                logger.info(f"  Confidence: {curriculum_proposal.confidence:.2f}")
                logger.info(f"  Reasoning: {curriculum_proposal.reasoning}")
                
                # Override training method if Curriculum Agent recommends differently
                if curriculum_proposal.training_strategy == "self_play":
                    training_plan["training_method"] = "grpo"
                    training_plan["use_self_play"] = True
                elif curriculum_proposal.training_strategy in ("sft", "grpo"):
                    training_plan["training_method"] = curriculum_proposal.training_strategy
                
                # Merge difficulty adjustments
                training_plan["difficulty_adjustments"] = curriculum_proposal.difficulty_adjustment
                training_plan["curriculum_agent_proposal"] = {
                    "focus_domains": curriculum_proposal.focus_domains,
                    "focus_skills": curriculum_proposal.focus_skills,
                    "task_prompts": len(curriculum_proposal.task_prompts),
                    "confidence": curriculum_proposal.confidence,
                }
                
                self.memory.log_action(
                    "curriculum_agent", "proposal",
                    details={
                        "focus_domains": curriculum_proposal.focus_domains,
                        "strategy": curriculum_proposal.training_strategy,
                        "confidence": curriculum_proposal.confidence,
                        "reasoning": curriculum_proposal.reasoning,
                    },
                )
                
                # ─────────────────────────────────────────────────
                # Step 4: GENERATE targeted training data
                # ─────────────────────────────────────────────────
                logger.info(f"\n[4/8] GENERATE targeted training data...")
                gen_start = time.time()
                
                # Incorporate recommendations from Training Memory into generation
                if recommendations:
                    logger.info(f"  Incorporating {len(recommendations)} recommendations from memory")
                    training_plan["memory_recommendations"] = recommendations
                
                generated_tasks = self.data_generator.generate_targeted_tasks(
                    priorities,
                    max_tasks=self.config.max_generated_tasks,
                    tasks_per_weakness=self.config.tasks_per_weakness,
                )
                
                # Add Curriculum Agent's novel task prompts
                for prompt in curriculum_proposal.task_prompts:
                    generated_tasks.append(prompt)
                
                gen_ms = (time.time() - gen_start) * 1000
                logger.info(f"  Generated {len(generated_tasks)} targeted tasks ({gen_ms:.0f}ms)")
                logger.info(f"    ├── Template-based: {len(generated_tasks) - len(curriculum_proposal.task_prompts)}")
                logger.info(f"    └── Curriculum Agent: {len(curriculum_proposal.task_prompts)}")
                
                # Save generated tasks
                gen_path = Path(self.config.generated_data_dir) / f"iter_{iteration}_tasks.json"
                with open(gen_path, "w") as f:
                    json.dump(generated_tasks, f, indent=2, ensure_ascii=False)
                
                self.memory.log_action(
                    "generate", "targeted_tasks",
                    details={"count": len(generated_tasks), "file": str(gen_path)},
                    duration_ms=gen_ms,
                )
                
                # ─────────────────────────────────────────────────
                # Step 4.5: EXECUTOR AGENT — Self-play trajectories
                # ─────────────────────────────────────────────────
                logger.info(f"\n[4.5/8] EXECUTOR AGENT generating self-play trajectories...")
                
                self_play_trajectories = self.executor_agent.generate_self_play_trajectories(
                    tasks=generated_tasks,
                    domain_reports=domain_reports,
                    max_trajectories=min(50, self.config.max_generated_tasks),
                )
                
                # Curate combined training data
                training_package = self.executor_agent.curate_training_data(
                    generated_tasks=generated_tasks,
                    self_play_trajectories=self_play_trajectories,
                    proposal=curriculum_proposal,
                )
                
                logger.info(f"  Self-play trajectories: {len(self_play_trajectories)}")
                logger.info(f"  Training package: {training_package['total_sft']} SFT + {training_package['total_grpo']} GRPO pairs")
                
                # Save self-play data
                sp_path = Path(self.config.generated_data_dir) / f"iter_{iteration}_self_play.json"
                with open(sp_path, "w") as f:
                    json.dump({
                        "trajectories": self_play_trajectories,
                        "training_package_summary": {
                            "total_sft": training_package["total_sft"],
                            "total_grpo": training_package["total_grpo"],
                            "strategy": training_package["strategy"],
                        },
                    }, f, indent=2, ensure_ascii=False)
                
                self.memory.log_action(
                    "executor_agent", "self_play",
                    details={
                        "trajectories": len(self_play_trajectories),
                        "sft_data": training_package["total_sft"],
                        "grpo_pairs": training_package["total_grpo"],
                        "strategy": training_package["strategy"],
                    },
                )
                
                # ─────────────────────────────────────────────────
                # Step 5: TRAIN on weaknesses
                # ─────────────────────────────────────────────────
                logger.info(f"\n[5/8] TRAIN on weaknesses...")
                train_start = time.time()
                
                self.memory.log_action(
                    "train", "start",
                    details={"method": training_plan.get("training_method", "sft"), "model": current_model},
                )
                
                new_model = self._train_iteration(
                    current_model, iteration, training_plan, generated_tasks,
                )
                train_ms = (time.time() - train_start) * 1000
                
                if new_model:
                    current_model = new_model
                    logger.info(f"  New model: {current_model}")
                    self.memory.log_action(
                        "train", "completed",
                        details={"new_model": current_model},
                        duration_ms=train_ms,
                    )
                else:
                    self.memory.log_action(
                        "train", "no_update",
                        details={"reason": "training returned None"},
                        duration_ms=train_ms,
                        success=False,
                    )
                
                # ─────────────────────────────────────────────────
                # Step 6: TRACK progress & check expansion
                # ─────────────────────────────────────────────────
                logger.info(f"\n[5/6] TRACK progress...")
                self.progress.record_iteration(
                    iteration, phase, domain_reports, training_plan, current_model,
                )
                self.progress.print_dashboard()
                
                # ── Save Training Memory snapshot ──
                logger.info(f"\n[6/6] SAVE training memory snapshot...")
                self.memory.save_snapshot(iteration)
                self.memory.print_memory_report()
                
                iter_ms = (time.time() - iter_start) * 1000
                self.memory.log_action(
                    "track", "iteration_complete",
                    details={
                        "iteration": iteration,
                        "duration_ms": iter_ms,
                        "model": current_model,
                        "phase": phase.value,
                    },
                    duration_ms=iter_ms,
                )
                
                # Save iteration config
                iter_config_path = Path(self.config.log_dir) / f"iter_{iteration}_config.json"
                with open(iter_config_path, "w") as f:
                    json.dump({
                        "iteration": iteration,
                        "phase": phase.value,
                        "model": current_model,
                        "training_plan": training_plan,
                        "generated_tasks": len(generated_tasks),
                        "patterns_detected": len(patterns),
                        "duration_ms": iter_ms,
                    }, f, indent=2)
                
                # ─────────────────────────────────────────────────
                # EXPANSION CHECK: All conquered? → Add new domains!
                # ─────────────────────────────────────────────────
                if self.progress.all_conquered(self.config.mastery_threshold):
                    logger.info("\n*** ALL CURRENT DOMAINS CONQUERED! ***")
                    
                    conquered = [r.domain for r in domain_reports]
                    candidates = self.expander.get_expansion_candidates(conquered)
                    
                    if candidates and self.continuous:
                        # In continuous mode: ADD new domains and keep going!
                        expansion_count += 1
                        new_domains = [c["name"].lower().replace(" ", "_") for c in candidates[:3]]
                        
                        logger.info(f"\n  Auto-expanding to {len(new_domains)} new domains:")
                        for c in candidates[:3]:
                            logger.info(f"    → {c['name']}: {c['description']}")
                        
                        self.config.domains.extend(new_domains)
                        
                        self.memory.log_action(
                            "expand", "auto_expansion",
                            details={
                                "expansion_count": expansion_count,
                                "new_domains": new_domains,
                                "total_domains": self.config.domains,
                            },
                        )
                        
                        # Save expansion plan
                        expansion_path = Path(self.config.log_dir) / f"expansion_{expansion_count}.json"
                        with open(expansion_path, "w") as f:
                            json.dump({
                                "expansion_count": expansion_count,
                                "conquered": conquered,
                                "new_domains": new_domains,
                                "candidates": candidates,
                                "timestamp": datetime.now().isoformat(),
                            }, f, indent=2)
                        
                        logger.info(f"  Continuing with {len(self.config.domains)} domains...")
                    elif not self.continuous:
                        logger.info("  All conquered in fixed mode — stopping.")
                        break
                    else:
                        logger.info("  No more expansion candidates. Mission complete!")
                        break
        
        except KeyboardInterrupt:
            logger.info("\n[GymCoach] Interrupted by user. Saving state...")
            self.memory.log_action(
                "system", "keyboard_interrupt",
                details={"iteration": iteration, "model": current_model},
            )
            self.memory.save_snapshot(iteration)
        
        except Exception as e:
            logger.error(f"\n[GymCoach] Unexpected error: {e}")
            import traceback
            tb = traceback.format_exc()
            self.memory.log_error(
                error_type="training_crash",
                description=str(e),
                severity=5,
                stack_trace=tb,
                model_path=current_model,
                iteration=iteration,
            )
            self.memory.save_snapshot(iteration)
            raise
        
        # Final summary
        self._print_final_summary()
    
    def _evaluate_all_domains(self, model_path: str) -> list[DomainReport]:
        """Evaluate model on all active domains.
        
        Logs all results and trajectories to Training Memory.
        """
        reports = []
        
        for domain in self.config.domains:
            try:
                report, result_dicts = self._evaluate_domain(model_path, domain)
                reports.append(report)
                
                # Log trajectories to memory
                self.memory.log_trajectories_from_results(
                    result_dicts, domain=domain,
                    iteration=self.memory.actions._current_iteration,
                    model_path=model_path,
                )
            except Exception as e:
                logger.error(f"  Failed to evaluate {domain}: {e}")
                import traceback
                self.memory.log_error(
                    error_type="evaluation_crash",
                    domain=domain,
                    description=str(e),
                    severity=4,
                    stack_trace=traceback.format_exc(),
                    model_path=model_path,
                )
                # Create a dummy report for failed evaluation
                reports.append(DomainReport(
                    domain=domain,
                    action_score=0.0,
                    reward_score=0.0,
                    num_tasks=0,
                    num_passed=0,
                    num_failed=0,
                    mastery=MasteryLevel.NOVICE,
                    timestamp=datetime.now().isoformat(),
                ))
        
        return reports
    
    def _evaluate_domain(self, model_path: str, domain: str) -> tuple:
        """Evaluate model on a single domain.
        
        Returns:
            (DomainReport, list[dict]) — report and raw result dicts for memory logging
        """
        from bioagents.evaluation.agent_runner import AgentRunner, RunConfig
        
        run_config = RunConfig(
            model_name_or_path=model_path,
            backend=self.config.backend,
            domain=domain,
            max_turns=15,
            temperature=0.1,  # Low for evaluation
            log_dir=str(Path(self.config.log_dir) / "eval"),
        )
        
        runner = AgentRunner(run_config)
        runner.load_model()
        task_results = runner.run_all_tasks()
        
        # Convert TaskResults to dicts for analysis
        result_dicts = []
        for r in task_results:
            result_dicts.append({
                "task_id": r.task_id,
                "action_score": r.action_score,
                "final_reward": r.final_reward,
                "total_turns": r.total_turns,
                "completed": r.completed,
                "turns": [
                    {
                        "turn_idx": t.turn_idx,
                        "raw_output": t.raw_output,
                        "parsed_tool_call": t.parsed_tool_call,
                        "tool_response": t.tool_response,
                        "is_final_answer": t.is_final_answer,
                    }
                    for t in r.turns
                ],
                "trajectory": r.trajectory,
            })
        
        # Clean up
        del runner
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        
        report = self.error_analyzer.analyze_domain(domain, result_dicts)
        return report, result_dicts
    
    def _train_iteration(
        self,
        model_path: str,
        iteration: int,
        training_plan: dict,
        generated_tasks: list[dict],
    ) -> Optional[str]:
        """Train the model for one iteration based on the training plan.
        
        Uses the existing SelfPlayLoop or GRPOTrainer depending on the plan.
        """
        output_dir = str(Path(self.config.output_dir) / f"iter_{iteration}")
        
        # Save training tasks
        tasks_path = Path(output_dir) / "training_tasks.json"
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tasks_path, "w") as f:
            json.dump(generated_tasks, f, indent=2, ensure_ascii=False)
        
        method = training_plan.get("training_method", "sft")
        
        if method == "sft":
            return self._train_sft(model_path, iteration, tasks_path, training_plan)
        elif method == "grpo":
            return self._train_grpo(model_path, iteration, tasks_path, training_plan)
        else:
            logger.warning(f"Unknown training method: {method}, skipping training")
            return None
    
    def _train_sft(
        self,
        model_path: str,
        iteration: int,
        tasks_path: Path,
        plan: dict,
    ) -> Optional[str]:
        """Run SFT training via existing SelfPlayLoop infrastructure."""
        from bioagents.gym.self_play import SelfPlayConfig, SelfPlayLoop
        
        focus_domains = plan.get("focus_domains", self.config.domains)
        
        sp_config = SelfPlayConfig(
            model_name_or_path=model_path,
            backend=self.config.backend,
            domains=focus_domains,
            tasks_per_domain=self.config.eval_tasks_per_domain,
            max_iterations=1,  # Single iteration
            learning_rate=plan.get("learning_rate", self.config.learning_rate),
            num_train_epochs=plan.get("epochs", self.config.train_epochs_per_iter),
            temperature=plan.get("temperature", 0.7),
            quality_threshold=0.5,
            min_trajectories_for_training=20,
            output_dir=str(Path(self.config.output_dir) / f"iter_{iteration}_sft"),
            log_dir=str(Path(self.config.log_dir) / f"iter_{iteration}_sft"),
            trajectory_dir=str(Path(self.config.generated_data_dir) / f"iter_{iteration}_traj"),
        )
        
        try:
            loop = SelfPlayLoop(sp_config)
            loop.run()
            
            # Return the path to the new model
            merged_path = str(Path(sp_config.output_dir) / "iter_1_sft" / "merged")
            if Path(merged_path).exists():
                return merged_path
        except Exception as e:
            logger.error(f"SFT training failed: {e}")
        
        return None
    
    def _train_grpo(
        self,
        model_path: str,
        iteration: int,
        tasks_path: Path,
        plan: dict,
    ) -> Optional[str]:
        """Run GRPO training (standard or FairGRPO based on plan)."""
        use_fairness = plan.get("use_fairness", False) or plan.get("training_method") == "fair_grpo"
        
        try:
            if use_fairness:
                return self._train_fair_grpo(model_path, iteration, tasks_path, plan)
            
            from bioagents.training.grpo_trainer import BioAgentGRPOConfig, train
            
            grpo_config = BioAgentGRPOConfig(
                model_name_or_path=model_path,
                domain=plan.get("focus_domains", ["medical_qa"])[0],
                tasks_path=str(tasks_path),
                output_dir=str(Path(self.config.output_dir) / f"iter_{iteration}_grpo"),
                num_train_epochs=plan.get("epochs", self.config.train_epochs_per_iter),
                learning_rate=plan.get("learning_rate", self.config.learning_rate),
                temperature=plan.get("temperature", 0.7),
                beta=0.04,
                run_name=f"gym_coach_iter_{iteration}",
                use_wandb=False,
            )
            
            trainer = train(grpo_config)
            
            final_path = str(Path(grpo_config.output_dir) / "final")
            if Path(final_path).exists():
                return final_path
        except Exception as e:
            logger.error(f"GRPO training failed: {e}")
        
        return None
    
    def _train_fair_grpo(
        self,
        model_path: str,
        iteration: int,
        tasks_path: Path,
        plan: dict,
    ) -> Optional[str]:
        """Run FairGRPO training with demographic-aware reward weighting.
        
        Integrates the FairGRPO framework (arXiv:2510.19893) into the
        GymCoach training loop. Automatically activated when:
          - plan["use_fairness"] is True
          - plan["training_method"] == "fair_grpo"
          - Fairness gap is detected above threshold during evaluation
        """
        try:
            from bioagents.training.grpo_trainer import FairGRPOConfig, train_fair_grpo
            
            fair_config = FairGRPOConfig(
                model_name_or_path=model_path,
                domain=plan.get("focus_domains", ["medical_qa"])[0],
                tasks_path=str(tasks_path),
                output_dir=str(Path(self.config.output_dir) / f"iter_{iteration}_fair_grpo"),
                num_train_epochs=plan.get("epochs", self.config.train_epochs_per_iter),
                learning_rate=plan.get("learning_rate", self.config.learning_rate),
                temperature=plan.get("temperature", 0.7),
                beta=0.04,
                run_name=f"gym_coach_iter_{iteration}_fair",
                use_wandb=False,
                # FairGRPO-specific
                fairness_enabled=True,
                fairness_weight=plan.get("fairness_weight", 0.1),
                alpha_repr=plan.get("alpha_repr", 0.5),
                alpha_perf=plan.get("alpha_perf", 0.5),
                max_fairness_gap=plan.get("max_fairness_gap", 0.15),
            )
            
            trainer = train_fair_grpo(fair_config)
            
            # Log fairness results to GymCoach memory
            from bioagents.evaluation.grpo_rewards import get_fairness_tracker
            tracker = get_fairness_tracker()
            fairness_summary = tracker.get_summary()
            
            self.memory.log_action(
                "fair_grpo_complete",
                f"iter_{iteration}",
                details={
                    "fairness_gaps": fairness_summary.get("fairness_gaps", {}),
                    "group_stats": {
                        k: v for k, v in fairness_summary.items()
                        if k != "fairness_gaps"
                    },
                },
            )
            
            final_path = str(Path(fair_config.output_dir) / "final")
            if Path(final_path).exists():
                return final_path
        except Exception as e:
            logger.error(f"FairGRPO training failed: {e}")
        
        return None
    
    def _print_final_summary(self):
        """Print the final coaching summary."""
        logger.info("\n" + "=" * 70)
        logger.info("  GymCoach — FINAL SUMMARY")
        logger.info("=" * 70)
        
        if self.progress.history:
            first = self.progress.history[0]
            last = self.progress.history[-1]
            
            logger.info(f"  Total iterations: {len(self.progress.history)}")
            logger.info(f"  Phase progression: {first['phase']} → {last['phase']}")
            logger.info()
            
            logger.info("  Domain Progress:")
            for domain in sorted(last.get("domains", {}).keys()):
                first_data = first.get("domains", {}).get(domain, {})
                last_data = last.get("domains", {}).get(domain, {})
                
                first_score = first_data.get("action_score", 0.0)
                last_score = last_data.get("action_score", 0.0)
                mastery = last_data.get("mastery", "unknown")
                
                delta = last_score - first_score
                logger.info(f"    {domain:<25} {first_score:.1%} → {last_score:.1%} "
                          f"(+{delta:.1%})  [{mastery}]")
            
            # Check if conquered
            if self.progress.all_conquered():
                logger.info()
                logger.info("  *** MISSION COMPLETE: ALL DOMAINS CONQUERED ***")
                logger.info("  The model is ready for domain expansion.")
        
        logger.info("=" * 70)


# ============================================================
# 8. CLI Entry Point
# ============================================================

def run_gym_coach(config_path: Optional[str] = None, **kwargs):
    """Entry point for GymCoach."""
    if config_path:
        import yaml
        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)
        config = CoachConfig(**cfg_dict)
    else:
        config = CoachConfig(**kwargs)
    
    coach = GymCoach(config)
    coach.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Healthcare AI GYM Coach")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--iterations", type=int, default=20, help="Max iterations")
    parser.add_argument("--backend", default="transformers", help="Inference backend")
    parser.add_argument("--mastery-threshold", type=float, default=0.90, help="Score to conquer a domain")
    
    args = parser.parse_args()
    
    kwargs = {
        "model_name_or_path": args.model,
        "max_iterations": args.iterations,
        "backend": args.backend,
        "mastery_threshold": args.mastery_threshold,
    }
    
    run_gym_coach(config_path=args.config, **kwargs)
