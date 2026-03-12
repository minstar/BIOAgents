"""Curriculum Learning Scheduler for Healthcare AI GYM.

Implements progressive difficulty training inspired by:
- DiagGym (arXiv:2510.24654): Progressive diagnostic complexity
- MedAgentGym (ICLR 2026): Scalable task difficulty progression
- Clinical education theory: See-one, Do-one, Teach-one

The curriculum scheduler controls which tasks are presented to the agent
at each training stage, progressively increasing difficulty as the agent
demonstrates competence at each level.

Difficulty Levels:
    Level 1 (Foundation):   Simple MCQA, single-tool tasks, explicit answers
    Level 2 (Developing):   Multi-step reasoning, 2-3 tool calls, evidence search
    Level 3 (Competent):    Complex cases, 4-6 tool calls, differential diagnosis
    Level 4 (Proficient):   Cross-domain, safety-critical, multi-phase pathways
    Level 5 (Expert):       Adversarial cases, cognitive bias tests, ambiguous presentations

The scheduler uses a competence-based gating mechanism: the agent must
achieve a minimum score threshold at the current level before unlocking
the next level. This prevents catastrophic failure from premature exposure
to hard tasks.

Usage:
    scheduler = CurriculumScheduler(config)
    scheduler.update_competence("clinical_diagnosis", level=2, score=0.72)
    tasks = scheduler.select_tasks("clinical_diagnosis", pool=all_tasks, n=10)
"""

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ============================================================
# Difficulty Classification
# ============================================================


class DifficultyLevel(IntEnum):
    """Progressive difficulty levels for curriculum learning."""
    FOUNDATION = 1    # Simple MCQA, single tool, explicit
    DEVELOPING = 2    # Multi-step, 2-3 tools, evidence search
    COMPETENT = 3     # Complex cases, 4-6 tools, DDx
    PROFICIENT = 4    # Cross-domain, safety-critical
    EXPERT = 5        # Adversarial, bias tests, ambiguous


# Mapping from task metadata difficulty strings to levels
DIFFICULTY_MAPPING = {
    "easy": DifficultyLevel.FOUNDATION,
    "simple": DifficultyLevel.FOUNDATION,
    "basic": DifficultyLevel.FOUNDATION,
    "medium": DifficultyLevel.DEVELOPING,
    "moderate": DifficultyLevel.COMPETENT,
    "hard": DifficultyLevel.PROFICIENT,
    "complex": DifficultyLevel.PROFICIENT,
    "expert": DifficultyLevel.EXPERT,
    "adversarial": DifficultyLevel.EXPERT,
}


def classify_task_difficulty(task: dict) -> DifficultyLevel:
    """Classify a task's difficulty level based on its metadata.

    Heuristic-based classification using:
    - Explicit difficulty label (if present)
    - Number of expected tool calls
    - Task type (MCQA vs open-ended)
    - Domain complexity
    - Presence of safety-critical elements
    """
    # Check explicit difficulty label
    description = task.get("description", {})
    if isinstance(description, dict):
        explicit_diff = description.get("difficulty", "").lower()
    elif isinstance(description, str):
        explicit_diff = ""
    else:
        explicit_diff = ""

    if explicit_diff in DIFFICULTY_MAPPING:
        return DIFFICULTY_MAPPING[explicit_diff]

    # Heuristic scoring
    score = 0

    # Number of expected tool calls
    eval_criteria = task.get("evaluation_criteria", {})
    actions = eval_criteria.get("actions", [])
    n_tools = len(actions)
    if n_tools <= 1:
        score += 1
    elif n_tools <= 3:
        score += 2
    elif n_tools <= 6:
        score += 3
    else:
        score += 4

    # MCQA is simpler than open-ended
    if task.get("options"):
        score -= 1

    # Domain complexity
    domain = task.get("domain", "")
    domain_complexity = {
        "medical_qa": 0,
        "drug_interaction": 1,
        "clinical_diagnosis": 1,
        "ehr_management": 2,
        "triage_emergency": 2,
        "psychiatry": 2,
        "obstetrics": 2,
        "visual_diagnosis": 3,
        "radiology_report": 3,
        "cross_domain": 4,
    }
    score += domain_complexity.get(domain, 1)

    # Safety-critical elements
    safety_keywords = ["emergency", "critical", "stat", "urgent", "contraindicated"]
    task_text = json.dumps(task.get("description", "")).lower()
    if any(kw in task_text for kw in safety_keywords):
        score += 1

    # Map score to difficulty level
    if score <= 1:
        return DifficultyLevel.FOUNDATION
    elif score <= 3:
        return DifficultyLevel.DEVELOPING
    elif score <= 5:
        return DifficultyLevel.COMPETENT
    elif score <= 7:
        return DifficultyLevel.PROFICIENT
    else:
        return DifficultyLevel.EXPERT


# ============================================================
# Competence Tracker
# ============================================================


@dataclass
class DomainCompetence:
    """Tracks an agent's competence at each difficulty level for a domain."""
    domain: str
    level_scores: Dict[int, List[float]] = field(
        default_factory=lambda: {lvl: [] for lvl in DifficultyLevel}
    )
    current_level: int = 1
    unlocked_level: int = 1
    total_tasks_completed: int = 0
    last_updated: str = ""

    def mean_score(self, level: int) -> float:
        """Get mean score for a specific difficulty level."""
        scores = self.level_scores.get(level, [])
        return sum(scores) / max(len(scores), 1) if scores else 0.0

    def tasks_at_level(self, level: int) -> int:
        """Number of tasks completed at a specific level."""
        return len(self.level_scores.get(level, []))

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "current_level": self.current_level,
            "unlocked_level": self.unlocked_level,
            "total_tasks_completed": self.total_tasks_completed,
            "level_summaries": {
                lvl: {
                    "count": len(scores),
                    "mean": round(sum(scores) / max(len(scores), 1), 4) if scores else 0.0,
                }
                for lvl, scores in self.level_scores.items()
            },
            "last_updated": self.last_updated,
        }


# ============================================================
# Curriculum Configuration
# ============================================================


@dataclass
class CurriculumConfig:
    """Configuration for the curriculum learning scheduler."""
    # Competence gating thresholds (score needed to unlock next level)
    level_thresholds: Dict[int, float] = field(default_factory=lambda: {
        1: 0.50,   # Foundation → Developing: need 50% at level 1
        2: 0.55,   # Developing → Competent: need 55% at level 2
        3: 0.60,   # Competent → Proficient: need 60% at level 3
        4: 0.65,   # Proficient → Expert: need 65% at level 4
    })

    # Minimum tasks at each level before considering promotion
    min_tasks_per_level: int = 5

    # Task mixing strategy
    # When training at level N, what % of tasks come from:
    #   current level vs lower levels (for reinforcement)
    current_level_ratio: float = 0.60     # 60% at current level
    review_ratio: float = 0.25            # 25% from lower levels (review)
    challenge_ratio: float = 0.15         # 15% from next level (stretch goals)

    # Anti-regression: if score drops below this at a level, revisit that level
    regression_threshold: float = 0.40

    # Warm-up: number of cycles at level 1 before allowing promotion
    warmup_cycles: int = 2

    # Level demotion: allow going back to lower levels if struggling
    allow_demotion: bool = True
    demotion_threshold: float = 0.30  # Demote if consistently below this

    # Persistence path
    state_file: str = "logs/curriculum_state.json"


# ============================================================
# Curriculum Scheduler
# ============================================================


class CurriculumScheduler:
    """Manages progressive difficulty training for autonomous agents.

    Key features:
    1. Competence-gated level progression
    2. Mixed difficulty sampling (current + review + challenge)
    3. Anti-regression detection and remediation
    4. Per-domain, per-agent difficulty tracking
    5. Persistence across training sessions
    """

    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()
        # {agent_id: {domain: DomainCompetence}}
        self._competence: Dict[str, Dict[str, DomainCompetence]] = {}
        self._cycle_count: Dict[str, int] = {}

    # ── Competence Tracking ──────────────────────────────

    def get_competence(self, agent_id: str, domain: str) -> DomainCompetence:
        """Get or create competence tracker for an agent-domain pair."""
        if agent_id not in self._competence:
            self._competence[agent_id] = {}
        if domain not in self._competence[agent_id]:
            self._competence[agent_id][domain] = DomainCompetence(domain=domain)
        return self._competence[agent_id][domain]

    def update_competence(
        self,
        agent_id: str,
        domain: str,
        level: int,
        score: float,
    ) -> dict:
        """Record a task result and check for level promotion/demotion.

        Args:
            agent_id: Agent identifier
            domain: Medical domain
            level: Difficulty level of the completed task
            score: Reward score [0, 1]

        Returns:
            Dict with promotion/demotion status and current level
        """
        comp = self.get_competence(agent_id, domain)
        comp.level_scores[level].append(score)
        comp.total_tasks_completed += 1
        comp.last_updated = datetime.now().isoformat()

        result = {
            "agent_id": agent_id,
            "domain": domain,
            "level": level,
            "score": score,
            "promoted": False,
            "demoted": False,
            "current_level": comp.current_level,
            "unlocked_level": comp.unlocked_level,
        }

        # Check for promotion
        current = comp.current_level
        if current < DifficultyLevel.EXPERT:
            threshold = self.config.level_thresholds.get(current, 0.60)
            n_tasks = comp.tasks_at_level(current)
            mean = comp.mean_score(current)

            if n_tasks >= self.config.min_tasks_per_level and mean >= threshold:
                # Check warm-up cycles
                agent_cycles = self._cycle_count.get(agent_id, 0)
                if current == 1 and agent_cycles < self.config.warmup_cycles:
                    logger.info(
                        f"[Curriculum] {agent_id}/{domain}: Level {current} "
                        f"threshold met but in warmup ({agent_cycles}/{self.config.warmup_cycles})"
                    )
                else:
                    comp.current_level = current + 1
                    comp.unlocked_level = max(comp.unlocked_level, comp.current_level)
                    result["promoted"] = True
                    result["current_level"] = comp.current_level
                    result["unlocked_level"] = comp.unlocked_level
                    logger.info(
                        f"[Curriculum] {agent_id}/{domain}: "
                        f"PROMOTED to Level {comp.current_level} "
                        f"(mean={mean:.3f} >= {threshold:.3f}, "
                        f"tasks={n_tasks})"
                    )

        # Check for demotion (anti-regression)
        if self.config.allow_demotion and comp.current_level > 1:
            recent_scores = comp.level_scores.get(comp.current_level, [])[-5:]
            if len(recent_scores) >= 3:
                recent_mean = sum(recent_scores) / len(recent_scores)
                if recent_mean < self.config.demotion_threshold:
                    comp.current_level -= 1
                    result["demoted"] = True
                    result["current_level"] = comp.current_level
                    logger.warning(
                        f"[Curriculum] {agent_id}/{domain}: "
                        f"DEMOTED to Level {comp.current_level} "
                        f"(recent_mean={recent_mean:.3f} < {self.config.demotion_threshold})"
                    )

        return result

    def increment_cycle(self, agent_id: str) -> None:
        """Increment the cycle counter for an agent."""
        self._cycle_count[agent_id] = self._cycle_count.get(agent_id, 0) + 1

    # ── Task Selection ───────────────────────────────────

    def select_tasks(
        self,
        agent_id: str,
        domain: str,
        task_pool: List[dict],
        n: int = 10,
    ) -> List[dict]:
        """Select tasks with curriculum-aware difficulty mixing.

        Classifies all tasks in the pool by difficulty, then samples
        according to the current level and mixing ratios.

        Args:
            agent_id: Agent identifier
            domain: Medical domain
            task_pool: All available tasks for the domain
            n: Number of tasks to select

        Returns:
            List of selected tasks (difficulty-mixed)
        """
        if not task_pool:
            return []

        comp = self.get_competence(agent_id, domain)
        current_level = comp.current_level

        # Classify tasks by difficulty
        level_buckets: Dict[int, List[dict]] = {
            lvl: [] for lvl in DifficultyLevel
        }
        for task in task_pool:
            level = classify_task_difficulty(task)
            level_buckets[level].append(task)

        # Calculate target counts for each difficulty range
        n_current = max(1, round(n * self.config.current_level_ratio))
        n_review = max(0, round(n * self.config.review_ratio))
        n_challenge = max(0, round(n * self.config.challenge_ratio))

        # Adjust if total doesn't match
        total_planned = n_current + n_review + n_challenge
        if total_planned < n:
            n_current += n - total_planned

        selected = []

        # 1. Current level tasks
        current_tasks = level_buckets.get(current_level, [])
        selected.extend(self._sample(current_tasks, n_current))

        # 2. Review tasks (lower levels)
        review_tasks = []
        for lvl in range(1, current_level):
            review_tasks.extend(level_buckets.get(lvl, []))
        selected.extend(self._sample(review_tasks, n_review))

        # 3. Challenge tasks (next level, stretch goals)
        next_level = min(current_level + 1, DifficultyLevel.EXPERT)
        challenge_tasks = level_buckets.get(next_level, [])
        selected.extend(self._sample(challenge_tasks, n_challenge))

        # Fill remaining slots if needed (from any available level)
        if len(selected) < n:
            all_remaining = [t for t in task_pool if t not in selected]
            selected.extend(self._sample(all_remaining, n - len(selected)))

        # Shuffle to avoid predictable ordering
        random.shuffle(selected)

        logger.info(
            f"[Curriculum] {agent_id}/{domain}: "
            f"Selected {len(selected)} tasks at Level {current_level} "
            f"(current={n_current}, review={n_review}, challenge={n_challenge})"
        )

        return selected[:n]

    def _sample(self, pool: List[dict], n: int) -> List[dict]:
        """Sample n items from pool without replacement."""
        if not pool or n <= 0:
            return []
        return random.sample(pool, min(n, len(pool)))

    # ── Status & Reporting ───────────────────────────────

    def get_agent_status(self, agent_id: str) -> dict:
        """Get comprehensive curriculum status for an agent."""
        domains = self._competence.get(agent_id, {})
        status = {
            "agent_id": agent_id,
            "total_cycles": self._cycle_count.get(agent_id, 0),
            "domains": {},
        }
        for domain, comp in domains.items():
            status["domains"][domain] = comp.to_dict()
        return status

    def get_all_status(self) -> dict:
        """Get curriculum status for all agents."""
        return {
            agent_id: self.get_agent_status(agent_id)
            for agent_id in self._competence
        }

    def get_promotion_readiness(self, agent_id: str, domain: str) -> dict:
        """Check how close an agent is to promotion in a domain."""
        comp = self.get_competence(agent_id, domain)
        current = comp.current_level

        if current >= DifficultyLevel.EXPERT:
            return {
                "current_level": current,
                "at_max": True,
                "readiness": 1.0,
            }

        threshold = self.config.level_thresholds.get(current, 0.60)
        n_tasks = comp.tasks_at_level(current)
        mean = comp.mean_score(current)

        tasks_ready = n_tasks >= self.config.min_tasks_per_level
        score_ready = mean >= threshold

        # Readiness as a percentage
        task_pct = min(1.0, n_tasks / max(self.config.min_tasks_per_level, 1))
        score_pct = min(1.0, mean / max(threshold, 0.01)) if threshold > 0 else 1.0
        readiness = task_pct * 0.3 + score_pct * 0.7

        return {
            "current_level": current,
            "at_max": False,
            "threshold": threshold,
            "current_mean": round(mean, 4),
            "tasks_completed": n_tasks,
            "tasks_needed": self.config.min_tasks_per_level,
            "tasks_ready": tasks_ready,
            "score_ready": score_ready,
            "readiness": round(readiness, 4),
        }

    # ── Persistence ──────────────────────────────────────

    def save_state(self, path: Optional[str] = None) -> None:
        """Save curriculum state to disk."""
        filepath = Path(path or self.config.state_file)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "timestamp": datetime.now().isoformat(),
            "cycle_counts": self._cycle_count,
            "agents": {},
        }
        for agent_id, domains in self._competence.items():
            state["agents"][agent_id] = {}
            for domain, comp in domains.items():
                state["agents"][agent_id][domain] = {
                    "current_level": comp.current_level,
                    "unlocked_level": comp.unlocked_level,
                    "total_tasks_completed": comp.total_tasks_completed,
                    "level_scores": {
                        str(k): v for k, v in comp.level_scores.items()
                    },
                    "last_updated": comp.last_updated,
                }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"[Curriculum] State saved to {filepath}")

    def load_state(self, path: Optional[str] = None) -> bool:
        """Load curriculum state from disk.

        Returns True if state was loaded, False if file doesn't exist.
        """
        filepath = Path(path or self.config.state_file)
        if not filepath.exists():
            return False

        try:
            with open(filepath) as f:
                state = json.load(f)

            self._cycle_count = state.get("cycle_counts", {})

            for agent_id, domains in state.get("agents", {}).items():
                for domain, data in domains.items():
                    comp = self.get_competence(agent_id, domain)
                    comp.current_level = data.get("current_level", 1)
                    comp.unlocked_level = data.get("unlocked_level", 1)
                    comp.total_tasks_completed = data.get("total_tasks_completed", 0)
                    comp.last_updated = data.get("last_updated", "")
                    for k, v in data.get("level_scores", {}).items():
                        comp.level_scores[int(k)] = v

            logger.info(f"[Curriculum] State loaded from {filepath}")
            return True
        except Exception as e:
            logger.warning(f"[Curriculum] Failed to load state: {e}")
            return False
