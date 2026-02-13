"""Training Memory System — Complete Recording & Error Prevention for Healthcare AI GYM.

Records ALL actions, trajectories, and errors throughout training so that:
1. Recurring errors are detected and flagged BEFORE they happen again
2. Every training decision is traceable back to what caused it
3. The system learns from its own mistakes across iterations

Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │                    Training Memory System                       │
    │                                                                 │
    │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
    │  │ ActionLog    │  │ ErrorMemory  │  │ TrajectoryStore       │ │
    │  │ every tool   │  │ categorized  │  │ full multi-turn       │ │
    │  │ call, param  │  │ dedup'd      │  │ conversations         │ │
    │  │ result       │  │ patterns     │  │ w/ rewards            │ │
    │  └──────┬──────┘  └──────┬───────┘  └───────────┬───────────┘ │
    │         │                │                       │              │
    │         └────────────────┼───────────────────────┘              │
    │                          ▼                                      │
    │               ┌──────────────────┐                              │
    │               │ PatternDetector  │                              │
    │               │ finds recurring  │                              │
    │               │ failures & gives │                              │
    │               │ recommendations  │                              │
    │               └──────────────────┘                              │
    └────────────────────────────────────────────────────────────────┘

Usage:
    memory = TrainingMemory("logs/training_memory")
    
    # Log actions
    memory.log_action("evaluate", domain="clinical_diagnosis", details={...})
    
    # Log errors
    memory.log_error(ErrorRecord(
        error_type="safety_violation", domain="triage",
        task_id="triage_001", description="Missed critical allergy",
        context={"patient_allergies": ["penicillin"]},
    ))
    
    # Log full trajectories
    memory.log_trajectory(trajectory_data)
    
    # Get recommendations before training
    recs = memory.get_recommendations("clinical_diagnosis")
    
    # Check for recurring errors
    patterns = memory.detect_recurring_patterns()
"""

import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


# ============================================================
# 1. Data Structures
# ============================================================

@dataclass
class ActionRecord:
    """Records a single action taken during training/evaluation."""
    timestamp: str
    iteration: int
    phase: str                    # "evaluate", "analyze", "generate", "train", "expand"
    action_type: str              # "tool_call", "model_response", "reward_compute", etc.
    domain: str = ""
    task_id: str = ""
    details: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass
class ErrorRecord:
    """Records an error that occurred during training/evaluation."""
    timestamp: str = ""
    iteration: int = 0
    error_type: str = ""          # "safety_violation", "tool_selection", "training_crash", etc.
    domain: str = ""
    task_id: str = ""
    description: str = ""
    severity: int = 1             # 1-5
    context: dict = field(default_factory=dict)     # full context when error occurred
    stack_trace: str = ""
    model_path: str = ""
    resolution: str = ""          # how it was resolved (filled later)
    fingerprint: str = ""         # hash for deduplication
    recurrence_count: int = 1     # how many times this error has occurred
    
    def compute_fingerprint(self) -> str:
        """Compute a fingerprint for deduplication of similar errors."""
        key = f"{self.error_type}:{self.domain}:{self.description[:100]}"
        self.fingerprint = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.fingerprint


@dataclass
class TrajectoryRecord:
    """Records a complete multi-turn trajectory."""
    timestamp: str = ""
    iteration: int = 0
    domain: str = ""
    task_id: str = ""
    model_path: str = ""
    turns: list = field(default_factory=list)         # list of turn dicts
    total_turns: int = 0
    action_score: float = 0.0
    reward_breakdown: dict = field(default_factory=dict)  # {accuracy, format, process, safety}
    tools_used: list = field(default_factory=list)    # ordered list of tool names
    final_answer: str = ""
    completed: bool = False
    failure_types: list = field(default_factory=list)  # if failed, what types
    duration_ms: float = 0.0


@dataclass
class RecurringPattern:
    """A detected recurring error pattern."""
    fingerprint: str
    error_type: str
    domain: str
    description: str
    occurrences: int
    first_seen: str              # timestamp
    last_seen: str               # timestamp
    iterations_seen: list = field(default_factory=list)
    severity: int = 1
    recommendation: str = ""
    resolved: bool = False


# ============================================================
# 2. Action Logger
# ============================================================

class ActionLogger:
    """Logs every action taken during the training process."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir / "actions"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_iteration = 0
        self._session_actions: list[ActionRecord] = []
    
    def set_iteration(self, iteration: int):
        """Set the current iteration for logging."""
        self._current_iteration = iteration
    
    def log(
        self,
        phase: str,
        action_type: str,
        domain: str = "",
        task_id: str = "",
        details: dict = None,
        duration_ms: float = 0.0,
        success: bool = True,
        error_message: str = "",
    ) -> ActionRecord:
        """Log an action."""
        record = ActionRecord(
            timestamp=datetime.now().isoformat(),
            iteration=self._current_iteration,
            phase=phase,
            action_type=action_type,
            domain=domain,
            task_id=task_id,
            details=details or {},
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
        )
        self._session_actions.append(record)
        
        # Write incrementally to avoid data loss
        self._append_to_log(record)
        return record
    
    def _append_to_log(self, record: ActionRecord):
        """Append a single action to the log file."""
        log_file = self.log_dir / f"iter_{record.iteration:04d}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    
    def get_session_summary(self) -> dict:
        """Get summary of actions in current session."""
        if not self._session_actions:
            return {"total_actions": 0}
        
        by_phase = Counter(a.phase for a in self._session_actions)
        by_type = Counter(a.action_type for a in self._session_actions)
        failures = [a for a in self._session_actions if not a.success]
        total_time = sum(a.duration_ms for a in self._session_actions)
        
        return {
            "total_actions": len(self._session_actions),
            "by_phase": dict(by_phase),
            "by_type": dict(by_type),
            "failures": len(failures),
            "total_time_ms": total_time,
            "iteration": self._current_iteration,
        }
    
    def get_actions_for_iteration(self, iteration: int) -> list[dict]:
        """Load all actions for a specific iteration."""
        log_file = self.log_dir / f"iter_{iteration:04d}.jsonl"
        if not log_file.exists():
            return []
        actions = []
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    actions.append(json.loads(line))
        return actions


# ============================================================
# 3. Error Memory
# ============================================================

class ErrorMemory:
    """Stores and analyzes errors across all training iterations.
    
    Key capabilities:
    - Deduplication via fingerprinting
    - Recurrence tracking
    - Pattern detection
    - Resolution tracking
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir / "errors"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._errors: list[ErrorRecord] = []
        self._fingerprint_index: dict[str, list[int]] = defaultdict(list)  # fp -> [indices]
        self._load_errors()
    
    def _load_errors(self):
        """Load existing errors from disk."""
        error_file = self.log_dir / "all_errors.jsonl"
        if not error_file.exists():
            return
        with open(error_file) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    data = json.loads(line)
                    record = ErrorRecord(**data)
                    self._errors.append(record)
                    if record.fingerprint:
                        self._fingerprint_index[record.fingerprint].append(i)
    
    def log_error(self, error: ErrorRecord) -> ErrorRecord:
        """Log an error and check for recurrence."""
        if not error.timestamp:
            error.timestamp = datetime.now().isoformat()
        error.compute_fingerprint()
        
        # Check recurrence
        if error.fingerprint in self._fingerprint_index:
            indices = self._fingerprint_index[error.fingerprint]
            error.recurrence_count = len(indices) + 1
            logger.warning(
                f"[TrainingMemory] RECURRING ERROR detected ({error.recurrence_count}x): "
                f"{error.error_type} in {error.domain} — {error.description[:80]}"
            )
        
        idx = len(self._errors)
        self._errors.append(error)
        self._fingerprint_index[error.fingerprint].append(idx)
        
        # Persist
        self._append_error(error)
        return error
    
    def _append_error(self, error: ErrorRecord):
        """Append error to persistent storage."""
        error_file = self.log_dir / "all_errors.jsonl"
        with open(error_file, "a") as f:
            f.write(json.dumps(asdict(error), ensure_ascii=False) + "\n")
    
    def get_recurring_errors(self, min_occurrences: int = 2) -> list[RecurringPattern]:
        """Find error patterns that recur across iterations."""
        patterns = []
        
        for fp, indices in self._fingerprint_index.items():
            if len(indices) < min_occurrences:
                continue
            
            errors = [self._errors[i] for i in indices]
            iterations_seen = sorted(set(e.iteration for e in errors))
            
            pattern = RecurringPattern(
                fingerprint=fp,
                error_type=errors[0].error_type,
                domain=errors[0].domain,
                description=errors[0].description,
                occurrences=len(indices),
                first_seen=errors[0].timestamp,
                last_seen=errors[-1].timestamp,
                iterations_seen=iterations_seen,
                severity=max(e.severity for e in errors),
                recommendation=self._generate_recommendation(errors),
            )
            patterns.append(pattern)
        
        # Sort by severity × occurrences
        patterns.sort(key=lambda p: p.severity * p.occurrences, reverse=True)
        return patterns
    
    def _generate_recommendation(self, errors: list[ErrorRecord]) -> str:
        """Generate a recommendation based on recurring error pattern."""
        error_type = errors[0].error_type
        domain = errors[0].domain
        count = len(errors)
        
        recommendations = {
            "safety_violation": (
                f"Safety violations in {domain} occurred {count}x. "
                f"Increase safety reward weight, add more safety-focused training data, "
                f"and consider explicit safety chain-of-thought prompting."
            ),
            "tool_selection": (
                f"Tool selection errors in {domain} occurred {count}x. "
                f"Add explicit tool-use demonstration in SFT data. "
                f"Consider tool-specific reward shaping."
            ),
            "premature_stop": (
                f"Premature stops in {domain} occurred {count}x. "
                f"Increase minimum turn requirement in training. "
                f"Add process reward for thorough investigation."
            ),
            "reasoning_error": (
                f"Reasoning errors in {domain} occurred {count}x despite correct process. "
                f"This likely indicates a knowledge gap. "
                f"Add medical knowledge QA pairs to SFT data for this domain."
            ),
            "format_error": (
                f"Format errors in {domain} occurred {count}x. "
                f"Increase format reward weight. "
                f"Add more structured output examples to SFT data."
            ),
            "knowledge_gap": (
                f"Knowledge gaps in {domain} occurred {count}x. "
                f"Augment training data with domain-specific medical knowledge. "
                f"Consider retrieval-augmented generation (RAG) for this domain."
            ),
            "over_investigation": (
                f"Over-investigation in {domain} occurred {count}x. "
                f"Add penalty for redundant tool calls. "
                f"Demonstrate efficient clinical workflows in SFT data."
            ),
            "parameter_error": (
                f"Parameter errors in {domain} occurred {count}x. "
                f"The model uses correct tools but wrong parameters. "
                f"Add more diverse parameter examples in training data."
            ),
            "guideline_noncompliance": (
                f"Guideline noncompliance in {domain} occurred {count}x. "
                f"Inject clinical guidelines into prompts. "
                f"Add guideline compliance reward."
            ),
            "training_crash": (
                f"Training crashes occurred {count}x. "
                f"Check GPU memory, learning rate, and data format. "
                f"Consider reducing batch size or using gradient checkpointing."
            ),
        }
        
        return recommendations.get(
            error_type,
            f"Error type '{error_type}' in {domain} occurred {count}x. "
            f"Review error context and add targeted training data."
        )
    
    def get_domain_error_summary(self, domain: str) -> dict:
        """Get error summary for a specific domain."""
        domain_errors = [e for e in self._errors if e.domain == domain]
        
        by_type = Counter(e.error_type for e in domain_errors)
        by_severity = Counter(e.severity for e in domain_errors)
        recurring = [
            fp for fp, indices in self._fingerprint_index.items()
            if len(indices) >= 2 and self._errors[indices[0]].domain == domain
        ]
        
        return {
            "total_errors": len(domain_errors),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "recurring_patterns": len(recurring),
            "most_common": by_type.most_common(3),
        }
    
    def get_all_errors_for_iteration(self, iteration: int) -> list[ErrorRecord]:
        """Get all errors from a specific iteration."""
        return [e for e in self._errors if e.iteration == iteration]
    
    def mark_resolved(self, fingerprint: str, resolution: str):
        """Mark a recurring error pattern as resolved."""
        if fingerprint in self._fingerprint_index:
            for idx in self._fingerprint_index[fingerprint]:
                self._errors[idx].resolution = resolution
            # Save resolution
            res_file = self.log_dir / "resolutions.jsonl"
            with open(res_file, "a") as f:
                f.write(json.dumps({
                    "fingerprint": fingerprint,
                    "resolution": resolution,
                    "timestamp": datetime.now().isoformat(),
                }, ensure_ascii=False) + "\n")


# ============================================================
# 4. Trajectory Store
# ============================================================

class TrajectoryStore:
    """Stores complete trajectories from evaluation and training."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir / "trajectories"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._stats = {"total": 0, "by_domain": {}, "by_outcome": {"pass": 0, "fail": 0}}
        self._load_stats()
    
    def _load_stats(self):
        """Load trajectory stats."""
        stats_file = self.log_dir / "stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                self._stats = json.load(f)
    
    def _save_stats(self):
        """Save trajectory stats."""
        stats_file = self.log_dir / "stats.json"
        with open(stats_file, "w") as f:
            json.dump(self._stats, f, indent=2)
    
    def store(self, trajectory: TrajectoryRecord):
        """Store a complete trajectory."""
        if not trajectory.timestamp:
            trajectory.timestamp = datetime.now().isoformat()
        
        # Save to iteration-specific file
        iter_dir = self.log_dir / f"iter_{trajectory.iteration:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        traj_file = iter_dir / f"{trajectory.domain}_{trajectory.task_id}.json"
        with open(traj_file, "w") as f:
            json.dump(asdict(trajectory), f, indent=2, ensure_ascii=False)
        
        # Update stats
        self._stats["total"] += 1
        self._stats["by_domain"][trajectory.domain] = (
            self._stats["by_domain"].get(trajectory.domain, 0) + 1
        )
        outcome = "pass" if trajectory.action_score >= 0.8 else "fail"
        self._stats["by_outcome"][outcome] = self._stats["by_outcome"].get(outcome, 0) + 1
        self._save_stats()
    
    def store_batch(self, trajectories: list[TrajectoryRecord]):
        """Store a batch of trajectories efficiently."""
        for traj in trajectories:
            self.store(traj)
    
    def get_failed_trajectories(
        self,
        domain: str = "",
        iteration: int = -1,
        min_score: float = 0.0,
        max_score: float = 0.5,
    ) -> list[dict]:
        """Get failed trajectories for analysis."""
        results = []
        
        if iteration >= 0:
            iter_dirs = [self.log_dir / f"iter_{iteration:04d}"]
        else:
            iter_dirs = sorted(self.log_dir.glob("iter_*"))
        
        for iter_dir in iter_dirs:
            if not iter_dir.is_dir():
                continue
            
            pattern = f"{domain}_*.json" if domain else "*.json"
            for traj_file in iter_dir.glob(pattern):
                try:
                    with open(traj_file) as f:
                        data = json.load(f)
                    score = data.get("action_score", 0.0)
                    if min_score <= score <= max_score:
                        results.append(data)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return results
    
    def get_tool_usage_stats(self, iteration: int = -1) -> dict:
        """Get tool usage statistics across trajectories."""
        tool_counts = Counter()
        tool_success = Counter()
        tool_fail = Counter()
        
        if iteration >= 0:
            iter_dirs = [self.log_dir / f"iter_{iteration:04d}"]
        else:
            iter_dirs = sorted(self.log_dir.glob("iter_*"))
        
        for iter_dir in iter_dirs:
            if not iter_dir.is_dir():
                continue
            for traj_file in iter_dir.glob("*.json"):
                try:
                    with open(traj_file) as f:
                        data = json.load(f)
                    tools = data.get("tools_used", [])
                    score = data.get("action_score", 0.0)
                    for tool in tools:
                        tool_counts[tool] += 1
                        if score >= 0.8:
                            tool_success[tool] += 1
                        else:
                            tool_fail[tool] += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return {
            "total_tool_calls": sum(tool_counts.values()),
            "unique_tools": len(tool_counts),
            "by_tool": {
                tool: {
                    "count": count,
                    "success": tool_success.get(tool, 0),
                    "fail": tool_fail.get(tool, 0),
                    "success_rate": (
                        tool_success.get(tool, 0) / count if count > 0 else 0
                    ),
                }
                for tool, count in tool_counts.most_common()
            },
        }
    
    def get_stats(self) -> dict:
        """Get overall trajectory statistics."""
        return self._stats.copy()


# ============================================================
# 5. Pattern Detector
# ============================================================

class PatternDetector:
    """Detects recurring patterns across errors and trajectories.
    
    Identifies:
    - Same error type recurring in same domain
    - Score plateaus (no improvement over N iterations)
    - Tool misuse patterns
    - Safety regression (safety score dropping)
    """
    
    def __init__(self, error_memory: ErrorMemory, trajectory_store: TrajectoryStore):
        self.error_memory = error_memory
        self.trajectory_store = trajectory_store
    
    def detect_all_patterns(self, progress_history: list[dict]) -> list[dict]:
        """Run all pattern detection algorithms."""
        patterns = []
        
        # 1. Recurring errors
        recurring = self.error_memory.get_recurring_errors(min_occurrences=2)
        for r in recurring:
            patterns.append({
                "type": "recurring_error",
                "severity": r.severity,
                "description": f"[{r.error_type}] in {r.domain}: {r.description} "
                             f"(occurred {r.occurrences}x across iterations {r.iterations_seen})",
                "recommendation": r.recommendation,
                "fingerprint": r.fingerprint,
            })
        
        # 2. Score plateaus
        plateaus = self._detect_plateaus(progress_history)
        for p in plateaus:
            patterns.append({
                "type": "score_plateau",
                "severity": 3,
                "description": f"Score plateau in {p['domain']}: "
                             f"{p['score']:.1%} for {p['duration']} iterations",
                "recommendation": (
                    f"Try different training approach for {p['domain']}. "
                    f"Current method may have saturated. Consider: "
                    f"{'GRPO' if p.get('current_method') == 'sft' else 'curriculum change'}, "
                    f"new data sources, or increasing model capacity."
                ),
            })
        
        # 3. Safety regressions
        regressions = self._detect_safety_regression(progress_history)
        for r in regressions:
            patterns.append({
                "type": "safety_regression",
                "severity": 5,  # highest
                "description": f"SAFETY REGRESSION in {r['domain']}: "
                             f"score dropped from {r['prev_score']:.1%} to {r['curr_score']:.1%}",
                "recommendation": (
                    f"CRITICAL: Safety score decreased in {r['domain']}. "
                    f"Immediately increase safety reward weight and add safety constraints. "
                    f"Do NOT deploy model until safety score recovers."
                ),
            })
        
        # Sort by severity
        patterns.sort(key=lambda p: p["severity"], reverse=True)
        return patterns
    
    def _detect_plateaus(self, history: list[dict], window: int = 4, threshold: float = 0.02) -> list[dict]:
        """Detect score plateaus — no improvement over N iterations."""
        plateaus = []
        
        if len(history) < window:
            return plateaus
        
        recent = history[-window:]
        domains = set()
        for entry in recent:
            domains.update(entry.get("domains", {}).keys())
        
        for domain in domains:
            scores = []
            for entry in recent:
                data = entry.get("domains", {}).get(domain, {})
                if "action_score" in data:
                    scores.append(data["action_score"])
            
            if len(scores) < window:
                continue
            
            # Check if score hasn't changed significantly
            score_range = max(scores) - min(scores)
            if score_range < threshold and scores[-1] < 0.90:  # Only flag if not yet conquered
                plateaus.append({
                    "domain": domain,
                    "score": scores[-1],
                    "duration": window,
                    "score_range": score_range,
                })
        
        return plateaus
    
    def _detect_safety_regression(self, history: list[dict]) -> list[dict]:
        """Detect safety score regressions."""
        regressions = []
        
        if len(history) < 2:
            return regressions
        
        prev = history[-2]
        curr = history[-1]
        
        for domain in curr.get("domains", {}):
            prev_data = prev.get("domains", {}).get(domain, {})
            curr_data = curr.get("domains", {}).get(domain, {})
            
            # Check safety-related metrics
            prev_safety = prev_data.get("failure_distribution", {}).get("safety_violation", 0)
            curr_safety = curr_data.get("failure_distribution", {}).get("safety_violation", 0)
            
            if curr_safety > prev_safety and curr_safety >= 2:
                regressions.append({
                    "domain": domain,
                    "prev_score": prev_data.get("action_score", 0),
                    "curr_score": curr_data.get("action_score", 0),
                    "prev_safety_violations": prev_safety,
                    "curr_safety_violations": curr_safety,
                })
        
        return regressions
    
    def get_preventive_warnings(self, domain: str, iteration: int) -> list[str]:
        """Get warnings BEFORE training a domain, based on past errors."""
        warnings = []
        
        summary = self.error_memory.get_domain_error_summary(domain)
        
        if summary["total_errors"] == 0:
            return warnings
        
        # Check for frequently recurring issues
        for error_type, count in summary.get("most_common", []):
            if count >= 3:
                warnings.append(
                    f"FREQUENT: '{error_type}' has occurred {count}x in {domain}. "
                    f"Ensure training data addresses this."
                )
        
        # Check severity
        severe_count = summary.get("by_severity", {}).get(5, 0) + summary.get("by_severity", {}).get(4, 0)
        if severe_count > 0:
            warnings.append(
                f"HIGH-SEVERITY: {severe_count} critical errors in {domain}. "
                f"Prioritize safety and correctness."
            )
        
        return warnings


# ============================================================
# 6. Training Memory (Main Interface)
# ============================================================

class TrainingMemory:
    """Unified interface for the complete Training Memory System.
    
    This is the main class that GymCoach uses to record and query
    all training history.
    
    Usage:
        memory = TrainingMemory("logs/training_memory")
        
        # Record
        memory.log_action("evaluate", "tool_call", domain="clinical_dx", ...)
        memory.log_error(ErrorRecord(...))
        memory.log_trajectory(TrajectoryRecord(...))
        
        # Query
        patterns = memory.detect_patterns(progress_history)
        warnings = memory.get_warnings("clinical_dx", iteration=5)
        summary = memory.get_full_summary()
    """
    
    def __init__(self, log_dir: str = "logs/training_memory"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-components
        self.actions = ActionLogger(self.log_dir)
        self.errors = ErrorMemory(self.log_dir)
        self.trajectories = TrajectoryStore(self.log_dir)
        self.patterns = PatternDetector(self.errors, self.trajectories)
        
        # Session tracking
        self._session_start = datetime.now().isoformat()
        self._session_id = hashlib.md5(
            self._session_start.encode()
        ).hexdigest()[:8]
        
        logger.info(f"[TrainingMemory] Initialized session {self._session_id}")
        logger.info(f"[TrainingMemory] Log dir: {self.log_dir}")
        logger.info(f"[TrainingMemory] Loaded {len(self.errors._errors)} historical errors")
        logger.info(f"[TrainingMemory] Loaded {self.trajectories._stats['total']} historical trajectories")
    
    def set_iteration(self, iteration: int):
        """Set current iteration for all sub-loggers."""
        self.actions.set_iteration(iteration)
    
    # ---------- Logging Methods ----------
    
    def log_action(
        self,
        phase: str,
        action_type: str,
        domain: str = "",
        task_id: str = "",
        details: dict = None,
        duration_ms: float = 0.0,
        success: bool = True,
        error_message: str = "",
    ) -> ActionRecord:
        """Log a single action."""
        return self.actions.log(
            phase=phase,
            action_type=action_type,
            domain=domain,
            task_id=task_id,
            details=details,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
        )
    
    def log_error(
        self,
        error_type: str,
        domain: str = "",
        task_id: str = "",
        description: str = "",
        severity: int = 1,
        context: dict = None,
        stack_trace: str = "",
        model_path: str = "",
        iteration: int = 0,
    ) -> ErrorRecord:
        """Log an error and check for recurrence."""
        error = ErrorRecord(
            timestamp=datetime.now().isoformat(),
            iteration=iteration or self.actions._current_iteration,
            error_type=error_type,
            domain=domain,
            task_id=task_id,
            description=description,
            severity=severity,
            context=context or {},
            stack_trace=stack_trace,
            model_path=model_path,
        )
        return self.errors.log_error(error)
    
    def log_trajectory(self, trajectory: TrajectoryRecord):
        """Log a complete trajectory."""
        self.trajectories.store(trajectory)
    
    def log_trajectories_from_results(
        self,
        task_results: list[dict],
        domain: str,
        iteration: int,
        model_path: str = "",
    ):
        """Convert task result dicts to TrajectoryRecords and store them."""
        for result in task_results:
            turns = result.get("turns", [])
            tools_used = []
            for t in turns:
                tc = t.get("parsed_tool_call", {})
                if tc and tc.get("name"):
                    tools_used.append(tc["name"])
            
            traj = TrajectoryRecord(
                timestamp=datetime.now().isoformat(),
                iteration=iteration,
                domain=domain,
                task_id=result.get("task_id", "unknown"),
                model_path=model_path,
                turns=turns,
                total_turns=result.get("total_turns", len(turns)),
                action_score=result.get("action_score", 0.0),
                reward_breakdown=result.get("trajectory", {}).get("reward_details", {}),
                tools_used=tools_used,
                final_answer=result.get("final_answer", ""),
                completed=result.get("completed", False),
                failure_types=[],
            )
            self.trajectories.store(traj)
    
    def log_errors_from_report(self, report, iteration: int = 0, model_path: str = ""):
        """Log all errors from a DomainReport."""
        for fc in report.failure_cases:
            self.log_error(
                error_type=fc.failure_type,
                domain=fc.domain,
                task_id=fc.task_id,
                description=fc.description,
                severity=fc.severity,
                context={
                    "expected_actions": fc.expected_actions,
                    "actual_actions": fc.actual_actions,
                    "missing_actions": fc.missing_actions,
                    "error_category": fc.error_category,
                },
                model_path=model_path,
                iteration=iteration,
            )
    
    # ---------- Query Methods ----------
    
    def detect_patterns(self, progress_history: list[dict]) -> list[dict]:
        """Detect all recurring patterns."""
        return self.patterns.detect_all_patterns(progress_history)
    
    def get_warnings(self, domain: str, iteration: int) -> list[str]:
        """Get preventive warnings before training a domain."""
        return self.patterns.get_preventive_warnings(domain, iteration)
    
    def get_recommendations(self, domain: str = "") -> list[str]:
        """Get training recommendations based on error history."""
        recurring = self.errors.get_recurring_errors(min_occurrences=2)
        
        if domain:
            recurring = [r for r in recurring if r.domain == domain]
        
        return [r.recommendation for r in recurring]
    
    def get_full_summary(self) -> dict:
        """Get comprehensive summary of all training memory."""
        action_summary = self.actions.get_session_summary()
        traj_stats = self.trajectories.get_stats()
        
        # Error summary by domain
        all_domains = set()
        for e in self.errors._errors:
            if e.domain:
                all_domains.add(e.domain)
        
        error_summaries = {}
        for domain in all_domains:
            error_summaries[domain] = self.errors.get_domain_error_summary(domain)
        
        recurring = self.errors.get_recurring_errors(min_occurrences=2)
        
        return {
            "session_id": self._session_id,
            "session_start": self._session_start,
            "actions": action_summary,
            "trajectories": traj_stats,
            "errors": {
                "total": len(self.errors._errors),
                "by_domain": error_summaries,
                "recurring_patterns": len(recurring),
                "top_recurring": [
                    {
                        "type": r.error_type,
                        "domain": r.domain,
                        "occurrences": r.occurrences,
                        "recommendation": r.recommendation,
                    }
                    for r in recurring[:5]
                ],
            },
        }
    
    def print_memory_report(self):
        """Print a human-readable memory report."""
        summary = self.get_full_summary()
        
        print()
        print("=" * 70)
        print("  Training Memory Report")
        print(f"  Session: {summary['session_id']} ({summary['session_start']})")
        print("=" * 70)
        
        # Actions
        a = summary["actions"]
        print(f"\n  Actions: {a['total_actions']} total")
        if a.get("by_phase"):
            for phase, count in sorted(a["by_phase"].items()):
                print(f"    {phase}: {count}")
        
        # Trajectories
        t = summary["trajectories"]
        print(f"\n  Trajectories: {t['total']} total")
        if t.get("by_domain"):
            for domain, count in sorted(t["by_domain"].items()):
                print(f"    {domain}: {count}")
        outcome = t.get("by_outcome", {})
        total_outcomes = sum(outcome.values())
        if total_outcomes > 0:
            pass_rate = outcome.get("pass", 0) / total_outcomes
            print(f"    Pass rate: {pass_rate:.1%} ({outcome.get('pass', 0)}/{total_outcomes})")
        
        # Errors
        e = summary["errors"]
        print(f"\n  Errors: {e['total']} total, {e['recurring_patterns']} recurring patterns")
        if e.get("top_recurring"):
            print("  Top Recurring:")
            for r in e["top_recurring"]:
                print(f"    [{r['type']}] {r['domain']}: {r['occurrences']}x")
                print(f"      -> {r['recommendation'][:80]}...")
        
        print("\n" + "=" * 70)
    
    def save_snapshot(self, iteration: int):
        """Save a complete snapshot of memory state for an iteration."""
        snapshot = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "session_id": self._session_id,
            "summary": self.get_full_summary(),
            "tool_usage": self.trajectories.get_tool_usage_stats(iteration),
        }
        
        snapshot_file = self.log_dir / f"snapshot_iter_{iteration:04d}.json"
        with open(snapshot_file, "w") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[TrainingMemory] Snapshot saved: {snapshot_file}")
