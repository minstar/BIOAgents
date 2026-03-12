"""Adaptive Reward Strategy System for BIOAgents RL Training.

Implements multiple reward computation strategies that the training pipeline
can dynamically select from, enabling the model to learn with the most
appropriate reward signal for each task type.

Strategies:
1. GRPO (Group Relative Policy Optimization) - Standard composite reward
2. MRPO (Multi-Reward Policy Optimization) - Token-level shaping with BERTScore
3. SARL (Search Agent RL) - Self-assessment based reward with tool usage bonus
4. Adaptive - Meta-strategy that selects the best strategy per task

The key innovation is that the model itself can select which reward strategy
to use before training, based on task characteristics. This is implemented
through the `RewardStrategySelector` which analyzes task metadata and selects
the optimal reward computation pipeline.

References:
- GRPO: arXiv:2402.03300
- MRPO: ICML submission (references/medical_qa/MRPO_ICML_submission.pdf)
- SARL: Search-SARL (trains/snapshot-po/reward_computation/)
- FairGRPO: arXiv:2510.19893
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


# ============================================================
# Strategy Definitions
# ============================================================


class RewardStrategyType(str, Enum):
    """Available reward computation strategies."""
    GRPO = "grpo"           # Standard GRPO composite (accuracy + format + process)
    MRPO = "mrpo"           # Multi-Reward PO with token shaping & BERTScore
    SARL = "sarl"           # Search Agent RL with self-assessment rewards
    FAIR_GRPO = "fair_grpo" # Fairness-aware GRPO
    DRPO = "drpo"           # Domain-aware Relative PO (QoQ-Med, NeurIPS 2025)
    CRPO = "crpo"           # Clinical-objective Relative PO (Clinical-R1)
    ADAPTIVE = "adaptive"   # Meta-strategy: auto-selects per task


@dataclass
class RewardStrategyConfig:
    """Configuration for a reward strategy."""
    strategy_type: RewardStrategyType = RewardStrategyType.GRPO
    # GRPO weights
    grpo_weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.4, "format": 0.2, "process": 0.4,
    })
    # MRPO parameters
    mrpo_alignment_weight: float = 0.3
    mrpo_relevance_weight: float = 0.3
    mrpo_factuality_weight: float = 0.4
    mrpo_bertscore_weight: float = 0.5
    mrpo_rouge_weight: float = 0.25
    mrpo_bleu_weight: float = 0.25
    # SARL parameters
    sarl_alpha: float = 0.8           # Self-assessment decay
    sarl_lambda: float = 0.5          # Assessment reward weight
    sarl_tool_bonus_per_call: float = 0.1
    sarl_tool_bonus_cap: float = 0.3
    sarl_format_answer_bonus: float = 0.1
    sarl_format_tool_bonus: float = 0.05
    # FairGRPO parameters
    fairness_weight: float = 0.1
    alpha_repr: float = 0.5
    alpha_perf: float = 0.5
    # DRPO parameters (QoQ-Med, NeurIPS 2025)
    drpo_domain_rarity_weight: float = 0.4     # Upweight rare/difficult domains
    drpo_modality_difficulty_weight: float = 0.3  # Scale by modality complexity
    drpo_base_strategy: str = "grpo"           # Base reward before domain scaling
    drpo_rarity_exponent: float = 0.5          # Rarity scaling power: freq^(-exp)
    drpo_modality_scales: Dict[str, float] = field(default_factory=lambda: {
        "text": 1.0,
        "vision": 1.3,         # VL tasks are harder
        "multi_modal": 1.5,    # Multi-modal cross-domain hardest
    })
    drpo_domain_difficulty: Dict[str, float] = field(default_factory=lambda: {
        "medical_qa": 0.8,           # Relatively easier (MCQA)
        "clinical_diagnosis": 1.0,
        "drug_interaction": 1.0,
        "ehr_management": 1.1,
        "triage_emergency": 1.2,
        "psychiatry": 1.1,
        "obstetrics": 1.1,
        "visual_diagnosis": 1.3,     # VL domain
        "radiology_report": 1.4,     # VL + structured reporting
        "cross_domain": 1.5,         # Multi-phase hardest
    })
    # CRPO parameters (Clinical-R1, arXiv:2512.00601)
    crpo_accuracy_weight: float = 0.35      # Correctness
    crpo_faithfulness_weight: float = 0.30  # No hallucination, grounded in evidence
    crpo_completeness_weight: float = 0.20  # Covers all relevant aspects
    crpo_safety_weight: float = 0.15        # No harmful recommendations
    crpo_hallucination_penalty: float = -0.3   # Penalty per hallucinated claim
    crpo_evidence_bonus: float = 0.15          # Bonus for citing evidence
    # Adaptive parameters
    adaptive_task_analysis: bool = True   # Analyze task to choose strategy
    adaptive_fallback: RewardStrategyType = RewardStrategyType.GRPO


# ============================================================
# Base Strategy Interface
# ============================================================


class RewardStrategy(ABC):
    """Abstract base class for reward computation strategies."""

    def __init__(self, config: RewardStrategyConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the strategy."""
        ...

    @property
    @abstractmethod
    def strategy_type(self) -> RewardStrategyType:
        """The strategy type enum."""
        ...

    @abstractmethod
    def compute_reward(
        self,
        completions: list,
        solution: list = None,
        **kwargs,
    ) -> list[float]:
        """Compute rewards for a batch of completions.

        Args:
            completions: List of model completions (TRL format)
            solution: Ground truth answers
            **kwargs: Strategy-specific parameters

        Returns:
            List of reward scores
        """
        ...

    def get_reward_breakdown(
        self,
        completions: list,
        solution: list = None,
        **kwargs,
    ) -> list[dict]:
        """Get detailed reward breakdown for analysis.

        Returns:
            List of dicts with component scores for each completion
        """
        rewards = self.compute_reward(completions, solution=solution, **kwargs)
        return [{"total": r, "strategy": self.name} for r in rewards]


# ============================================================
# GRPO Strategy (Standard)
# ============================================================


class GRPORewardStrategy(RewardStrategy):
    """Standard GRPO composite reward strategy.

    Combines accuracy, format, and process rewards with configurable weights.
    This is the default strategy used in Healthcare AI GYM.
    """

    @property
    def name(self) -> str:
        return "GRPO (Standard Composite)"

    @property
    def strategy_type(self) -> RewardStrategyType:
        return RewardStrategyType.GRPO

    def compute_reward(
        self,
        completions: list,
        solution: list = None,
        **kwargs,
    ) -> list[float]:
        from bioagents.evaluation.grpo_rewards import (
            grpo_accuracy_reward,
            grpo_format_reward,
            grpo_process_reward,
        )

        w = self.config.grpo_weights
        accuracy_scores = grpo_accuracy_reward(
            completions, solution=solution, **kwargs,
        )
        format_scores = grpo_format_reward(completions, **kwargs)
        process_scores = grpo_process_reward(
            completions, solution=solution, **kwargs,
        )

        rewards = []
        for acc, fmt, proc in zip(accuracy_scores, format_scores, process_scores):
            total = (
                w.get("accuracy", 0.4) * acc
                + w.get("format", 0.2) * fmt
                + w.get("process", 0.4) * proc
            )
            rewards.append(total)
        return rewards

    def get_reward_breakdown(
        self,
        completions: list,
        solution: list = None,
        **kwargs,
    ) -> list[dict]:
        from bioagents.evaluation.grpo_rewards import (
            grpo_accuracy_reward,
            grpo_format_reward,
            grpo_process_reward,
        )

        accuracy_scores = grpo_accuracy_reward(
            completions, solution=solution, **kwargs,
        )
        format_scores = grpo_format_reward(completions, **kwargs)
        process_scores = grpo_process_reward(
            completions, solution=solution, **kwargs,
        )

        w = self.config.grpo_weights
        breakdowns = []
        for acc, fmt, proc in zip(accuracy_scores, format_scores, process_scores):
            total = (
                w.get("accuracy", 0.4) * acc
                + w.get("format", 0.2) * fmt
                + w.get("process", 0.4) * proc
            )
            breakdowns.append({
                "total": total,
                "strategy": self.name,
                "accuracy": acc,
                "format": fmt,
                "process": proc,
                "weights": w,
            })
        return breakdowns


# ============================================================
# MRPO Strategy (Multi-Reward Policy Optimization)
# ============================================================


class MRPORewardStrategy(RewardStrategy):
    """Multi-Reward Policy Optimization strategy.

    Implements the MRPO framework with token-level reward shaping:
    - Alignment reward: How well the response aligns with expected structure
    - Relevance reward: BERTScore-based semantic relevance
    - Factuality reward: ROUGE/BLEU-based factual overlap
    - Token shaping: Per-token reward weighting for fine-grained learning

    Reference: MRPO ICML submission, grpo_vqa_Qwen3_token_shaping.py
    """

    @property
    def name(self) -> str:
        return "MRPO (Multi-Reward Token Shaping)"

    @property
    def strategy_type(self) -> RewardStrategyType:
        return RewardStrategyType.MRPO

    def compute_reward(
        self,
        completions: list,
        solution: list = None,
        bert_scorer=None,
        **kwargs,
    ) -> list[float]:
        from bioagents.evaluation.grpo_rewards import (
            _extract_content,
            _extract_answer_from_content,
            compute_rouge1_f1,
            compute_bleu1,
            compute_bertscore_f1,
        )

        contents = _extract_content(completions)
        solutions = solution or [""] * len(contents)
        if isinstance(solutions, str):
            solutions = [solutions] * len(contents)

        rewards = []
        for content, sol in zip(contents, solutions):
            student_answer = _extract_answer_from_content(content)
            ground_truth = sol.strip()

            if not ground_truth:
                rewards.append(0.0)
                continue

            # Multiple-choice exact match (binary)
            if len(ground_truth) <= 2 and ground_truth.upper() in "ABCDE":
                from bioagents.evaluation.rewards import _extract_answer_from_response
                extracted = _extract_answer_from_response(content)
                if not extracted:
                    extracted = _extract_answer_from_response(student_answer)
                if extracted and extracted.upper() == ground_truth.upper():
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
                continue

            # --- MRPO multi-signal reward ---

            # 1. Alignment: structural quality (think/answer tags, tool format)
            alignment = self._compute_alignment(content)

            # 2. Relevance: BERTScore semantic similarity
            relevance = compute_bertscore_f1(
                student_answer, ground_truth, scorer=bert_scorer,
            )

            # 3. Factuality: ROUGE-1 + BLEU-1 factual overlap
            rouge_score = compute_rouge1_f1(student_answer, ground_truth)
            bleu_score = compute_bleu1(student_answer, ground_truth)
            factuality = (
                self.config.mrpo_rouge_weight * rouge_score
                + self.config.mrpo_bleu_weight * bleu_score
                + (1 - self.config.mrpo_rouge_weight - self.config.mrpo_bleu_weight)
                * compute_bertscore_f1(student_answer, ground_truth, scorer=bert_scorer)
            )

            # Weighted MRPO combination
            mrpo_reward = (
                self.config.mrpo_alignment_weight * alignment
                + self.config.mrpo_relevance_weight * relevance
                + self.config.mrpo_factuality_weight * factuality
            )

            rewards.append(float(mrpo_reward))

        return rewards

    def _compute_alignment(self, content: str) -> float:
        """Compute alignment score: structural quality of the response."""
        score = 0.0

        # Check for thinking structure
        has_think = bool(re.search(r"<think>.*?</think>", content, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", content, re.DOTALL))

        if has_think:
            score += 0.3
        if has_answer:
            score += 0.3

        # Check for tool call format
        try:
            parsed = json.loads(content.strip())
            if isinstance(parsed, dict) and "name" in parsed:
                score += 0.2
        except (json.JSONDecodeError, ValueError):
            pass

        # Check for medical reasoning markers
        medical_markers = [
            "diagnosis", "treatment", "mechanism", "evidence",
            "because", "therefore", "patient", "clinical",
        ]
        marker_count = sum(1 for m in medical_markers if m in content.lower())
        score += min(0.2, marker_count * 0.03)

        return min(1.0, score)

    def get_reward_breakdown(
        self,
        completions: list,
        solution: list = None,
        bert_scorer=None,
        **kwargs,
    ) -> list[dict]:
        from bioagents.evaluation.grpo_rewards import (
            _extract_content,
            _extract_answer_from_content,
            compute_rouge1_f1,
            compute_bleu1,
            compute_bertscore_f1,
        )

        contents = _extract_content(completions)
        solutions = solution or [""] * len(contents)
        if isinstance(solutions, str):
            solutions = [solutions] * len(contents)

        breakdowns = []
        for content, sol in zip(contents, solutions):
            student_answer = _extract_answer_from_content(content)
            ground_truth = sol.strip()

            alignment = self._compute_alignment(content)
            relevance = compute_bertscore_f1(
                student_answer, ground_truth, scorer=bert_scorer,
            ) if ground_truth else 0.0
            rouge_score = compute_rouge1_f1(student_answer, ground_truth) if ground_truth else 0.0
            bleu_score = compute_bleu1(student_answer, ground_truth) if ground_truth else 0.0

            total = (
                self.config.mrpo_alignment_weight * alignment
                + self.config.mrpo_relevance_weight * relevance
                + self.config.mrpo_factuality_weight * (
                    self.config.mrpo_rouge_weight * rouge_score
                    + self.config.mrpo_bleu_weight * bleu_score
                )
            )

            breakdowns.append({
                "total": total,
                "strategy": self.name,
                "alignment": alignment,
                "relevance": relevance,
                "rouge1": rouge_score,
                "bleu1": bleu_score,
                "factuality": rouge_score * 0.5 + bleu_score * 0.5,
            })

        return breakdowns


# ============================================================
# SARL Strategy (Search Agent RL)
# ============================================================


class SARLRewardStrategy(RewardStrategy):
    """Search Agent RL reward strategy.

    Implements the SARL paper reward design (Eq. 2-3):
      R_total = r_final * alpha^(T-1) + lambda * r_assess + tool_bonus + format_bonus

    Key features:
    - Self-assessment decay: Penalizes excessive self-correction attempts
    - Transition rewards: Rewards correct→correct, penalizes correct→incorrect
    - Tool usage bonus: Prevents parametric knowledge shortcutting
    - Format compliance bonus: Rewards proper tool call and answer formatting

    Reference: Search-SARL (trains/snapshot-po/reward_computation/)
    """

    @property
    def name(self) -> str:
        return "SARL (Search Agent RL)"

    @property
    def strategy_type(self) -> RewardStrategyType:
        return RewardStrategyType.SARL

    def compute_reward(
        self,
        completions: list,
        solution: list = None,
        **kwargs,
    ) -> list[float]:
        from bioagents.evaluation.grpo_rewards import _extract_content

        contents = _extract_content(completions)
        solutions = solution or [""] * len(contents)
        if isinstance(solutions, str):
            solutions = [solutions] * len(contents)

        rewards = []
        for content, sol in zip(contents, solutions):
            reward = self._compute_sarl_reward(content, sol)
            rewards.append(reward)

        return rewards

    def _compute_sarl_reward(self, content: str, ground_truth: str) -> float:
        """Compute full SARL reward for a single completion."""
        alpha = self.config.sarl_alpha
        lam = self.config.sarl_lambda

        # Extract all answer blocks (for self-assessment tracking)
        answers = self._extract_all_answers(content)

        # Determine correctness of each answer
        correct_flags = [self._is_correct(a, ground_truth) for a in answers]

        # r_final: binary correctness of final answer
        is_correct_final = bool(correct_flags[-1]) if correct_flags else False
        r_final = 1.0 if is_correct_final else 0.0

        # Count self-assessment judgments (COMPLETE/INCOMPLETE)
        t_raw = self._count_self_assessments(content)
        t_eff = max(1, t_raw)
        penalty = float(alpha ** (t_eff - 1))

        # Transition rewards across self-assessment steps
        n_transitions = min(t_raw, max(0, len(correct_flags) - 1))
        r_assess = 0.0
        for j in range(n_transitions):
            r_assess += self._transition_reward(correct_flags[j], correct_flags[j + 1])

        # Base SARL reward
        reward = (r_final * penalty) + (lam * r_assess)

        # Tool usage bonus (capped)
        tool_calls = len(re.findall(
            r'(?:"name"\s*:\s*"(?:search|browse|search_pubmed|search_medical_wiki|'
            r'retrieve_evidence|browse_article|browse_wiki_entry)"'
            r'|<name>(?:search|browse)</name>)',
            content, re.IGNORECASE,
        ))
        tool_bonus = min(
            tool_calls * self.config.sarl_tool_bonus_per_call,
            self.config.sarl_tool_bonus_cap,
        )
        # Only award tool bonus if final answer is correct
        if is_correct_final:
            reward += tool_bonus

        # Format compliance bonus
        has_answer_tag = bool(re.search(r"<answer>.*?</answer>", content, re.DOTALL))
        has_valid_tool = bool(re.search(
            r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*[^{}]*"arguments"\s*:\s*\{',
            content, re.DOTALL,
        ))
        if has_answer_tag:
            reward += self.config.sarl_format_answer_bonus
        if has_valid_tool:
            reward += self.config.sarl_format_tool_bonus

        return reward

    def _extract_all_answers(self, text: str) -> list[str]:
        """Extract all <answer>...</answer> blocks in order."""
        answers = [
            m.group(1).strip()
            for m in re.finditer(r"<answer>(.*?)</answer>", text or "", re.DOTALL | re.IGNORECASE)
        ]
        if answers:
            return answers
        return [str(text or "").strip()]

    def _is_correct(self, answer: str, ground_truth: str) -> bool:
        """Forgiving correctness check."""
        pred = (answer or "").strip().strip('"').strip("'").lower()
        gt = (ground_truth or "").strip().strip('"').strip("'").lower()
        if not pred or not gt:
            return False
        # Exact match or containment
        return pred == gt or gt in pred or pred in gt

    def _count_self_assessments(self, text: str) -> int:
        """Count COMPLETE/INCOMPLETE self-assessment markers."""
        incomplete = len(re.findall(r"\bINCOMPLETE\b", text or "", re.IGNORECASE))
        complete = len(re.findall(r"\bCOMPLETE\b", text or "", re.IGNORECASE))
        return incomplete + complete

    def _transition_reward(self, before_correct: bool, after_correct: bool) -> float:
        """SARL transition reward between self-assessment steps."""
        if not before_correct and after_correct:
            return 1.0   # Improved: incorrect → correct
        if before_correct and after_correct:
            return 1.0   # Maintained: correct → correct
        if not before_correct and not after_correct:
            return -0.1  # Stagnant: incorrect → incorrect
        return -2.0       # Regressed: correct → incorrect

    def get_reward_breakdown(
        self,
        completions: list,
        solution: list = None,
        **kwargs,
    ) -> list[dict]:
        from bioagents.evaluation.grpo_rewards import _extract_content

        contents = _extract_content(completions)
        solutions = solution or [""] * len(contents)
        if isinstance(solutions, str):
            solutions = [solutions] * len(contents)

        breakdowns = []
        for content, sol in zip(contents, solutions):
            answers = self._extract_all_answers(content)
            correct_flags = [self._is_correct(a, sol) for a in answers]
            is_correct_final = bool(correct_flags[-1]) if correct_flags else False

            t_raw = self._count_self_assessments(content)
            penalty = float(self.config.sarl_alpha ** max(1, t_raw) - 1) if t_raw > 0 else 1.0

            tool_calls = len(re.findall(
                r'"name"\s*:\s*"(?:search|browse|search_pubmed|search_medical_wiki)',
                content, re.IGNORECASE,
            ))

            total = self._compute_sarl_reward(content, sol)

            breakdowns.append({
                "total": total,
                "strategy": self.name,
                "r_final": 1.0 if is_correct_final else 0.0,
                "num_answers": len(answers),
                "num_self_assessments": t_raw,
                "assessment_penalty": penalty,
                "num_tool_calls": tool_calls,
                "is_correct_final": is_correct_final,
            })

        return breakdowns


# ============================================================
# DRPO Strategy (Domain-aware Relative Policy Optimization)
# ============================================================


class DRPORewardStrategy(RewardStrategy):
    """Domain-aware Relative Policy Optimization reward strategy.

    Implements the DRPO framework from QoQ-Med (NeurIPS 2025 Oral):
    - Hierarchically scales normalized rewards by domain rarity
    - Modality difficulty scaling (text < vision < multi-modal)
    - Prevents the model from over-fitting to easy/frequent domains
    - Ensures balanced learning across all clinical specialties

    DRPO reward = base_reward * domain_scale * modality_scale

    where:
        domain_scale = (1/domain_freq)^rarity_exponent * domain_difficulty
        modality_scale = modality_difficulty[task_modality]

    Reference: QoQ-Med (arXiv:2506.00711)
    """

    def __init__(self, config: RewardStrategyConfig):
        super().__init__(config)
        self._domain_counts: Dict[str, int] = {}
        self._total_tasks: int = 0

    @property
    def name(self) -> str:
        return "DRPO (Domain-aware Relative PO)"

    @property
    def strategy_type(self) -> RewardStrategyType:
        return RewardStrategyType.DRPO

    def record_domain(self, domain: str) -> None:
        """Record domain occurrence for frequency-based rarity scaling."""
        self._domain_counts[domain] = self._domain_counts.get(domain, 0) + 1
        self._total_tasks += 1

    def _get_domain_rarity_scale(self, domain: str) -> float:
        """Compute domain rarity scale: upweight rare domains.

        Uses inverse frequency with configurable exponent:
            scale = (total / (n_domains * domain_count))^exponent
        """
        if not self._domain_counts or self._total_tasks == 0:
            return 1.0

        n_domains = max(len(self._domain_counts), 1)
        domain_count = self._domain_counts.get(domain, 1)
        uniform_count = self._total_tasks / n_domains

        rarity = uniform_count / max(domain_count, 1)
        return rarity ** self.config.drpo_rarity_exponent

    def _get_modality_scale(self, task: Optional[dict] = None) -> float:
        """Get modality difficulty scale based on task type."""
        if not task:
            return 1.0

        domain = task.get("domain", "")
        vl_domains = {"visual_diagnosis", "radiology_report"}
        cross_domains = {"cross_domain"}

        if domain in cross_domains:
            modality = "multi_modal"
        elif domain in vl_domains:
            modality = "vision"
        else:
            modality = "text"

        return self.config.drpo_modality_scales.get(modality, 1.0)

    def _get_domain_difficulty(self, domain: str) -> float:
        """Get pre-configured domain difficulty scale."""
        return self.config.drpo_domain_difficulty.get(domain, 1.0)

    def compute_reward(
        self,
        completions: list,
        solution: list = None,
        task: Optional[dict] = None,
        domain: str = "",
        **kwargs,
    ) -> list[float]:
        """Compute DRPO-scaled rewards.

        Wraps a base reward strategy and applies domain-aware scaling.
        """
        from bioagents.evaluation.grpo_rewards import (
            grpo_accuracy_reward,
            grpo_format_reward,
            grpo_process_reward,
        )

        # Determine domain
        task_domain = domain or (task.get("domain", "") if task else "")

        # Record domain for rarity tracking
        if task_domain:
            self.record_domain(task_domain)

        # Compute base rewards (same as GRPO)
        w = self.config.grpo_weights
        accuracy_scores = grpo_accuracy_reward(
            completions, solution=solution, **kwargs,
        )
        format_scores = grpo_format_reward(completions, **kwargs)
        process_scores = grpo_process_reward(
            completions, solution=solution, **kwargs,
        )

        # Compute DRPO scaling factors
        rarity_scale = self._get_domain_rarity_scale(task_domain)
        modality_scale = self._get_modality_scale(task)
        difficulty_scale = self._get_domain_difficulty(task_domain)

        # Combined DRPO scale = rarity * modality * difficulty
        # Clamp to [0.5, 3.0] to prevent extreme scaling
        drpo_scale = rarity_scale * modality_scale * difficulty_scale
        drpo_scale = max(0.5, min(3.0, drpo_scale))

        rewards = []
        for acc, fmt, proc in zip(accuracy_scores, format_scores, process_scores):
            base_reward = (
                w.get("accuracy", 0.4) * acc
                + w.get("format", 0.2) * fmt
                + w.get("process", 0.4) * proc
            )
            # Apply DRPO domain-aware scaling
            scaled_reward = base_reward * drpo_scale
            rewards.append(float(scaled_reward))

        return rewards

    def get_reward_breakdown(
        self,
        completions: list,
        solution: list = None,
        task: Optional[dict] = None,
        domain: str = "",
        **kwargs,
    ) -> list[dict]:
        from bioagents.evaluation.grpo_rewards import (
            grpo_accuracy_reward,
            grpo_format_reward,
            grpo_process_reward,
        )

        task_domain = domain or (task.get("domain", "") if task else "")
        w = self.config.grpo_weights

        accuracy_scores = grpo_accuracy_reward(
            completions, solution=solution, **kwargs,
        )
        format_scores = grpo_format_reward(completions, **kwargs)
        process_scores = grpo_process_reward(
            completions, solution=solution, **kwargs,
        )

        rarity_scale = self._get_domain_rarity_scale(task_domain)
        modality_scale = self._get_modality_scale(task)
        difficulty_scale = self._get_domain_difficulty(task_domain)
        drpo_scale = max(0.5, min(3.0, rarity_scale * modality_scale * difficulty_scale))

        breakdowns = []
        for acc, fmt, proc in zip(accuracy_scores, format_scores, process_scores):
            base = (
                w.get("accuracy", 0.4) * acc
                + w.get("format", 0.2) * fmt
                + w.get("process", 0.4) * proc
            )
            breakdowns.append({
                "total": base * drpo_scale,
                "strategy": self.name,
                "base_reward": base,
                "accuracy": acc,
                "format": fmt,
                "process": proc,
                "domain": task_domain,
                "drpo_scale": drpo_scale,
                "rarity_scale": rarity_scale,
                "modality_scale": modality_scale,
                "difficulty_scale": difficulty_scale,
            })
        return breakdowns

    def get_domain_stats(self) -> dict:
        """Get domain frequency and scaling statistics."""
        stats = {}
        for domain, count in self._domain_counts.items():
            stats[domain] = {
                "count": count,
                "rarity_scale": round(self._get_domain_rarity_scale(domain), 4),
                "difficulty_scale": self._get_domain_difficulty(domain),
            }
        return stats


# ============================================================
# CRPO Strategy (Clinical-objective Relative Policy Optimization)
# ============================================================


class CRPORewardStrategy(RewardStrategy):
    """Clinical-objective Relative Policy Optimization reward strategy.

    Extends GRPO to medical domains by jointly optimizing four clinical
    objectives critical for high-stakes medical reasoning:

    1. **Accuracy**: Correctness of the final answer
    2. **Faithfulness**: Grounded in evidence, no hallucinations
    3. **Completeness**: Covers all clinically relevant aspects
    4. **Safety**: No harmful recommendations or missed red flags

    Key innovations from Clinical-R1 (arXiv:2512.00601):
    - Rule-based faithfulness scoring (evidence citation detection)
    - Completeness rubric matching (key clinical elements coverage)
    - Hallucination penalty (unsupported claims detection)
    - Evidence citation bonus (search/browse tool usage reward)

    Reference: Clinical-R1 (arXiv:2512.00601)
    """

    @property
    def name(self) -> str:
        return "CRPO (Clinical-objective Relative PO)"

    @property
    def strategy_type(self) -> RewardStrategyType:
        return RewardStrategyType.CRPO

    def compute_reward(
        self,
        completions: list,
        solution: list = None,
        rubric: Optional[list] = None,
        **kwargs,
    ) -> list[float]:
        from bioagents.evaluation.grpo_rewards import (
            _extract_content,
            _extract_answer_from_content,
            grpo_accuracy_reward,
        )

        contents = _extract_content(completions)
        solutions = solution or [""] * len(contents)
        if isinstance(solutions, str):
            solutions = [solutions] * len(contents)
        rubrics = rubric or [None] * len(contents)

        # Accuracy from base GRPO
        accuracy_scores = grpo_accuracy_reward(
            completions, solution=solution, **kwargs,
        )

        rewards = []
        for i, (content, sol) in enumerate(zip(contents, solutions)):
            acc = accuracy_scores[i]
            faith = self._compute_faithfulness(content)
            comp = self._compute_completeness(
                content, sol, rubrics[i] if i < len(rubrics) else None
            )
            safe = self._compute_safety(content)

            crpo_reward = (
                self.config.crpo_accuracy_weight * acc
                + self.config.crpo_faithfulness_weight * faith
                + self.config.crpo_completeness_weight * comp
                + self.config.crpo_safety_weight * safe
            )

            rewards.append(float(max(0.0, min(1.5, crpo_reward))))

        return rewards

    def _compute_faithfulness(self, content: str) -> float:
        """Compute faithfulness score: grounded in evidence, no hallucinations.

        Heuristic-based scoring:
        - +bonus for citing evidence (search/browse tool usage)
        - +bonus for referencing guidelines or literature
        - -penalty for hedging-free assertions without evidence
        - -penalty for fabricated statistics/citations
        """
        score = 0.5  # Baseline neutral

        # Evidence citation indicators (positive)
        evidence_patterns = [
            r'"name"\s*:\s*"(?:search|search_pubmed|search_evidence|search_guidelines|browse)',
            r"(?:according to|based on|evidence suggests|studies show|guidelines recommend)",
            r"(?:PMID|doi:|pubmed|guideline|protocol)",
            r"(?:search\(|browse\(|retrieve_evidence\()",
        ]
        evidence_count = 0
        for pattern in evidence_patterns:
            evidence_count += len(re.findall(pattern, content, re.IGNORECASE))

        score += min(0.3, evidence_count * self.config.crpo_evidence_bonus)

        # Hallucination indicators (negative)
        hallucination_patterns = [
            # Fabricated statistics without source
            r"\b\d{2,3}%\s+(?:of patients|survival rate|cure rate|efficacy)\b",
            # Fake citation patterns
            r"(?:Smith et al\.|recent study|a 20\d{2} study)\s+(?:showed|found|demonstrated)",
            # Over-confident claims without hedging
            r"\b(?:always|never|guaranteed|100%|definitely)\b",
        ]
        hallucination_count = 0
        for pattern in hallucination_patterns:
            hallucination_count += len(re.findall(pattern, content, re.IGNORECASE))

        # Only penalize if there's no evidence to back up claims
        if evidence_count == 0 and hallucination_count > 0:
            score += hallucination_count * self.config.crpo_hallucination_penalty

        # Appropriate hedging (positive)
        hedging_patterns = [
            r"\b(?:may|might|could|possibly|consider|differential|likely|probable)\b",
            r"\b(?:further workup|additional testing|recommend consultation)\b",
        ]
        hedging_count = sum(
            len(re.findall(p, content, re.IGNORECASE)) for p in hedging_patterns
        )
        score += min(0.15, hedging_count * 0.03)

        return max(0.0, min(1.0, score))

    def _compute_completeness(
        self, content: str, ground_truth: str, rubric: Optional[dict] = None
    ) -> float:
        """Compute completeness score: covers all clinically relevant aspects.

        Uses rubric matching when available, otherwise heuristic-based.
        """
        score = 0.0

        # If rubric is provided, use it for precise scoring
        if rubric and isinstance(rubric, dict):
            required_elements = rubric.get("required_elements", [])
            if required_elements:
                found = 0
                for element in required_elements:
                    if isinstance(element, str) and element.lower() in content.lower():
                        found += 1
                score = found / max(len(required_elements), 1)
                return score

        # Heuristic-based completeness scoring
        # Check for clinical reasoning structure
        clinical_elements = {
            "history": [r"\b(?:history|HPI|presenting|chief complaint|symptoms)\b"],
            "examination": [r"\b(?:exam|vital|BP|HR|SpO2|temperature|physical)\b"],
            "differential": [r"\b(?:differential|DDx|consider|rule out|diagnos)\b"],
            "investigation": [r"\b(?:lab|imaging|test|CT|MRI|X-ray|blood|CBC|BMP)\b"],
            "assessment": [r"\b(?:assessment|impression|diagnosis|ICD|conclusion)\b"],
            "plan": [r"\b(?:plan|treatment|prescri|recommend|follow.up|refer)\b"],
        }

        elements_found = 0
        for element_name, patterns in clinical_elements.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    elements_found += 1
                    break

        score = elements_found / max(len(clinical_elements), 1)

        # Bonus for structured output
        if re.search(r"(?:1\.|2\.|3\.|\n-|\n\*|step\s*\d)", content, re.IGNORECASE):
            score = min(1.0, score + 0.1)

        return score

    def _compute_safety(self, content: str) -> float:
        """Compute safety score for CRPO: no harmful recommendations.

        Lightweight rule-based check (heavier checks in safety_eval.py).
        """
        score = 1.0  # Start safe

        # Critical safety violations
        critical_patterns = [
            (r"\b(?:do not call 911|do not go to ER|stay home)\b", -0.5),
            (r"\b(?:ignore|disregard)\s+(?:symptoms|chest pain|bleeding)\b", -0.4),
            (r"\b(?:stop taking)\s+(?:insulin|anticoagulant|blood thinner)\b", -0.4),
            (r"\b(?:self-medicate|self-treat|no need for doctor)\b", -0.3),
        ]

        for pattern, penalty in critical_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score += penalty

        # Positive safety indicators
        safety_positive = [
            r"\b(?:consult|refer|specialist|emergency|911|urgent)\b",
            r"\b(?:contraindicated|allergy|interaction|caution|monitor)\b",
            r"\b(?:informed consent|risk.benefit|side effect)\b",
        ]
        positive_count = sum(
            1 for p in safety_positive if re.search(p, content, re.IGNORECASE)
        )
        score = min(1.0, score + positive_count * 0.05)

        return max(0.0, score)

    def get_reward_breakdown(
        self,
        completions: list,
        solution: list = None,
        rubric: Optional[list] = None,
        **kwargs,
    ) -> list[dict]:
        from bioagents.evaluation.grpo_rewards import (
            _extract_content,
            grpo_accuracy_reward,
        )

        contents = _extract_content(completions)
        solutions = solution or [""] * len(contents)
        if isinstance(solutions, str):
            solutions = [solutions] * len(contents)
        rubrics = rubric or [None] * len(contents)

        accuracy_scores = grpo_accuracy_reward(
            completions, solution=solution, **kwargs,
        )

        breakdowns = []
        for i, (content, sol) in enumerate(zip(contents, solutions)):
            acc = accuracy_scores[i]
            faith = self._compute_faithfulness(content)
            comp = self._compute_completeness(
                content, sol, rubrics[i] if i < len(rubrics) else None
            )
            safe = self._compute_safety(content)

            total = (
                self.config.crpo_accuracy_weight * acc
                + self.config.crpo_faithfulness_weight * faith
                + self.config.crpo_completeness_weight * comp
                + self.config.crpo_safety_weight * safe
            )

            breakdowns.append({
                "total": max(0.0, min(1.5, total)),
                "strategy": self.name,
                "accuracy": acc,
                "faithfulness": faith,
                "completeness": comp,
                "safety": safe,
                "weights": {
                    "accuracy": self.config.crpo_accuracy_weight,
                    "faithfulness": self.config.crpo_faithfulness_weight,
                    "completeness": self.config.crpo_completeness_weight,
                    "safety": self.config.crpo_safety_weight,
                },
            })

        return breakdowns


# ============================================================
# Adaptive Strategy (Meta-Strategy)
# ============================================================


@dataclass
class TaskCharacteristics:
    """Analyzed characteristics of a task for strategy selection."""
    has_multiple_choice: bool = False
    has_tool_use_expected: bool = False
    has_search_expected: bool = False
    has_patient_data: bool = False
    domain: str = ""
    difficulty: str = ""
    num_expected_tools: int = 0
    is_multi_turn: bool = False


class AdaptiveRewardStrategy(RewardStrategy):
    """Meta-strategy that dynamically selects the best reward strategy per task.

    This is the key innovation: before training, the system analyzes each task's
    characteristics and selects the optimal reward computation strategy.

    Selection heuristics:
    - Tasks with heavy search/browse requirements → SARL (search-optimized)
    - Tasks requiring fine-grained medical reasoning → MRPO (token shaping)
    - Tasks with patient demographics → FairGRPO
    - General tasks → GRPO (standard composite)
    """

    def __init__(self, config: RewardStrategyConfig):
        super().__init__(config)
        self._strategies: Dict[RewardStrategyType, RewardStrategy] = {
            RewardStrategyType.GRPO: GRPORewardStrategy(config),
            RewardStrategyType.MRPO: MRPORewardStrategy(config),
            RewardStrategyType.SARL: SARLRewardStrategy(config),
            RewardStrategyType.DRPO: DRPORewardStrategy(config),
            RewardStrategyType.CRPO: CRPORewardStrategy(config),
        }
        self._selection_log: list[dict] = []

    @property
    def name(self) -> str:
        return "Adaptive (Auto-Select)"

    @property
    def strategy_type(self) -> RewardStrategyType:
        return RewardStrategyType.ADAPTIVE

    def analyze_task(self, task: dict) -> TaskCharacteristics:
        """Analyze a task to determine its characteristics for strategy selection."""
        eval_criteria = task.get("evaluation_criteria", {})
        actions = eval_criteria.get("actions", [])
        description = task.get("description", {})

        # Determine characteristics
        chars = TaskCharacteristics()
        chars.domain = task.get("domain", description.get("category", ""))
        chars.difficulty = description.get("difficulty", "medium")
        chars.has_multiple_choice = bool(task.get("options"))
        chars.num_expected_tools = len(actions)
        chars.has_tool_use_expected = chars.num_expected_tools > 0

        # Check for search/browse tools
        search_tools = {"search_pubmed", "browse_article", "search_medical_wiki",
                        "browse_wiki_entry", "retrieve_evidence", "search",
                        "browse", "search_imaging_knowledge", "search_radiology_knowledge",
                        "search_guidelines", "search_drugs_by_class", "search_alternatives"}
        tool_names = {a.get("name", "") for a in actions}
        chars.has_search_expected = bool(tool_names & search_tools)

        # Check for patient data
        chars.has_patient_data = bool(
            task.get("patient_id")
            or any("patient" in a.get("name", "").lower() for a in actions)
        )

        # Multi-turn detection
        chars.is_multi_turn = chars.num_expected_tools > 2

        return chars

    def select_strategy(self, task: dict) -> RewardStrategy:
        """Select the best reward strategy for a given task.

        Args:
            task: Task dictionary with metadata and evaluation criteria

        Returns:
            The selected RewardStrategy instance
        """
        chars = self.analyze_task(task)

        # Decision tree for strategy selection
        # Enhanced with DRPO (domain-aware) and CRPO (clinical-objective)
        selected: RewardStrategyType

        if chars.domain in ("visual_diagnosis", "radiology_report"):
            # VL domains with domain imbalance → DRPO for domain-aware scaling
            selected = RewardStrategyType.DRPO
            reason = f"VL domain '{chars.domain}' → DRPO domain-aware scaling"

        elif chars.domain == "cross_domain":
            # Multi-phase cross-domain → DRPO + modality scaling
            selected = RewardStrategyType.DRPO
            reason = "Cross-domain pathway → DRPO multi-modal difficulty scaling"

        elif chars.has_search_expected and chars.num_expected_tools >= 2:
            # Heavy search tasks → SARL excels at rewarding tool usage
            selected = RewardStrategyType.SARL
            reason = "Task requires search/browse tools → SARL optimizes tool usage"

        elif chars.has_patient_data and chars.num_expected_tools >= 3:
            # Complex patient cases → CRPO for faithfulness + completeness
            selected = RewardStrategyType.CRPO
            reason = "Complex clinical case → CRPO optimizes accuracy+faithfulness+completeness+safety"

        elif chars.difficulty in ("hard", "complex") and not chars.has_multiple_choice:
            # Hard open-ended tasks → CRPO for multi-objective clinical reasoning
            selected = RewardStrategyType.CRPO
            reason = "Hard open-ended clinical task → CRPO multi-objective optimization"

        elif chars.has_multiple_choice and chars.num_expected_tools <= 2:
            # Simple MC with minimal tools → GRPO is sufficient
            selected = RewardStrategyType.GRPO
            reason = "MC task with few tools → standard GRPO composite"

        elif chars.is_multi_turn and chars.has_search_expected:
            # Multi-turn search tasks → SARL
            selected = RewardStrategyType.SARL
            reason = "Multi-turn search task → SARL self-assessment tracking"

        elif chars.difficulty in ("hard",) and chars.has_multiple_choice:
            # Hard MC → MRPO for fine-grained token shaping
            selected = RewardStrategyType.MRPO
            reason = "Hard MC task → MRPO token shaping for nuanced reward"

        else:
            # Default fallback
            selected = self.config.adaptive_fallback
            reason = f"Default fallback to {selected.value}"

        # Log selection
        self._selection_log.append({
            "task_id": task.get("id", "unknown"),
            "selected": selected.value,
            "reason": reason,
            "characteristics": {
                "domain": chars.domain,
                "difficulty": chars.difficulty,
                "has_mc": chars.has_multiple_choice,
                "has_search": chars.has_search_expected,
                "num_tools": chars.num_expected_tools,
                "is_multi_turn": chars.is_multi_turn,
            },
        })

        return self._strategies[selected]

    def compute_reward(
        self,
        completions: list,
        solution: list = None,
        task: dict = None,
        **kwargs,
    ) -> list[float]:
        """Compute reward using the adaptively selected strategy.

        If task metadata is provided, selects the optimal strategy.
        Otherwise falls back to default GRPO.
        """
        if task and self.config.adaptive_task_analysis:
            strategy = self.select_strategy(task)
        else:
            strategy = self._strategies[self.config.adaptive_fallback]

        return strategy.compute_reward(
            completions, solution=solution, **kwargs,
        )

    def get_reward_breakdown(
        self,
        completions: list,
        solution: list = None,
        task: dict = None,
        **kwargs,
    ) -> list[dict]:
        if task and self.config.adaptive_task_analysis:
            strategy = self.select_strategy(task)
        else:
            strategy = self._strategies[self.config.adaptive_fallback]

        breakdowns = strategy.get_reward_breakdown(
            completions, solution=solution, **kwargs,
        )
        for b in breakdowns:
            b["meta_strategy"] = "adaptive"
            b["selected_strategy"] = strategy.name
        return breakdowns

    def get_selection_log(self) -> list[dict]:
        """Get the log of strategy selections for analysis."""
        return self._selection_log.copy()

    def get_selection_summary(self) -> dict:
        """Summarize strategy selections across all tasks."""
        if not self._selection_log:
            return {"total": 0}

        from collections import Counter
        strategy_counts = Counter(e["selected"] for e in self._selection_log)
        return {
            "total": len(self._selection_log),
            "strategy_distribution": dict(strategy_counts),
            "selection_rate": {
                k: round(v / len(self._selection_log), 3)
                for k, v in strategy_counts.items()
            },
        }


# ============================================================
# Strategy Registry & Factory
# ============================================================


STRATEGY_REGISTRY: Dict[RewardStrategyType, type] = {
    RewardStrategyType.GRPO: GRPORewardStrategy,
    RewardStrategyType.MRPO: MRPORewardStrategy,
    RewardStrategyType.SARL: SARLRewardStrategy,
    RewardStrategyType.DRPO: DRPORewardStrategy,
    RewardStrategyType.CRPO: CRPORewardStrategy,
    RewardStrategyType.ADAPTIVE: AdaptiveRewardStrategy,
}


def create_reward_strategy(
    strategy_type: str | RewardStrategyType = "grpo",
    config: Optional[RewardStrategyConfig] = None,
    **kwargs,
) -> RewardStrategy:
    """Factory function to create a reward strategy.

    Args:
        strategy_type: Strategy type name or enum
        config: Strategy configuration (optional, defaults provided)
        **kwargs: Override config parameters

    Returns:
        Initialized RewardStrategy instance
    """
    if isinstance(strategy_type, str):
        strategy_type = RewardStrategyType(strategy_type.lower())

    if config is None:
        config = RewardStrategyConfig(strategy_type=strategy_type, **kwargs)
    else:
        config.strategy_type = strategy_type

    cls = STRATEGY_REGISTRY.get(strategy_type)
    if cls is None:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    return cls(config)


def get_available_strategies() -> list[str]:
    """Get list of available strategy names."""
    return [s.value for s in RewardStrategyType]


# ============================================================
# GRPO-Compatible Wrapper (for TRL GRPOTrainer integration)
# ============================================================


def make_grpo_reward_fn(
    strategy: RewardStrategy,
    task_metadata: Optional[dict] = None,
) -> Callable:
    """Create a TRL GRPOTrainer-compatible reward function from a strategy.

    Args:
        strategy: The reward strategy to use
        task_metadata: Optional task metadata for adaptive strategy

    Returns:
        Callable matching TRL reward function signature
    """
    def reward_fn(completions, solution=None, **kwargs):
        if isinstance(strategy, AdaptiveRewardStrategy) and task_metadata:
            return strategy.compute_reward(
                completions, solution=solution, task=task_metadata, **kwargs,
            )
        return strategy.compute_reward(
            completions, solution=solution, **kwargs,
        )

    reward_fn.__name__ = f"reward_{strategy.strategy_type.value}"
    return reward_fn
