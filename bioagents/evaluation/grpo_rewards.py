"""GRPO-compatible reward functions for TRL training pipeline.

These wrappers conform to the TRL GRPOTrainer reward function signature:
    reward_fn(completions, **kwargs) -> list[float]

where `completions` is a list of completion dicts from the model.

Reference: grpo_vqa_Qwen3_token_shaping.py (MRPO framework)
"""

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from bioagents.evaluation.rewards import (
    accuracy_reward_exact_match,
    accuracy_reward_soft,
    format_reward_tool_call,
    format_reward_think_answer,
    process_reward_tool_usage,
    process_reward_reasoning_quality,
    _extract_answer_from_response,
)

# ============================================================
# ROUGE / BLEU / BERTScore helpers (lazy-loaded)
# ============================================================

_rouge_scorer_module = None
_bleu_module = None


def _get_rouge_scorer():
    """Lazy load rouge_score module."""
    global _rouge_scorer_module
    if _rouge_scorer_module is None:
        try:
            from rouge_score import rouge_scorer
            _rouge_scorer_module = rouge_scorer
        except ImportError:
            logger.warning("rouge_score not installed. ROUGE-1 will return 0.0")
    return _rouge_scorer_module


def _normalize_for_rouge(text: str) -> str:
    """Normalize text for ROUGE: strip tags, lowercase, collapse whitespace."""
    if text is None:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", str(text))
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def compute_rouge1_f1(prediction: str, reference: str) -> float:
    """Compute ROUGE-1 F1."""
    scorer_module = _get_rouge_scorer()
    if scorer_module is None:
        return 0.0

    pred_norm = _normalize_for_rouge(prediction)
    ref_norm = _normalize_for_rouge(reference)
    try:
        scorer = scorer_module.RougeScorer(["rouge1"], use_stemmer=True)
        scores = scorer.score(ref_norm, pred_norm)
        return float(scores["rouge1"].fmeasure)
    except Exception:
        return 0.0


def compute_bleu1(prediction: str, reference: str) -> float:
    """Compute BLEU-1."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = [reference.split()]
        cand_tokens = prediction.split()
        smoothie = SmoothingFunction().method4
        return sentence_bleu(
            ref_tokens, cand_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothie,
        )
    except ImportError:
        logger.warning("nltk not installed. BLEU-1 will return 0.0")
        return 0.0
    except Exception:
        return 0.0


def compute_bertscore_f1(prediction: str, reference: str, scorer=None) -> float:
    """Compute BERTScore F1 using a pre-initialized scorer or creating one."""
    # Truncate to ~450 words to stay within BiomedBERT's 512 token limit
    max_words = 450
    pred_words = prediction.split()
    ref_words = reference.split()
    if len(pred_words) > max_words:
        prediction = " ".join(pred_words[:max_words])
    if len(ref_words) > max_words:
        reference = " ".join(ref_words[:max_words])

    if scorer is not None:
        try:
            P, R, F1 = scorer.score([prediction], [reference], verbose=False)
            return F1.mean().item()
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            return 0.0
    # Fallback: try to create scorer on-the-fly
    try:
        from bert_score import BERTScorer
        device = f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
        scorer = BERTScorer(
            model_type="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            num_layers=12,
            lang="en",
            rescale_with_baseline=False,
            idf=False,
            device=device,
        )
        P, R, F1 = scorer.score([prediction], [reference], verbose=False)
        return F1.mean().item()
    except Exception as e:
        logger.warning(f"BERTScore not available: {e}")
        return 0.0


# ============================================================
# TRL-compatible reward functions for GRPO training
# ============================================================


def _extract_content(completions: list) -> list[str]:
    """Extract text content from TRL completion format.
    
    TRL format: completions = [[{"content": "...", "role": "assistant"}], ...]
    """
    contents = []
    for completion in completions:
        if isinstance(completion, list) and len(completion) > 0:
            contents.append(completion[0].get("content", ""))
        elif isinstance(completion, dict):
            contents.append(completion.get("content", ""))
        elif isinstance(completion, str):
            contents.append(completion)
        else:
            contents.append("")
    return contents


def _extract_answer_from_content(content: str) -> str:
    """Extract answer from model output.
    
    Supports:
    - <answer>X</answer> format
    - submit_answer tool call
    - Direct answer text
    """
    # Check for <answer> tags
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Check for submit_answer tool call
    try:
        parsed = json.loads(content.strip())
        if isinstance(parsed, dict) and parsed.get("name") == "submit_answer":
            args = parsed.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return args
            if isinstance(args, dict):
                answer = args.get("answer", "")
                if isinstance(answer, list):
                    answer = answer[0] if answer else ""
                return str(answer)
            return str(args)
    except json.JSONDecodeError:
        pass
    
    # Check for tool call embedded in text
    tool_match = re.search(r'"name"\s*:\s*"submit_answer".*?"answer"\s*:\s*"([^"]*)"', content)
    if tool_match:
        return tool_match.group(1).strip()
    
    return content.strip()


def grpo_accuracy_reward(
    completions: list,
    solution: list = None,
    answer: list = None,
    bert_scorer=None,
    **kwargs,
) -> list[float]:
    """GRPO-compatible accuracy reward for medical QA.
    
    Combines ROUGE-1, BLEU-1, and BERTScore (weighted 0.25/0.25/0.5).
    For multiple-choice, uses exact match on option letter.
    
    Args:
        completions: List of model completions (TRL format)
        solution: List of ground truth answers
        answer: Alternative key for ground truth answers
        bert_scorer: Pre-initialized BERTScorer instance (optional)
        
    Returns:
        List of reward scores [0.0, 1.0]
    """
    contents = _extract_content(completions)
    solutions = solution or answer or [""] * len(contents)
    
    if isinstance(solutions, str):
        solutions = [solutions] * len(contents)
    
    rewards = []
    for content, sol in zip(contents, solutions):
        student_answer = _extract_answer_from_content(content)
        ground_truth = sol.strip()
        
        if not ground_truth:
            rewards.append(0.0)
            continue
        
        # Multiple-choice exact match
        if len(ground_truth) <= 2 and ground_truth.upper() in "ABCDE":
            # Try extracting from the full content (handles <answer> tags, tool calls, etc.)
            extracted = _extract_answer_from_response(content)
            if not extracted:
                # Also try from the extracted student answer
                extracted = _extract_answer_from_response(student_answer)
            if extracted and extracted.upper() == ground_truth.upper():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
            continue
        
        # Open-ended: combine ROUGE-1, BLEU-1, BERTScore
        rouge_score = compute_rouge1_f1(student_answer, ground_truth)
        bleu_score = compute_bleu1(student_answer, ground_truth)
        bert_score = compute_bertscore_f1(student_answer, ground_truth, scorer=bert_scorer)
        
        score = rouge_score * 0.25 + bleu_score * 0.25 + bert_score * 0.5
        rewards.append(float(score))
    
    return rewards


def grpo_format_reward(completions: list, **kwargs) -> list[float]:
    """GRPO-compatible format reward.
    
    For agent tool-use: checks valid JSON tool call format.
    For QA: checks <answer>...</answer> or submit_answer format.
    
    Args:
        completions: List of model completions (TRL format)
        
    Returns:
        List of reward scores [0.0, 1.0]
    """
    contents = _extract_content(completions)
    rewards = []
    
    for content in contents:
        # Check for answer tags (QA format)
        has_answer_tags = bool(
            re.search(r"<answer>.*?</answer>", content, re.DOTALL)
        )
        
        # Check for valid tool call (agent format)
        has_valid_tool = False
        try:
            parsed = json.loads(content.strip())
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                has_valid_tool = True
        except json.JSONDecodeError:
            pass
        
        # Also check code-block tool calls
        if not has_valid_tool:
            code_match = re.search(
                r'```(?:json)?\s*\n?({.*?})\s*\n?```', content, re.DOTALL
            )
            if code_match:
                try:
                    parsed = json.loads(code_match.group(1))
                    if "name" in parsed:
                        has_valid_tool = True
                except json.JSONDecodeError:
                    pass
        
        if has_answer_tags or has_valid_tool:
            rewards.append(1.0)
        elif len(content.strip()) > 10:
            rewards.append(0.3)  # Partial credit for non-empty response
        else:
            rewards.append(0.0)
    
    return rewards


def grpo_process_reward(
    completions: list,
    solution: list = None,
    expected_tools: list = None,
    **kwargs,
) -> list[float]:
    """GRPO-compatible process reward for reasoning quality.
    
    Evaluates the quality of the agent's reasoning process using
    heuristic markers (medical terminology, structured reasoning, etc.)
    
    For full LLM-as-Judge process reward, use grpo_process_reward_llm_judge
    (requires OpenAI API).
    
    Args:
        completions: List of model completions (TRL format)
        solution: List of ground truth answers (for context)
        expected_tools: List of expected tool call sequences
        
    Returns:
        List of reward scores [0.0, 1.0]
    """
    contents = _extract_content(completions)
    solutions = solution or [""] * len(contents)
    
    if isinstance(solutions, str):
        solutions = [solutions] * len(contents)
    
    rewards = []
    for content, sol in zip(contents, solutions):
        score = process_reward_reasoning_quality(content, sol)
        rewards.append(score)
    
    return rewards


def grpo_tool_use_reward(
    completions: list,
    expected_actions: list = None,
    tool_call_logs: list = None,
    **kwargs,
) -> list[float]:
    """GRPO-compatible tool usage reward.
    
    Evaluates whether the agent made appropriate tool calls.
    Requires tool_call_logs from environment interaction.
    
    Args:
        completions: List of model completions (TRL format)
        expected_actions: List of lists of expected actions per sample
        tool_call_logs: List of lists of actual tool calls per sample
        
    Returns:
        List of reward scores [0.0, 1.0]
    """
    if not expected_actions or not tool_call_logs:
        return [0.0] * len(completions)
    
    rewards = []
    for exp, actual in zip(expected_actions, tool_call_logs):
        if not exp:
            rewards.append(1.0)
            continue
        score = process_reward_tool_usage(actual or [], exp)
        rewards.append(score)
    
    return rewards


def grpo_coherence_reward(
    completions: list,
    **kwargs,
) -> list[float]:
    """GRPO-compatible coherence reward.

    Measures response structure, logical flow, no contradictions,
    clear final answer, appropriate length.

    Returns:
        List of coherence scores in [0, 1].
    """
    from bioagents.evaluation.rewards import _compute_coherence_score

    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)
        score = _compute_coherence_score(text, is_final=True)
        rewards.append(score)
    return rewards


def grpo_assertion_reward(
    completions: list,
    nl_assertions: list = None,
    tool_call_logs: list = None,
    **kwargs,
) -> list[float]:
    """GRPO-compatible clinical assertion reward.

    Evaluates how well each completion satisfies clinician-defined
    natural language assertions from evaluation_criteria.

    Args:
        completions: List of model completions (TRL format).
        nl_assertions: List of assertion lists (one per completion).
            If a single list is passed, it applies to all completions.
        tool_call_logs: Per-completion tool call logs.

    Returns:
        List of assertion scores in [0, 1].
    """
    from bioagents.evaluation.rewards import nl_assertion_reward

    rewards = []
    for i, completion in enumerate(completions):
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        assertions_i = None
        if nl_assertions:
            if isinstance(nl_assertions, list) and nl_assertions:
                if isinstance(nl_assertions[0], list):
                    assertions_i = nl_assertions[i] if i < len(nl_assertions) else []
                else:
                    assertions_i = nl_assertions

        tc_log = []
        if tool_call_logs and i < len(tool_call_logs):
            tc_log = tool_call_logs[i] or []

        result = nl_assertion_reward(
            response=text,
            nl_assertions=assertions_i,
            tool_call_log=tc_log,
        )
        rewards.append(result["total"])

    return rewards


def grpo_composite_reward(
    completions: list,
    solution: list = None,
    answer: list = None,
    expected_actions: list = None,
    nl_assertions: list = None,
    tool_call_logs: list = None,
    bert_scorer=None,
    weights: dict = None,
    **kwargs,
) -> list[float]:
    """GRPO-compatible 6D composite reward combining all signals.

    Default weights: accuracy=0.25, format=0.10, process=0.20,
                     safety=0.20, coherence=0.10, assertion=0.15

    Args:
        completions: List of model completions (TRL format)
        solution: Ground truth answers
        answer: Alternative key for answers
        expected_actions: Expected tool call sequences
        nl_assertions: Clinician-defined assertions per completion
        tool_call_logs: Actual tool call logs
        bert_scorer: Pre-initialized BERTScorer
        weights: Custom weights dict

    Returns:
        List of composite reward scores
    """
    if weights is None:
        weights = {
            "accuracy": 0.25,
            "format": 0.10,
            "process": 0.20,
            "safety": 0.20,
            "coherence": 0.10,
            "assertion": 0.15,
        }

    accuracy_scores = grpo_accuracy_reward(
        completions, solution=solution, answer=answer,
        bert_scorer=bert_scorer, **kwargs,
    )
    format_scores = grpo_format_reward(completions, **kwargs)
    process_scores = grpo_process_reward(
        completions, solution=solution, **kwargs,
    )

    # Safety reward (lazy-loaded, graceful fallback)
    safety_scores = None
    if weights.get("safety", 0) > 0:
        try:
            safety_scores = _get_grpo_safety_reward()(completions, **kwargs)
        except Exception:
            safety_scores = [1.0] * len(completions)

    # Coherence reward
    coherence_scores = None
    if weights.get("coherence", 0) > 0:
        coherence_scores = grpo_coherence_reward(completions, **kwargs)

    # Clinical assertion reward
    assertion_scores = None
    if weights.get("assertion", 0) > 0 and nl_assertions:
        try:
            assertion_scores = grpo_assertion_reward(
                completions, nl_assertions=nl_assertions,
                tool_call_logs=tool_call_logs, **kwargs,
            )
        except Exception:
            assertion_scores = [0.5] * len(completions)

    rewards = []
    for i in range(len(completions)):
        acc = accuracy_scores[i] if i < len(accuracy_scores) else 0.0
        fmt = format_scores[i] if i < len(format_scores) else 0.0
        proc = process_scores[i] if i < len(process_scores) else 0.0
        safe = safety_scores[i] if safety_scores and i < len(safety_scores) else 1.0
        coh = coherence_scores[i] if coherence_scores and i < len(coherence_scores) else 0.5
        asrt = assertion_scores[i] if assertion_scores and i < len(assertion_scores) else 0.5

        total = (
            weights.get("accuracy", 0.25) * acc
            + weights.get("format", 0.10) * fmt
            + weights.get("process", 0.20) * proc
            + weights.get("safety", 0.20) * safe
            + weights.get("coherence", 0.10) * coh
            + weights.get("assertion", 0.15) * asrt
        )
        rewards.append(total)

    return rewards


# ============================================================
# FairGRPO: Fairness-Aware Reward Functions
# ============================================================
# Reference: FairGRPO (arXiv:2510.19893) — Hierarchical RL for
# equitable clinical reasoning with demographic-aware weighting.


# --- Demographic group extraction ---

DEMOGRAPHIC_FEATURES = {
    "age_group": {
        "pediatric": lambda age: age is not None and age < 18,
        "young_adult": lambda age: age is not None and 18 <= age < 40,
        "middle_aged": lambda age: age is not None and 40 <= age < 65,
        "elderly": lambda age: age is not None and age >= 65,
    },
    "sex": {
        "male": lambda sex: sex and sex.lower() in ("m", "male"),
        "female": lambda sex: sex and sex.lower() in ("f", "female"),
    },
    "ethnicity": {
        "white": lambda eth: eth and "white" in eth.lower(),
        "black": lambda eth: eth and "black" in eth.lower(),
        "hispanic": lambda eth: eth and "hispanic" in eth.lower(),
        "asian": lambda eth: eth and "asian" in eth.lower(),
        "other": lambda eth: eth and not any(
            k in eth.lower() for k in ("white", "black", "hispanic", "asian")
        ),
    },
}


def extract_demographic_groups(patient_data: dict) -> dict[str, str]:
    """Extract demographic group labels from patient data.

    Args:
        patient_data: Patient record dict (may contain age, sex, ethnicity, etc.)

    Returns:
        Dict mapping feature_name -> group_label (e.g., {"age_group": "elderly", "sex": "female"})
    """
    groups = {}

    age = patient_data.get("age")
    sex = patient_data.get("sex", patient_data.get("gender"))
    ethnicity = patient_data.get("ethnicity", patient_data.get("race"))

    # Age group
    for label, fn in DEMOGRAPHIC_FEATURES["age_group"].items():
        if fn(age):
            groups["age_group"] = label
            break

    # Sex
    for label, fn in DEMOGRAPHIC_FEATURES["sex"].items():
        if fn(sex):
            groups["sex"] = label
            break

    # Ethnicity
    if ethnicity:
        for label, fn in DEMOGRAPHIC_FEATURES["ethnicity"].items():
            if fn(ethnicity):
                groups["ethnicity"] = label
                break

    return groups


# --- Fairness group performance tracker ---

class FairnessTracker:
    """Tracks per-group performance for FairGRPO adaptive weighting.

    Maintains running statistics of reward scores per demographic group,
    enabling:
      1. Representation-aware weighting (upweight underrepresented groups)
      2. Performance-aware weighting (upweight groups with lower scores)
      3. Predictive parity tracking (monitor fairness gap across groups)
    """

    def __init__(self):
        self._group_scores: dict[str, dict[str, list[float]]] = {}
        # {feature: {group_label: [scores...]}}
        self._group_counts: dict[str, dict[str, int]] = {}

    def record(self, groups: dict[str, str], reward: float) -> None:
        """Record a reward for the given demographic groups."""
        for feature, label in groups.items():
            if feature not in self._group_scores:
                self._group_scores[feature] = {}
                self._group_counts[feature] = {}
            if label not in self._group_scores[feature]:
                self._group_scores[feature][label] = []
                self._group_counts[feature][label] = 0
            self._group_scores[feature][label].append(reward)
            self._group_counts[feature][label] += 1

    def get_fairness_weight(
        self,
        groups: dict[str, str],
        alpha_repr: float = 0.5,
        alpha_perf: float = 0.5,
    ) -> float:
        """Compute FairGRPO importance weight for a sample.

        Combines:
          - Representation weight: inverse of group frequency (upweight rare groups)
          - Performance weight: inverse of group mean reward (upweight struggling groups)

        Args:
            groups: Demographic group labels for this sample
            alpha_repr: Weight for representation component
            alpha_perf: Weight for performance component

        Returns:
            Importance weight >= 1.0 (higher = more emphasis on this sample)
        """
        if not groups:
            return 1.0

        weights = []
        for feature, label in groups.items():
            if feature not in self._group_counts:
                continue

            counts = self._group_counts[feature]
            total = sum(counts.values())
            if total == 0:
                continue

            n_groups = len(counts)
            group_count = counts.get(label, 1)

            # Representation weight: (total / n_groups) / group_count
            # Upweights underrepresented groups
            uniform_count = total / max(n_groups, 1)
            w_repr = uniform_count / max(group_count, 1)

            # Performance weight: global_mean / group_mean
            # Upweights groups with lower average reward
            scores = self._group_scores[feature]
            all_scores = [s for group_scores in scores.values() for s in group_scores]
            global_mean = sum(all_scores) / max(len(all_scores), 1)
            group_scores = scores.get(label, [])
            group_mean = sum(group_scores) / max(len(group_scores), 1)

            w_perf = (global_mean + 0.01) / (group_mean + 0.01)

            # Combined weight
            w = alpha_repr * w_repr + alpha_perf * w_perf
            weights.append(w)

        if not weights:
            return 1.0

        # Average across features, clamp to [0.5, 3.0]
        avg_weight = sum(weights) / len(weights)
        return max(0.5, min(3.0, avg_weight))

    def get_fairness_gap(self) -> dict[str, float]:
        """Compute the fairness gap (max - min mean reward) per feature.

        Returns:
            Dict mapping feature -> gap value. Lower is more fair.
        """
        gaps = {}
        for feature, scores in self._group_scores.items():
            means = []
            for label, label_scores in scores.items():
                if label_scores:
                    means.append(sum(label_scores) / len(label_scores))
            if len(means) >= 2:
                gaps[feature] = max(means) - min(means)
            else:
                gaps[feature] = 0.0
        return gaps

    def get_summary(self) -> dict:
        """Return a summary of per-group statistics."""
        summary = {}
        for feature, scores in self._group_scores.items():
            feature_summary = {}
            for label, label_scores in scores.items():
                n = len(label_scores)
                mean = sum(label_scores) / max(n, 1)
                feature_summary[label] = {
                    "count": n,
                    "mean_reward": round(mean, 4),
                }
            summary[feature] = feature_summary
        summary["fairness_gaps"] = self.get_fairness_gap()
        return summary

    def reset(self) -> None:
        """Reset all tracked data (e.g., between training epochs)."""
        self._group_scores.clear()
        self._group_counts.clear()


# Global fairness tracker instance
_fairness_tracker = FairnessTracker()


def get_fairness_tracker() -> FairnessTracker:
    """Get the global FairnessTracker instance."""
    return _fairness_tracker


# --- FairGRPO reward functions ---


def grpo_fairness_reward(
    completions: list,
    solution: list = None,
    patient_data: list = None,
    base_reward_fn: str = "accuracy",
    alpha_repr: float = 0.5,
    alpha_perf: float = 0.5,
    **kwargs,
) -> list[float]:
    """FairGRPO-compatible fairness-aware reward function.

    Computes a base reward (accuracy/composite) and applies demographic-aware
    importance weighting from the FairGRPO framework.

    The weight rebalances the training signal so that:
      - Underrepresented demographic groups receive higher reward signal
      - Groups where the model performs worse get amplified feedback
      - Overall training converges toward equitable performance

    Args:
        completions: List of model completions (TRL format)
        solution: Ground truth answers
        patient_data: List of patient data dicts with demographic info
        base_reward_fn: Name of base reward function to weight
        alpha_repr: Weight for representation component (0-1)
        alpha_perf: Weight for performance component (0-1)

    Returns:
        List of fairness-weighted reward scores
    """
    # Compute base rewards
    if base_reward_fn == "composite":
        base_rewards = grpo_composite_reward(completions, solution=solution, **kwargs)
    elif base_reward_fn == "accuracy":
        base_rewards = grpo_accuracy_reward(completions, solution=solution, **kwargs)
    else:
        base_rewards = grpo_accuracy_reward(completions, solution=solution, **kwargs)

    tracker = get_fairness_tracker()

    # If no patient data, return base rewards unchanged
    if not patient_data:
        return base_rewards

    weighted_rewards = []
    for i, (reward, pd) in enumerate(zip(base_rewards, patient_data)):
        if pd and isinstance(pd, dict):
            groups = extract_demographic_groups(pd)
            # Record for tracking
            tracker.record(groups, reward)
            # Apply fairness weight
            w = tracker.get_fairness_weight(groups, alpha_repr, alpha_perf)
            weighted_rewards.append(reward * w)
        else:
            weighted_rewards.append(reward)

    return weighted_rewards


def grpo_fair_composite_reward(
    completions: list,
    solution: list = None,
    answer: list = None,
    patient_data: list = None,
    expected_actions: list = None,
    tool_call_logs: list = None,
    bert_scorer=None,
    weights: dict = None,
    fairness_weight: float = 0.1,
    alpha_repr: float = 0.5,
    alpha_perf: float = 0.5,
    **kwargs,
) -> list[float]:
    """FairGRPO composite reward: base composite + fairness signal.

    Extends grpo_composite_reward with a fairness component that penalizes
    the model for demographic disparities in performance.

    reward = (1 - fairness_weight) * composite + fairness_weight * fairness_bonus

    where fairness_bonus = 1.0 if this sample helps close the fairness gap.

    Args:
        completions: Model completions
        solution: Ground truth
        patient_data: Patient demographic data
        fairness_weight: Weight for fairness component [0-1]
        alpha_repr: FairGRPO representation weight
        alpha_perf: FairGRPO performance weight
        ... (other args passed to composite reward)

    Returns:
        List of fair composite reward scores
    """
    if weights is None:
        weights = {"accuracy": 0.4, "format": 0.2, "process": 0.4}

    # Base composite scores
    base_scores = grpo_composite_reward(
        completions, solution=solution, answer=answer,
        expected_actions=expected_actions, tool_call_logs=tool_call_logs,
        bert_scorer=bert_scorer, weights=weights, **kwargs,
    )

    if not patient_data or fairness_weight <= 0:
        return base_scores

    tracker = get_fairness_tracker()
    fair_scores = []

    for i, (base, pd) in enumerate(zip(base_scores, patient_data)):
        if pd and isinstance(pd, dict):
            groups = extract_demographic_groups(pd)
            tracker.record(groups, base)

            # Fairness bonus: higher for underperforming groups
            w = tracker.get_fairness_weight(groups, alpha_repr, alpha_perf)
            # Normalize weight to bonus in [0, 1]: (w - 0.5) / 2.5
            fairness_bonus = min(1.0, max(0.0, (w - 0.5) / 2.5))

            fair_score = (1 - fairness_weight) * base + fairness_weight * fairness_bonus
            fair_scores.append(fair_score)
        else:
            fair_scores.append(base)

    return fair_scores


# ============================================================
# Registry for GRPO reward functions
# ============================================================


def _get_grpo_safety_reward():
    """Lazy-load safety reward to avoid circular imports."""
    from bioagents.evaluation.safety_eval import grpo_safety_reward
    return grpo_safety_reward


class _LazyRewardRegistry(dict):
    """Registry that supports lazy-loaded reward functions."""
    
    _lazy_loaders = {
        "safety": _get_grpo_safety_reward,
    }
    
    def __contains__(self, key):
        return super().__contains__(key) or key in self._lazy_loaders
    
    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if key in self._lazy_loaders:
            fn = self._lazy_loaders[key]()
            self[key] = fn  # cache it
            return fn
        raise KeyError(key)
    
    def keys(self):
        return list(super().keys()) + list(self._lazy_loaders.keys())


# ============================================================
# LLM-as-Judge Reward (MediX-R1 inspired)
# ============================================================

_LLM_JUDGE_PROMPT = """\
You are an expert medical evaluator. Score the following medical response on a scale of 0-10.

## Reference Answer
{reference}

## Model Response
{response}

## Evaluation Criteria
1. **Accuracy (0-10)**: Is the response medically correct? Does it match the reference?
2. **Completeness (0-10)**: Does it cover all key clinical points from the reference?
3. **Reasoning (0-10)**: Does it show logical, evidence-based clinical reasoning?
4. **Safety (0-10)**: Does it avoid harmful advice, check contraindications?

Respond with ONLY a JSON object:
{{"accuracy": <0-10>, "completeness": <0-10>, "reasoning": <0-10>, "safety": <0-10>}}"""

_llm_judge_client = None


def _get_llm_judge_client():
    """Get or create LLM judge client. Supports OpenAI API or local vLLM."""
    global _llm_judge_client
    if _llm_judge_client is not None:
        return _llm_judge_client

    judge_backend = os.environ.get("LLM_JUDGE_BACKEND", "openai")
    judge_model = os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-mini")
    judge_api_key = os.environ.get("LLM_JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    judge_base_url = os.environ.get("LLM_JUDGE_BASE_URL", "")

    if judge_backend == "openai":
        try:
            import openai
            client_kwargs = {"api_key": judge_api_key}
            if judge_base_url:
                client_kwargs["base_url"] = judge_base_url
            _llm_judge_client = {
                "type": "openai",
                "client": openai.OpenAI(**client_kwargs),
                "model": judge_model,
            }
        except ImportError:
            logger.warning("openai not installed. LLM-as-Judge reward disabled.")
            _llm_judge_client = {"type": "none"}
    elif judge_backend == "vllm":
        try:
            import openai
            base_url = judge_base_url or "http://localhost:8000/v1"
            _llm_judge_client = {
                "type": "openai",
                "client": openai.OpenAI(api_key="none", base_url=base_url),
                "model": judge_model,
            }
        except ImportError:
            _llm_judge_client = {"type": "none"}
    else:
        _llm_judge_client = {"type": "none"}

    return _llm_judge_client


def _llm_judge_score(response_text: str, reference_text: str) -> float:
    """Score a single response using LLM-as-Judge.

    Returns a float in [0, 1] (average of 4 criteria, each normalized to 0-1).
    Falls back to 0.5 on any error.
    """
    client_info = _get_llm_judge_client()
    if client_info.get("type") == "none":
        return 0.5

    prompt = _LLM_JUDGE_PROMPT.format(
        reference=reference_text[:2000],
        response=response_text[:2000],
    )

    try:
        client = client_info["client"]
        resp = client.chat.completions.create(
            model=client_info["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{[^}]+\}", raw)
        if json_match:
            scores = json.loads(json_match.group())
            avg = sum(
                float(scores.get(k, 5))
                for k in ("accuracy", "completeness", "reasoning", "safety")
            ) / 4.0
            return max(0.0, min(1.0, avg / 10.0))
    except Exception as e:
        logger.debug(f"LLM-as-Judge error: {e}")

    return 0.5


def grpo_llm_judge_reward(
    completions: list,
    solution: list = None,
    answer: list = None,
    **kwargs,
) -> list[float]:
    """GRPO-compatible LLM-as-Judge reward for open-ended medical responses.

    Uses an external LLM (GPT-4o-mini by default, or local vLLM) to evaluate
    the quality of model-generated medical responses against reference answers.

    Inspired by MediX-R1 (ICLR 2026) LLM-as-judge accuracy reward.

    Environment variables:
        LLM_JUDGE_BACKEND: "openai" (default) | "vllm"
        LLM_JUDGE_MODEL: Model name (default "gpt-4o-mini")
        LLM_JUDGE_API_KEY: API key (falls back to OPENAI_API_KEY)
        LLM_JUDGE_BASE_URL: Custom base URL (for vLLM or Azure)

    Args:
        completions: Model completions (TRL format)
        solution: Reference answers
        answer: Alternative key for reference answers

    Returns:
        List of float scores in [0, 1]
    """
    refs = solution or answer or []
    rewards = []

    for i, completion in enumerate(completions):
        text = _extract_text(completion)
        ref = refs[i] if i < len(refs) else ""

        if not ref or not text:
            rewards.append(0.5)
            continue

        score = _llm_judge_score(text, str(ref))
        rewards.append(score)

    return rewards


def _get_mrpo_strategy_reward():
    """Lazy-load MRPO strategy reward."""
    from bioagents.evaluation.reward_strategies import create_reward_strategy, make_grpo_reward_fn
    strategy = create_reward_strategy("mrpo")
    return make_grpo_reward_fn(strategy)


def _get_sarl_strategy_reward():
    """Lazy-load SARL strategy reward."""
    from bioagents.evaluation.reward_strategies import create_reward_strategy, make_grpo_reward_fn
    strategy = create_reward_strategy("sarl")
    return make_grpo_reward_fn(strategy)


def _get_adaptive_strategy_reward():
    """Lazy-load Adaptive strategy reward."""
    from bioagents.evaluation.reward_strategies import create_reward_strategy, make_grpo_reward_fn
    strategy = create_reward_strategy("adaptive")
    return make_grpo_reward_fn(strategy)


# ============================================================
# Medical Embedding Reward (BiomedBERT / MedCPT semantic similarity)
# ============================================================


def grpo_medical_embedding_reward(
    completions: list,
    solution: list = None,
    answer: list = None,
    **kwargs,
) -> list[float]:
    """GRPO-compatible semantic similarity reward using medical embeddings.

    Computes cosine similarity between model response and reference answer
    using a medical domain-specific embedding model (BiomedBERT or MedCPT).

    Falls back to sentence-transformers all-MiniLM-L6-v2 if medical models
    are unavailable.

    Inspired by MediX-R1 medical embedding reward.

    Environment variables:
        MEDICAL_EMBED_MODEL: Model name (default "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        MEDICAL_EMBED_DEVICE: Device (default "cpu")

    Args:
        completions: Model completions (TRL format).
        solution: Reference answers.
        answer: Alternative key for reference answers.

    Returns:
        List of float scores in [0, 1].
    """
    refs = solution or answer or kwargs.get("solution", [])
    if not refs:
        return [0.5] * len(completions)

    # Extract response text from completions
    responses = []
    for c in completions:
        if isinstance(c, dict):
            text = c.get("content", "") or c.get("text", "")
        elif isinstance(c, list):
            text = c[-1].get("content", "") if c else ""
        else:
            text = str(c)
        responses.append(_extract_answer_from_response(text) or text[:500])

    ref_texts = []
    for r in refs:
        ref_texts.append(str(r) if r else "")

    # Pad if needed
    while len(ref_texts) < len(responses):
        ref_texts.append(ref_texts[-1] if ref_texts else "")

    try:
        return _compute_embedding_similarity(responses, ref_texts[:len(responses)])
    except Exception as e:
        logger.warning(f"Medical embedding reward failed: {e}, falling back to 0.5")
        return [0.5] * len(completions)


def _compute_embedding_similarity(responses: list, references: list) -> list:
    """Compute cosine similarity using sentence-transformers.

    Lazily loads the embedding model on first call.
    """
    global _EMBED_MODEL, _EMBED_TOKENIZER

    if "_EMBED_MODEL" not in globals() or _EMBED_MODEL is None:
        model_name = os.environ.get(
            "MEDICAL_EMBED_MODEL",
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        )
        device = os.environ.get("MEDICAL_EMBED_DEVICE", "cpu")

        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer(model_name, device=device)
            _EMBED_TOKENIZER = None
            logger.info(f"Loaded medical embedding model: {model_name}")
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer
                fallback = "all-MiniLM-L6-v2"
                _EMBED_MODEL = SentenceTransformer(fallback, device=device)
                _EMBED_TOKENIZER = None
                logger.info(f"Loaded fallback embedding model: {fallback}")
            except ImportError:
                logger.warning("sentence-transformers not installed, using simple overlap")
                _EMBED_MODEL = "FALLBACK"
                _EMBED_TOKENIZER = None

    if _EMBED_MODEL == "FALLBACK":
        # Simple word-overlap fallback
        scores = []
        for resp, ref in zip(responses, references):
            resp_tokens = set(resp.lower().split())
            ref_tokens = set(ref.lower().split())
            if not ref_tokens:
                scores.append(0.5)
                continue
            overlap = len(resp_tokens & ref_tokens) / max(len(ref_tokens), 1)
            scores.append(min(overlap, 1.0))
        return scores

    import numpy as np

    resp_embeddings = _EMBED_MODEL.encode(responses, show_progress_bar=False)
    ref_embeddings = _EMBED_MODEL.encode(references, show_progress_bar=False)

    scores = []
    for r_emb, ref_emb in zip(resp_embeddings, ref_embeddings):
        cosine_sim = float(
            np.dot(r_emb, ref_emb)
            / (np.linalg.norm(r_emb) * np.linalg.norm(ref_emb) + 1e-8)
        )
        # Map from [-1, 1] to [0, 1]
        score = (cosine_sim + 1.0) / 2.0
        scores.append(max(0.0, min(1.0, score)))

    return scores


# Initialize global embedding model cache
_EMBED_MODEL = None
_EMBED_TOKENIZER = None


def _get_drpo_strategy_reward():
    """Lazy-load DRPO strategy reward."""
    from bioagents.evaluation.reward_strategies import create_reward_strategy, make_grpo_reward_fn
    strategy = create_reward_strategy("drpo")
    return make_grpo_reward_fn(strategy)


def _get_crpo_strategy_reward():
    """Lazy-load CRPO strategy reward."""
    from bioagents.evaluation.reward_strategies import create_reward_strategy, make_grpo_reward_fn
    strategy = create_reward_strategy("crpo")
    return make_grpo_reward_fn(strategy)


# ============================================================
# C-4: Med-PRM Style Step-level Process Reward
# Reference: Med-PRM (EMNLP 2025, arXiv:2506.11474)
#   "Medical Process Reward Model" — guideline-verifiable intermediate
#   reasoning steps in clinical trajectories are rewarded at each turn,
#   not just at final answer. Complements existing 5D final reward.
#
# Also inspired by DAPO (arXiv:2412.18279, NeurIPS 2025) dense step reward.
# ============================================================

# Clinical reasoning step order for medical diagnosis tasks
_CLINICAL_STEP_ORDER = [
    # Stage 1: Information gathering
    ("symptom", ["get_symptoms", "get_chief_complaint", "get_history", "get_vitals",
                  "get_patient_info", "get_lab_results", "get_lab_trend"]),
    # Stage 2: Investigation
    ("investigation", ["order_labs", "order_imaging", "get_imaging_results",
                        "search_literature", "search_pubmed", "get_guidelines"]),
    # Stage 3: Differential diagnosis
    ("differential", ["generate_differential", "rank_differential", "check_criteria",
                       "assess_risk", "calculate_score"]),
    # Stage 4: Treatment / management plan
    ("treatment", ["prescribe_medication", "recommend_treatment", "get_drug_info",
                    "check_drug_interaction", "create_care_plan", "submit_answer"]),
]

_STEP_STAGE_MAP: dict[str, int] = {}
for _stage_idx, (_stage_name, _tools) in enumerate(_CLINICAL_STEP_ORDER):
    for _tool in _tools:
        _STEP_STAGE_MAP[_tool.lower()] = _stage_idx


def _score_trajectory_steps(tool_calls: list[str], domain: str = "") -> float:
    """Score a trajectory's step ordering quality (0.0-1.0).

    Rewards trajectories that follow the expected clinical reasoning order
    (symptom → investigation → differential → treatment). Out-of-order
    transitions are penalized.

    Args:
        tool_calls: Ordered list of tool names called in the trajectory.
        domain: Optional domain hint (currently unused, for future extension).

    Returns:
        Step-order score in [0, 1].
    """
    if not tool_calls:
        return 0.5  # neutral for missing data

    stages_used = []
    for tool in tool_calls:
        stage = _STEP_STAGE_MAP.get(tool.lower().strip())
        if stage is not None:
            stages_used.append(stage)

    if not stages_used:
        return 0.5  # no recognizable clinical tools → neutral

    # Reward: fraction of consecutive pairs that are non-decreasing (ordered)
    ordered_pairs = sum(
        1 for a, b in zip(stages_used, stages_used[1:]) if b >= a
    )
    total_pairs = len(stages_used) - 1
    if total_pairs <= 0:
        return 0.7  # single-step trajectory gets mild reward

    ordering_score = ordered_pairs / total_pairs  # [0, 1]

    # Bonus: trajectory reaches the treatment/submit stage
    reach_bonus = 0.15 if stages_used[-1] == len(_CLINICAL_STEP_ORDER) - 1 else 0.0

    return min(1.0, ordering_score * 0.85 + reach_bonus)


def grpo_stepwise_medical_reward(
    completions: list,
    tool_call_logs: list = None,
    domain: list = None,
    step_reward_weight: float = 0.1,
    base_reward_fn: str = "composite",
    **kwargs,
) -> list[float]:
    """Med-PRM style step-level reward combined with base outcome reward.

    Evaluates each trajectory turn-by-turn for clinical reasoning order
    (symptom → investigation → differential → treatment), following the
    Med-PRM (EMNLP 2025) principle that intermediate step quality should
    be rewarded, not just final answer correctness.

    Args:
        completions: List of model completions (TRL format).
        tool_call_logs: Per-completion list of tool call dicts.
            Each entry: list of {"name": str, "arguments": dict} or str.
        domain: Optional per-completion domain names (for future use).
        step_reward_weight: Weight of step-level reward vs base reward.
            Final = (1 - w) * base_reward + w * step_reward.
            Default 0.10 (conservative: 90% outcome, 10% process).
        base_reward_fn: Base reward to combine with step reward.
            One of: "composite", "accuracy", "tool_use".

    Returns:
        List of combined rewards in [0, 1].
    """
    # Compute base rewards
    if base_reward_fn == "composite":
        base_rewards = grpo_composite_reward(completions, **kwargs)
    elif base_reward_fn == "tool_use":
        base_rewards = grpo_tool_use_reward(completions, **kwargs)
    else:
        base_rewards = grpo_accuracy_reward(completions, **kwargs)

    step_rewards = []
    for i, completion in enumerate(completions):
        # Extract tool call sequence for this completion
        tc_log = []
        if tool_call_logs and i < len(tool_call_logs):
            raw = tool_call_logs[i] or []
            for entry in raw:
                if isinstance(entry, dict):
                    name = entry.get("name") or entry.get("tool") or ""
                    if name:
                        tc_log.append(name)
                elif isinstance(entry, str):
                    tc_log.append(entry)

        dom = ""
        if domain and i < len(domain):
            dom = domain[i] or ""

        step_score = _score_trajectory_steps(tc_log, dom)
        step_rewards.append(step_score)

    # Combine: weighted sum
    w = max(0.0, min(0.5, step_reward_weight))  # cap at 0.5 for safety
    combined = [
        (1.0 - w) * br + w * sr
        for br, sr in zip(base_rewards, step_rewards)
    ]
    return combined


GRPO_REWARD_REGISTRY: Dict[str, Callable] = _LazyRewardRegistry({
    "accuracy": grpo_accuracy_reward,
    "format": grpo_format_reward,
    "process": grpo_process_reward,
    "tool_use": grpo_tool_use_reward,
    "coherence": grpo_coherence_reward,
    "assertion": grpo_assertion_reward,
    "composite": grpo_composite_reward,
    # LLM-as-Judge reward (MediX-R1 inspired)
    "llm_judge": grpo_llm_judge_reward,
    # Medical embedding semantic similarity reward
    "medical_embedding": grpo_medical_embedding_reward,
    # FairGRPO reward functions
    "fairness": grpo_fairness_reward,
    "fair_composite": grpo_fair_composite_reward,
    # Med-PRM style step-level process reward (C-4, arXiv:2506.11474)
    "stepwise_medical": grpo_stepwise_medical_reward,
})

# Register strategy-based rewards (lazy-loaded to avoid circular imports)
GRPO_REWARD_REGISTRY._lazy_loaders["mrpo"] = _get_mrpo_strategy_reward
GRPO_REWARD_REGISTRY._lazy_loaders["sarl"] = _get_sarl_strategy_reward
GRPO_REWARD_REGISTRY._lazy_loaders["adaptive"] = _get_adaptive_strategy_reward
GRPO_REWARD_REGISTRY._lazy_loaders["drpo"] = _get_drpo_strategy_reward
GRPO_REWARD_REGISTRY._lazy_loaders["crpo"] = _get_crpo_strategy_reward


def get_grpo_reward_functions(names: list[str]) -> list[Callable]:
    """Get GRPO reward functions by name.
    
    Args:
        names: List of reward function names
        
    Returns:
        List of reward function callables
    """
    funcs = []
    for name in names:
        if name not in GRPO_REWARD_REGISTRY:
            raise ValueError(
                f"Unknown GRPO reward '{name}'. "
                f"Available: {list(GRPO_REWARD_REGISTRY.keys())}"
            )
        funcs.append(GRPO_REWARD_REGISTRY[name])
    return funcs
