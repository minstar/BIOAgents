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
            return parsed.get("arguments", {}).get("answer", "")
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


def grpo_composite_reward(
    completions: list,
    solution: list = None,
    answer: list = None,
    expected_actions: list = None,
    tool_call_logs: list = None,
    bert_scorer=None,
    weights: dict = None,
    **kwargs,
) -> list[float]:
    """GRPO-compatible composite reward combining all signals.
    
    Default weights: accuracy=0.4, format=0.2, process=0.4
    
    Args:
        completions: List of model completions (TRL format)
        solution: Ground truth answers
        answer: Alternative key for answers
        expected_actions: Expected tool call sequences
        tool_call_logs: Actual tool call logs
        bert_scorer: Pre-initialized BERTScorer
        weights: Custom weights dict
        
    Returns:
        List of composite reward scores
    """
    if weights is None:
        weights = {"accuracy": 0.4, "format": 0.2, "process": 0.4}
    
    accuracy_scores = grpo_accuracy_reward(
        completions, solution=solution, answer=answer,
        bert_scorer=bert_scorer, **kwargs,
    )
    format_scores = grpo_format_reward(completions, **kwargs)
    process_scores = grpo_process_reward(
        completions, solution=solution, **kwargs,
    )
    
    rewards = []
    for acc, fmt, proc in zip(accuracy_scores, format_scores, process_scores):
        total = (
            weights["accuracy"] * acc
            + weights["format"] * fmt
            + weights["process"] * proc
        )
        rewards.append(total)
    
    return rewards


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


GRPO_REWARD_REGISTRY: Dict[str, Callable] = _LazyRewardRegistry({
    "accuracy": grpo_accuracy_reward,
    "format": grpo_format_reward,
    "process": grpo_process_reward,
    "tool_use": grpo_tool_use_reward,
    "composite": grpo_composite_reward,
})


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
