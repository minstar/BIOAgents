"""Reward function for BIOAgents medical GRPO on veRL.

Supports optional cosine length-scaling reward (Yeo et al., 2025,
"Demystifying Long Chain-of-Thought Reasoning in LLMs", arXiv:2502.03373).
Enable via environment variable COSINE_REWARD=1.
"""
import json
import math
import os
import re
from typing import Any, Optional

_DEBUG_LOG = os.environ.get("REWARD_DEBUG_LOG", "")
_debug_count = 0

# ── Degenerate response detection ────────────────────────────────────
# Detect repetitive/degenerate responses and assign penalty reward.
# Enable via DEGENERATE_FILTER=1 (default: disabled for backwards compat)
DEGENERATE_FILTER_ENABLED = os.environ.get("DEGENERATE_FILTER", "") == "1"
DEGENERATE_REWARD = float(os.environ.get("DEGENERATE_REWARD", "-1.0"))
DEGENERATE_NGRAM_THRESHOLD = float(os.environ.get("DEGENERATE_NGRAM_THRESHOLD", "0.3"))
DEGENERATE_MIN_LENGTH = int(os.environ.get("DEGENERATE_MIN_LENGTH", "200"))


def _strip_tool_markup(text: str) -> str:
    """Strip tool/turn markup to isolate unique model-generated content."""
    # Remove tool responses (environment output, not model text)
    cleaned = re.sub(r"<tool_response>.*?</tool_response>", " ", text, flags=re.DOTALL)
    # Remove tool call XML tags
    cleaned = re.sub(r"</?(?:tool_call|function=[^>]*)>", " ", cleaned)
    # Remove JSON arguments in tool calls
    cleaned = re.sub(r"\{[^}]{0,500}\}", " ", cleaned)
    # Remove turn markers that repeat in every multi-turn exchange
    cleaned = re.sub(r"</?think>", " ", cleaned)
    cleaned = re.sub(r"\b(user|assistant|system)\b", " ", cleaned)
    # Remove common filler phrases that naturally repeat across turns
    cleaned = re.sub(r"I'm here and ready to help[.!]?", " ", cleaned)
    cleaned = re.sub(r"What would you like to ask or discuss\?", " ", cleaned)
    cleaned = re.sub(r"please (?:try )?typ(?:e|ing) (?:it )?(?:out )?again", " ", cleaned, flags=re.IGNORECASE)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_degenerate_response(text: str) -> bool:
    """Detect degenerate/repetitive responses.

    Strips tool markup first to avoid false positives from repeated XML
    structure in multi-turn tool-use responses.

    Checks for:
    1. High ratio of repeated n-grams (n=4) in model-generated text
    2. Exact phrase repetition loops
    3. Very short responses with no tool calls (model gave up)
    """
    if len(text) < DEGENERATE_MIN_LENGTH:
        if "<function=" not in text and "<tool_call>" not in text:
            return True
        return False

    # Strip tool markup before checking repetition
    cleaned = _strip_tool_markup(text)

    # Check the latter half for repetition
    half = cleaned[len(cleaned) // 2:]
    words = half.lower().split()

    if len(words) < 30:
        return False

    # 4-gram repetition ratio on cleaned text
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    if not ngrams:
        return False
    unique_ratio = len(set(ngrams)) / len(ngrams)
    # Low unique ratio = high repetition (threshold: <15% unique = degenerate)
    if unique_ratio < 0.15:
        return True

    # Check for exact phrase loops (same phrase repeated many times)
    for phrase_len in [5, 10, 20]:
        if len(words) < phrase_len * 6:
            continue
        phrases = [tuple(words[i:i+phrase_len]) for i in range(0, len(words) - phrase_len, phrase_len)]
        if phrases:
            most_common = max(set(phrases), key=phrases.count)
            if phrases.count(most_common) >= len(phrases) * 0.6:
                return True

    return False

# ── Cosine length-scaling reward (arXiv:2502.03373) ──────────────────
# CosFn(t, T, η_min, η_max) = η_min + ½(η_max − η_min)(1 + cos(tπ/T))
# Goes from η_max at t=0 to η_min at t=T.
COSINE_REWARD_ENABLED = os.environ.get("COSINE_REWARD", "") == "1"

# Max response length in tokens (must match data.max_response_length)
COSINE_L_MAX = int(os.environ.get("COSINE_L_MAX", "12288"))
# Chars-per-token estimate for converting solution_str length to tokens
COSINE_CHARS_PER_TOKEN = float(os.environ.get("COSINE_CHARS_PER_TOKEN", "5.0"))

# Reward at L_gen=0 / L_gen=L_max for correct answers
COSINE_R0_CORRECT = float(os.environ.get("COSINE_R0_CORRECT", "1.1"))
COSINE_RL_CORRECT = float(os.environ.get("COSINE_RL_CORRECT", "0.7"))
# Reward at L_gen=0 / L_gen=L_max for wrong answers
COSINE_R0_WRONG = float(os.environ.get("COSINE_R0_WRONG", "0.0"))
COSINE_RL_WRONG = float(os.environ.get("COSINE_RL_WRONG", "-0.3"))
# Penalty when response hits max length (clipped)
COSINE_R_EXCEED = float(os.environ.get("COSINE_R_EXCEED", "-0.5"))


def _cosine_fn(t: float, T: float, eta_min: float, eta_max: float) -> float:
    """Cosine annealing: eta_max at t=0, eta_min at t=T."""
    t = max(0.0, min(t, T))
    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(t * math.pi / T))


def cosine_length_reward(is_correct: bool, response_len_chars: int) -> float:
    """Compute cosine length-scaling reward.

    Adapts Yeo et al. (2025) for multi-turn agentic setting:
    - Correct short → highest reward (1.1)
    - Correct long → reduced reward (0.7)
    - Wrong short → zero (0.0)
    - Wrong long → penalty (-0.3)
    - Exceeded max length → strong penalty (-0.5)
    """
    est_tokens = response_len_chars / COSINE_CHARS_PER_TOKEN

    if est_tokens >= COSINE_L_MAX:
        return COSINE_R_EXCEED

    if is_correct:
        return _cosine_fn(est_tokens, COSINE_L_MAX, COSINE_RL_CORRECT, COSINE_R0_CORRECT)
    else:
        return _cosine_fn(est_tokens, COSINE_L_MAX, COSINE_RL_WRONG, COSINE_R0_WRONG)

# Valid tool names from tool_config_consolidated.yaml (25 tools)
VALID_TOOLS = frozenset({
    "think", "submit_answer",
    "search_pubmed", "search_medical_wiki", "search_medical_literature",
    "retrieve_evidence",
    "analyze_answer_options", "get_differential_diagnosis",
    "check_diagnostic_criteria", "compare_treatments",
    "get_drug_info", "check_interaction", "check_medication_safety",
    "get_patient_info", "get_patient_records",
    "get_vital_signs", "get_lab_results", "order_test",
    "analyze_medical_image", "get_image_report", "search_imaging_knowledge",
    "calculate_clinical_score", "search_clinical_guidelines",
    "record_clinical_action", "perform_assessment",
    "assess_obstetric_status", "get_ed_status",
})

# Penalty per invalid tool call (tool name not in VALID_TOOLS)
INVALID_TOOL_PENALTY = 0.2


def count_invalid_tool_calls(solution_str: str) -> int:
    """Count tool calls to names not in VALID_TOOLS."""
    tool_calls = re.findall(r"<function=([^>]+)>", solution_str)
    return sum(1 for name in tool_calls if name.strip() not in VALID_TOOLS)


def extract_answer_letter(text: str) -> Optional[str]:
    """Extract answer letter from model response.

    Uses the LAST match in multi-turn responses to avoid picking up
    intermediate tool responses or earlier reasoning attempts.
    """
    # Pattern 1: "Answer: X" — use last match (critical for multi-turn)
    matches = re.findall(r"Answer:\s*([A-E])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # Pattern 2: "the answer is X" — use last match
    matches = re.findall(r"the answer is\s*([A-E])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # Pattern 3: standalone letter at end "D" or "(D)"
    match = re.search(r"\(?([A-E])\)?\s*$", text.strip())
    if match:
        return match.group(1).upper()

    return None


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> float:
    """Compute reward for medical QA tasks.

    For MCQA: binary reward (1.0 correct, 0.0 wrong) + format bonus.
    For open-ended: partial credit based on keyword overlap.
    """
    if extra_info is None:
        extra_info = {}

    # Parse extra_info if it's a string
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)

    # During validation, use binary accuracy (no cosine scaling)
    is_validate = extra_info.get("validate", False)

    # Degenerate response filter: applied AFTER normal scoring below
    # (we need to know if the answer was correct first)

    # Normalize ground_truth: extract answer letter from formats like "ANSWER: (D)", "(D)", "D"
    if ground_truth:
        gt_stripped = ground_truth.strip()
        # Try explicit patterns first: "ANSWER: (X)" or "(X)" at end
        gt_match = re.search(r"(?:ANSWER:\s*)?[\(]([A-E])[\)]", gt_stripped, re.IGNORECASE)
        if gt_match:
            ground_truth = gt_match.group(1).upper()
        elif len(gt_stripped) == 1 and gt_stripped.upper() in "ABCDE":
            ground_truth = gt_stripped.upper()

    has_options = extra_info.get("has_options", bool(ground_truth and len(ground_truth) == 1))

    if has_options:
        # MCQA scoring
        predicted = extract_answer_letter(solution_str)
        correct = ground_truth.strip().upper() if ground_truth else ""

        # Debug logging — first 20 samples
        global _debug_count
        if _DEBUG_LOG and _debug_count < 20:
            _debug_count += 1
            # Show first 200 + last 200 chars to understand format
            head = repr(solution_str[:200])
            tail = repr(solution_str[-200:]) if len(solution_str) > 200 else ""
            has_tool_call = "<tool_call>" in solution_str
            has_think = "<think>" in solution_str
            print(f"[REWARD #{_debug_count}] pred={predicted} gt={correct} len={len(solution_str)} "
                  f"tool_call={has_tool_call} think={has_think}")
            print(f"  HEAD: {head}")
            if tail:
                print(f"  TAIL: {tail}")

        if predicted is None:
            is_correct = False
            base_reward = 0.0
        else:
            is_correct = predicted == correct
            accuracy = 1.0 if is_correct else 0.0
            # Format bonus: only for correct answers to avoid rewarding wrong-but-formatted
            format_bonus = 0.1 if (accuracy > 0 and re.search(r"Answer:\s*[A-E]", solution_str)) else 0.0
            base_reward = accuracy + format_bonus

        # Cosine length-scaling reward (replaces base_reward when enabled, training only)
        if COSINE_REWARD_ENABLED and not is_validate:
            base_reward = cosine_length_reward(is_correct, len(solution_str))
            # Still add format bonus on top for correct answers
            if is_correct and re.search(r"Answer:\s*[A-E]", solution_str):
                base_reward += 0.1

        # Penalty for calling tools not in the provided tool list
        n_invalid = count_invalid_tool_calls(solution_str)
        penalty = n_invalid * INVALID_TOOL_PENALTY
        reward = base_reward - penalty

        # Degenerate filter: wrong answer + repetitive = hard penalty
        if DEGENERATE_FILTER_ENABLED and not is_validate and not is_correct:
            if is_degenerate_response(solution_str):
                return DEGENERATE_REWARD
        return reward
    else:
        # Open-ended: simple keyword overlap scoring
        if not ground_truth or not solution_str:
            base_reward = 0.0
            is_correct = False
        else:
            gt_words = set(ground_truth.lower().split())
            pred_words = set(solution_str.lower().split())

            if not gt_words:
                base_reward = 0.0
                is_correct = False
            else:
                overlap = len(gt_words & pred_words) / len(gt_words)
                base_reward = min(overlap, 1.0)
                is_correct = overlap > 0.5

        # Cosine length-scaling reward (training only)
        # For open-ended questions: only apply cosine scaling when answer is
        # meaningfully correct (overlap > 0.5).  When incorrect, keep the raw
        # overlap score (0.0-0.5) instead of the cosine wrong-answer schedule,
        # which produces persistent negative signal on ~27% of data and makes
        # the model overly cautious / non-answering.
        if COSINE_REWARD_ENABLED and not is_validate:
            if is_correct:
                # Good answer → reward with length efficiency bonus
                base_reward = cosine_length_reward(True, len(solution_str))
            # else: keep base_reward = overlap (0.0–0.5), no cosine penalty

        # Penalty for calling tools not in the provided tool list
        n_invalid = count_invalid_tool_calls(solution_str)
        penalty = n_invalid * INVALID_TOOL_PENALTY
        reward = base_reward - penalty

        # Degenerate filter: low-quality + repetitive = hard penalty
        if DEGENERATE_FILTER_ENABLED and not is_validate and not is_correct:
            if is_degenerate_response(solution_str):
                return DEGENERATE_REWARD
        return reward
