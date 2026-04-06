"""Reward function for BIOAgents medical GRPO on veRL."""
import json
import os
import re
from typing import Any, Optional

_DEBUG_LOG = os.environ.get("REWARD_DEBUG_LOG", "")
_debug_count = 0


def extract_answer_letter(text: str) -> Optional[str]:
    """Extract answer letter from model response."""
    # Pattern 1: "Answer: X" at end
    match = re.search(r"Answer:\s*([A-E])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 2: "the answer is X"
    match = re.search(r"the answer is\s*([A-E])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

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
            return 0.0  # no answer extracted = 0

        # Format bonus: model used "Answer: X" format
        format_bonus = 0.1 if re.search(r"Answer:\s*[A-E]", solution_str) else 0.0

        accuracy = 1.0 if predicted == correct else 0.0
        return accuracy + format_bonus
    else:
        # Open-ended: simple keyword overlap scoring
        if not ground_truth or not solution_str:
            return 0.0

        gt_words = set(ground_truth.lower().split())
        pred_words = set(solution_str.lower().split())

        if not gt_words:
            return 0.0

        overlap = len(gt_words & pred_words) / len(gt_words)
        return min(overlap, 1.0)
