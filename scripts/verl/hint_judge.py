"""
Bidirectional Hint Judge for BT-OPD (Bidirectional Truncated On-Policy Distillation).

Extracts corrective hints from both positive and negative trajectories,
extending OpenClaw's positive-only approach to bidirectional signals.

Usage:
    judge = HintJudge(model_name="Qwen3.5-9B", api_base="http://localhost:8000")
    hints = await judge.extract_hints_batch(trajectories)
"""

import re
import asyncio
from dataclasses import dataclass
from typing import Optional

HINT_JUDGE_SYSTEM_PROMPT = """\
You are a process reward model used for bidirectional hindsight hint extraction.
You are given:
1) The assistant response at turn t.
2) The next state at turn t+1, along with its **role**.
3) The **outcome** of this trajectory (CORRECT or INCORRECT).

## Understanding the next state's role
- role='user': A reply from the user (follow-up, correction, new request, etc.).
- role='tool': The return value of a tool the assistant invoked. This content was NOT \
available before the assistant's action — it exists BECAUSE the assistant called the tool. \
A successful, non-error tool output generally means the assistant's action was appropriate; \
do NOT treat it as information the assistant should have already known.

## Bidirectional Hint Extraction
For CORRECT outcomes:
- Extract what the model did well that led to the correct answer.
- Highlight key reasoning steps or tool-use decisions worth reinforcing.
- The hint should capture the successful strategy concisely.

For INCORRECT outcomes:
- Identify what went wrong in the reasoning or tool-use.
- Describe the CORRECT approach the model should have taken.
- The hint should provide actionable corrective guidance.

Your goal is to decide whether the next state reveals useful hindsight information
that could generate a hint to improve or reinforce the assistant response at turn t.

Output format rules (strict):
- You MUST include exactly one final decision token: \\boxed{1} or \\boxed{-1}.
- If and only if decision is \\boxed{1}, provide a concise, information-dense hint \
in 1-3 sentences, wrapped between [HINT_START] and [HINT_END].
- If decision is \\boxed{-1}, do not provide a hint block.
- Hint must be concrete and actionable."""

HINT_JUDGE_USER_TEMPLATE = """\
## Trajectory outcome: {outcome}

## Assistant response (turn t)
{response_text}

## Next state (turn t+1) [role: {next_state_role}]
{next_state_text}

Now output your decision and (if positive) the hint in the required format."""

HINT_RE = re.compile(r"\[HINT_START\](.*?)\[HINT_END\]", re.DOTALL)
DECISION_RE = re.compile(r"\\boxed\{([+-]?1)\}")


@dataclass
class HintResult:
    """Result of hint extraction for a single turn."""
    has_hint: bool
    hint_text: Optional[str]
    is_correct: bool  # whether the trajectory was correct
    turn_index: int


def parse_judge_output(text: str) -> tuple[int, Optional[str]]:
    """Parse judge output into (decision, hint_text).

    Returns:
        (1, hint_text) if hint extracted
        (-1, None) if no useful hint
        (0, None) if parsing failed
    """
    decision_match = DECISION_RE.search(text)
    if not decision_match:
        return 0, None

    decision = int(decision_match.group(1))
    if decision != 1:
        return -1, None

    hint_match = HINT_RE.search(text)
    if not hint_match:
        return -1, None

    hint_text = hint_match.group(1).strip()
    if len(hint_text) < 10:
        return -1, None

    return 1, hint_text


def select_best_hint(hints: list[tuple[int, Optional[str]]]) -> Optional[str]:
    """Select the best hint from multiple judge votes (majority voting).

    Among all valid hints (score==1, len>10), select the longest one
    (most information-dense), following OpenClaw's approach.
    """
    valid_hints = [h for score, h in hints if score == 1 and h and len(h) > 10]
    if not valid_hints:
        return None
    return max(valid_hints, key=len)


def build_hint_messages(
    response_text: str,
    next_state_text: str,
    next_state_role: str,
    is_correct: bool,
) -> list[dict[str, str]]:
    """Build chat messages for the hint judge.

    Args:
        response_text: The assistant's response at turn t
        next_state_text: The next state content (user reply or tool output)
        next_state_role: 'user' or 'tool'
        is_correct: Whether the trajectory led to correct answer
    """
    outcome = "CORRECT" if is_correct else "INCORRECT"
    return [
        {"role": "system", "content": HINT_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": HINT_JUDGE_USER_TEMPLATE.format(
                outcome=outcome,
                response_text=response_text[:4000],  # truncate long responses
                next_state_role=next_state_role,
                next_state_text=next_state_text[:2000],  # truncate long tool outputs
            ),
        },
    ]


def inject_hint_into_prompt(
    messages: list[dict[str, str]],
    hint: str,
) -> list[dict[str, str]]:
    """Inject hint into the last user message to create enhanced teacher context.

    This creates the "enhanced prompt" for computing teacher log-probs:
    A_t = log π_teacher(a_t | s + hint) - log π_θ(a_t | s)

    Args:
        messages: Original conversation messages
        hint: The extracted hint to inject

    Returns:
        New list of messages with hint appended to last user message
    """
    enhanced = [m.copy() for m in messages]
    # Find last user message
    for i in range(len(enhanced) - 1, -1, -1):
        if enhanced[i]["role"] == "user":
            enhanced[i]["content"] += f"\n\n[hint]\n{hint}"
            break
    return enhanced
