"""SFT (Supervised Fine-Tuning) data generator for BIOAgents.

Converts agent trajectories and medical QA data into SFT-ready training format.

Supports:
1. Trajectory-based SFT: Convert successful agent runs into training data
2. Direct QA SFT: Convert medical questions into tool-use instruction format
3. Instruction-tuning SFT: Convert medical instruction data into chat format

Output format: JSONL with "messages" key (OpenAI chat format)
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

from loguru import logger


def trajectory_to_sft(
    trajectory_path: str,
    min_reward: float = 0.5,
    domain: str = "clinical_diagnosis",
) -> list[dict]:
    """Convert a logged trajectory into SFT training examples.

    Filters trajectories by minimum reward threshold and converts
    the successful tool-use sequences into chat-format training data.

    Args:
        trajectory_path: Path to the trajectory JSON file (from AgentRunner)
        min_reward: Minimum reward threshold for including the trajectory
        domain: Domain name for system prompt context

    Returns:
        List of SFT examples in OpenAI chat format
    """
    with open(trajectory_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check reward threshold
    final_reward = data.get("final_reward", 0.0)
    action_score = data.get("action_score", 0.0)
    effective_reward = max(final_reward, action_score)

    if effective_reward < min_reward:
        logger.debug(
            f"Skipping trajectory (reward={effective_reward:.3f} < {min_reward}): "
            f"{trajectory_path}"
        )
        return []

    turns = data.get("turns", [])
    if not turns:
        return []

    # Build messages from turns
    messages = []

    # System prompt (simplified for SFT)
    system_content = _get_system_prompt_for_sft(domain)
    messages.append({"role": "system", "content": system_content})

    # Initial observation from the first turn's prompt context
    # (In the trajectory, turns have raw_output, parsed_tool_call, tool_response)

    for turn in turns:
        raw_output = turn.get("raw_output", "")
        parsed_tool_call = turn.get("parsed_tool_call")
        tool_response = turn.get("tool_response", "")
        is_final = turn.get("is_final_answer", False)

        if parsed_tool_call:
            # Assistant makes a tool call
            messages.append({
                "role": "assistant",
                "content": json.dumps(parsed_tool_call, ensure_ascii=False),
            })
            # Tool response
            if tool_response:
                tool_name = parsed_tool_call.get("name", "tool")
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_name}:\n{tool_response[:2000]}",
                })
        elif is_final and raw_output:
            # Final answer
            messages.append({"role": "assistant", "content": raw_output})

    if len(messages) <= 1:
        return []

    return [{"messages": messages, "metadata": {
        "source": "trajectory",
        "domain": domain,
        "task_id": data.get("task_id", ""),
        "reward": effective_reward,
    }}]


def qa_tasks_to_sft(
    tasks: list[dict],
    include_reasoning: bool = True,
    domain: str = "medical_qa",
) -> list[dict]:
    """Convert medical QA tasks into SFT format with ideal tool-use sequences.

    Generates training examples that demonstrate the ideal pattern:
    1. Read question → 2. Search for evidence → 3. Think/reason → 4. Submit answer

    Args:
        tasks: List of task dicts (from medqa_loader or tasks.json)
        include_reasoning: Whether to include reasoning steps
        domain: Domain name

    Returns:
        List of SFT examples in chat format
    """
    examples = []
    system_prompt = _get_system_prompt_for_sft(domain)

    for task in tasks:
        correct_answer = task.get("correct_answer", "")
        if not correct_answer:
            continue

        ticket = task.get("ticket", "")
        question = task.get("raw_question", ticket)
        options = task.get("options", [])
        answer_text = task.get("raw_answer", "")

        messages = [{"role": "system", "content": system_prompt}]

        # User presents the question
        messages.append({"role": "user", "content": ticket})

        # Ideal sequence: think → search → submit
        if include_reasoning:
            # Step 1: Think about the question
            think_content = _generate_think_step(question, options, correct_answer, answer_text)
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "name": "think",
                    "arguments": {"thought": think_content}
                }),
            })
            messages.append({
                "role": "user",
                "content": "Tool result for think:\n",
            })

        # Step 2: Search for evidence
        search_query = _generate_search_query(question)
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "name": "retrieve_evidence",
                "arguments": {"query": search_query},
            }),
        })
        messages.append({
            "role": "user",
            "content": "Tool result for retrieve_evidence:\n[Evidence retrieved successfully]",
        })

        # Step 3: Submit answer with reasoning
        reasoning = _generate_reasoning(question, correct_answer, answer_text, options)
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "name": "submit_answer",
                "arguments": {
                    "answer": correct_answer,
                    "reasoning": reasoning,
                },
            }),
        })

        examples.append({
            "messages": messages,
            "metadata": {
                "source": task.get("description", {}).get("source", "unknown"),
                "domain": domain,
                "task_id": task.get("id", ""),
                "correct_answer": correct_answer,
            },
        })

    logger.info(f"Generated {len(examples)} SFT examples from QA tasks")
    return examples


def instruction_to_sft(
    instructions: list[dict],
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Convert medical instruction data into SFT chat format.

    Args:
        instructions: List of dicts with 'instruction', 'input', 'output' keys
        max_samples: Maximum number of samples

    Returns:
        List of SFT examples in chat format
    """
    examples = []
    system_prompt = (
        "You are a knowledgeable medical AI assistant. "
        "Provide accurate, evidence-based answers to medical questions."
    )

    for inst in instructions[:max_samples]:
        instruction_text = inst.get("instruction", "")
        input_text = inst.get("input", "")
        output_text = inst.get("output", "")

        if not instruction_text or not output_text:
            continue

        # Skip non-informative inputs
        if input_text and input_text.strip() not in ("", "<noinput>"):
            user_content = f"{instruction_text}\n\n{input_text}"
        else:
            user_content = instruction_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_text},
        ]

        examples.append({
            "messages": messages,
            "metadata": {"source": "instruction"},
        })

    logger.info(f"Generated {len(examples)} SFT examples from instructions")
    return examples


def save_sft_dataset(
    examples: list[dict],
    output_path: str,
    format: str = "jsonl",
):
    """Save SFT dataset to file.

    Args:
        examples: List of SFT examples
        output_path: Output file path
        format: 'jsonl' or 'json'
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    else:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(examples)} SFT examples to {output_path}")


# ---- Helper functions ----


def _get_system_prompt_for_sft(domain: str) -> str:
    """Get a concise system prompt for SFT training."""
    if domain == "medical_qa":
        return (
            "You are a medical AI assistant that answers medical questions using "
            "evidence-based reasoning. Use tools to search for evidence, then "
            "submit your answer with clear reasoning. Available tools: "
            "search_pubmed, search_medical_wiki, retrieve_evidence, "
            "browse_article, browse_wiki_entry, analyze_answer_options, "
            "think, submit_answer."
        )
    elif domain == "clinical_diagnosis":
        return (
            "You are a medical AI assistant for clinical diagnosis. Use tools "
            "to review patient records, order tests, check drug interactions, "
            "and make clinical recommendations. Available tools: "
            "get_patient_info, get_vital_signs, get_lab_results, order_lab_test, "
            "get_medications, check_drug_interaction, prescribe_medication, "
            "get_clinical_notes, add_clinical_note, get_differential_diagnosis, "
            "search_clinical_guidelines, record_diagnosis, search_medical_literature, "
            "transfer_to_specialist, think."
        )
    return "You are a medical AI assistant."


def _normalize_options(options) -> list[dict]:
    """Normalize options to list[dict] format with 'label' and 'text' keys.

    Handles:
    - list[dict] with 'label' and 'text' keys (already normalized)
    - dict like {"A": "text", "B": "text"}
    - list[str] like ["option1", "option2"]
    """
    if isinstance(options, dict):
        return [{"label": k, "text": str(v)} for k, v in options.items()]
    if isinstance(options, list):
        if options and isinstance(options[0], dict) and "label" in options[0]:
            return options
        if options and isinstance(options[0], str):
            labels = "ABCDEFGHIJ"
            return [{"label": labels[i], "text": o} for i, o in enumerate(options) if i < len(labels)]
    return []


def _generate_think_step(
    question: str, options, correct_answer: str, answer_text: str
) -> str:
    """Generate a reasoning thought for the think tool."""
    q_snippet = question[:200]
    norm_opts = _normalize_options(options)
    opts_str = ", ".join(
        f"{o['label']}: {o['text'][:50]}" for o in norm_opts
    )
    return (
        f"Let me analyze this question. The question asks about: {q_snippet}... "
        f"The options are: {opts_str}. "
        f"I need to consider each option carefully and find supporting evidence."
    )


def _generate_search_query(question: str) -> str:
    """Generate a search query from a question."""
    # Extract key medical terms (simple heuristic)
    q_lower = question.lower()
    # Remove common non-medical words
    stop_words = {
        "the", "a", "an", "is", "was", "were", "are", "of", "in", "to",
        "for", "with", "which", "following", "most", "likely", "due",
        "patient", "year", "old", "man", "woman", "comes", "physician",
        "because", "history", "shows", "laboratory", "studies", "show",
        "examination", "physical", "his", "her", "this", "that", "what",
    }
    words = question.split()[:30]
    medical_words = [w.strip(".,;:()") for w in words if w.lower().strip(".,;:()") not in stop_words and len(w) > 2]
    return " ".join(medical_words[:10])


def _generate_reasoning(
    question: str, correct_answer: str, answer_text: str, options
) -> str:
    """Generate reasoning text for the answer submission."""
    # Find the correct option text
    correct_text = answer_text
    norm_opts = _normalize_options(options)
    for opt in norm_opts:
        if opt["label"] == correct_answer:
            correct_text = opt["text"]
            break

    return (
        f"Based on the evidence gathered, the answer is {correct_answer}: "
        f"{correct_text}."
    )
