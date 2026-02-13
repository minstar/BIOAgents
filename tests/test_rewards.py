"""Tests for BIOAgents reward functions.

Tests both the core reward functions (bioagents.evaluation.rewards)
and the GRPO-compatible wrappers (bioagents.evaluation.grpo_rewards).
"""

import json
import pytest


# ============================================================
# Core Reward Function Tests
# ============================================================

class TestAccuracyRewardExactMatch:
    """Tests for accuracy_reward_exact_match."""

    def test_correct_mc_answer(self):
        from bioagents.evaluation.rewards import accuracy_reward_exact_match
        assert accuracy_reward_exact_match("A", "A") == 1.0

    def test_correct_mc_answer_case_insensitive(self):
        from bioagents.evaluation.rewards import accuracy_reward_exact_match
        assert accuracy_reward_exact_match("a", "A") == 1.0

    def test_wrong_mc_answer(self):
        from bioagents.evaluation.rewards import accuracy_reward_exact_match
        assert accuracy_reward_exact_match("B", "A") == 0.0

    def test_answer_from_submit_tool(self):
        from bioagents.evaluation.rewards import accuracy_reward_exact_match
        tool_call = json.dumps({"name": "submit_answer", "arguments": {"answer": "C"}})
        assert accuracy_reward_exact_match(tool_call, "C") == 1.0

    def test_answer_from_text_with_prefix(self):
        from bioagents.evaluation.rewards import accuracy_reward_exact_match
        response = "Based on the evidence, the correct answer is B"
        assert accuracy_reward_exact_match(response, "B") == 1.0

    def test_no_answer_found(self):
        from bioagents.evaluation.rewards import accuracy_reward_exact_match
        response = "I need more information to answer this question."
        assert accuracy_reward_exact_match(response, "A") == 0.0


class TestAccuracyRewardSoft:
    """Tests for accuracy_reward_soft."""

    def test_exact_match_long(self):
        from bioagents.evaluation.rewards import accuracy_reward_soft
        answer = "Metformin reduces hepatic glucose production"
        score = accuracy_reward_soft(answer, answer)
        assert score == 1.0

    def test_partial_overlap(self):
        from bioagents.evaluation.rewards import accuracy_reward_soft
        response = "Metformin is a drug that reduces glucose production in the liver"
        reference = "Metformin reduces hepatic glucose production through AMPK activation"
        score = accuracy_reward_soft(response, reference)
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        from bioagents.evaluation.rewards import accuracy_reward_soft
        response = "The sky is blue"
        reference = "Metformin reduces hepatic glucose production"
        score = accuracy_reward_soft(response, reference)
        assert score == 0.0

    def test_empty_response(self):
        from bioagents.evaluation.rewards import accuracy_reward_soft
        assert accuracy_reward_soft("", "test answer") == 0.0

    def test_mc_falls_back_to_exact(self):
        from bioagents.evaluation.rewards import accuracy_reward_soft
        # Single-letter answers should use exact match
        assert accuracy_reward_soft("A", "A") == 1.0
        assert accuracy_reward_soft("B", "A") == 0.0


class TestFormatRewardToolCall:
    """Tests for format_reward_tool_call."""

    def test_valid_json_tool_call(self):
        from bioagents.evaluation.rewards import format_reward_tool_call
        tool = json.dumps({"name": "search_pubmed", "arguments": {"query": "diabetes"}})
        assert format_reward_tool_call(tool) == 1.0

    def test_json_missing_arguments(self):
        from bioagents.evaluation.rewards import format_reward_tool_call
        tool = json.dumps({"name": "search_pubmed"})
        assert format_reward_tool_call(tool) == 0.5

    def test_json_in_code_block(self):
        from bioagents.evaluation.rewards import format_reward_tool_call
        response = '```json\n{"name": "search_pubmed", "arguments": {"query": "test"}}\n```'
        assert format_reward_tool_call(response) == 0.8

    def test_invalid_format(self):
        from bioagents.evaluation.rewards import format_reward_tool_call
        assert format_reward_tool_call("This is just text") == 0.0

    def test_answer_letter_format(self):
        from bioagents.evaluation.rewards import format_reward_tool_call
        assert format_reward_tool_call("A", expected_format="answer_letter") == 1.0
        assert format_reward_tool_call("Based on analysis, the answer is B", expected_format="answer_letter") == 1.0
        assert format_reward_tool_call("Random text", expected_format="answer_letter") == 0.0


class TestFormatRewardThinkAnswer:
    """Tests for format_reward_think_answer."""

    def test_both_tags(self):
        from bioagents.evaluation.rewards import format_reward_think_answer
        response = "<think>Considering the evidence...</think> <answer>Metformin</answer>"
        assert format_reward_think_answer(response) == 1.0

    def test_only_think(self):
        from bioagents.evaluation.rewards import format_reward_think_answer
        response = "<think>Let me analyze this</think> The answer is B."
        assert format_reward_think_answer(response) == 0.5

    def test_neither(self):
        from bioagents.evaluation.rewards import format_reward_think_answer
        response = "The answer is A."
        assert format_reward_think_answer(response) == 0.0


class TestFormatRewardComposite:
    """Tests for format_reward_composite."""

    def test_final_answer_long(self):
        from bioagents.evaluation.rewards import format_reward_composite
        response = "Based on the clinical evidence, the patient most likely has pneumonia. " * 3
        assert format_reward_composite(response, is_final=True) == 1.0

    def test_final_answer_short(self):
        from bioagents.evaluation.rewards import format_reward_composite
        assert format_reward_composite("A", is_final=True) == 0.3

    def test_intermediate_tool_call(self):
        from bioagents.evaluation.rewards import format_reward_composite
        tool = json.dumps({"name": "search_pubmed", "arguments": {"query": "test"}})
        assert format_reward_composite(tool, is_final=False) == 1.0

    def test_intermediate_no_tool(self):
        from bioagents.evaluation.rewards import format_reward_composite
        assert format_reward_composite("Just some text", is_final=False) == 0.0


class TestProcessRewardToolUsage:
    """Tests for process_reward_tool_usage."""

    def test_all_expected_tools_called(self):
        from bioagents.evaluation.rewards import process_reward_tool_usage
        expected = [
            {"name": "search_pubmed", "arguments": {}, "compare_args": []},
            {"name": "browse_article", "arguments": {}, "compare_args": []},
        ]
        actual = [
            {"tool_name": "search_pubmed", "arguments": {}},
            {"tool_name": "browse_article", "arguments": {}},
        ]
        score = process_reward_tool_usage(actual, expected)
        # 60% coverage (1.0) + 20% diversity (2/2=1.0) + 20% thoroughness (2/5=0.4)
        assert score == pytest.approx(0.6 * 1.0 + 0.2 * 1.0 + 0.2 * 0.4, abs=0.01)

    def test_missing_expected_tool(self):
        from bioagents.evaluation.rewards import process_reward_tool_usage
        expected = [
            {"name": "search_pubmed", "arguments": {}, "compare_args": []},
            {"name": "browse_article", "arguments": {}, "compare_args": []},
        ]
        actual = [
            {"tool_name": "search_pubmed", "arguments": {}},
        ]
        score = process_reward_tool_usage(actual, expected)
        # coverage: 1/2=0.5, diversity: 1/1=1.0, thoroughness: 1/5=0.2
        assert score == pytest.approx(0.6 * 0.5 + 0.2 * 1.0 + 0.2 * 0.2, abs=0.01)

    def test_excessive_repetition(self):
        from bioagents.evaluation.rewards import process_reward_tool_usage
        expected = [{"name": "search_pubmed", "arguments": {}, "compare_args": []}]
        actual = [
            {"tool_name": "search_pubmed", "arguments": {}},
            {"tool_name": "search_pubmed", "arguments": {}},
            {"tool_name": "search_pubmed", "arguments": {}},
        ]
        score = process_reward_tool_usage(actual, expected)
        # coverage: 1/1=1.0, diversity: 1/3=0.333 (same sig repeated),
        # thoroughness: 1/5=0.2
        assert score == pytest.approx(0.6 * 1.0 + 0.2 * (1 / 3) + 0.2 * 0.2, abs=0.01)

    def test_same_tool_different_args_not_penalised(self):
        """Calling the same tool with different arguments should NOT be penalised."""
        from bioagents.evaluation.rewards import process_reward_tool_usage
        expected = [{"name": "get_lab_results", "arguments": {}, "compare_args": []}]
        actual = [
            {"tool_name": "get_lab_results", "arguments": {"lab_name": "creatinine"}},
            {"tool_name": "get_lab_results", "arguments": {"lab_name": "WBC"}},
            {"tool_name": "get_lab_results", "arguments": {"lab_name": "BNP"}},
        ]
        score = process_reward_tool_usage(actual, expected)
        # coverage: 1/1=1.0, diversity: 3/3=1.0 (all unique sigs),
        # thoroughness: 1/5=0.2
        assert score == pytest.approx(0.6 * 1.0 + 0.2 * 1.0 + 0.2 * 0.2, abs=0.01)

    def test_no_expected_actions(self):
        from bioagents.evaluation.rewards import process_reward_tool_usage
        # Empty expected + empty actual → default 0.5
        assert process_reward_tool_usage([], []) == 0.5

    def test_arg_matching(self):
        from bioagents.evaluation.rewards import process_reward_tool_usage
        expected = [
            {
                "name": "search_pubmed",
                "arguments": {"query": "diabetes"},
                "compare_args": ["query"],
            }
        ]
        actual_correct = [
            {"tool_name": "search_pubmed", "arguments": {"query": "diabetes"}},
        ]
        actual_wrong = [
            {"tool_name": "search_pubmed", "arguments": {"query": "heart disease"}},
        ]
        assert process_reward_tool_usage(actual_correct, expected) > process_reward_tool_usage(actual_wrong, expected)


class TestProcessRewardReasoningQuality:
    """Tests for process_reward_reasoning_quality."""

    def test_good_reasoning(self):
        from bioagents.evaluation.rewards import process_reward_reasoning_quality
        response = (
            "Based on the clinical findings, the patient presents with symptoms "
            "consistent with pneumonia. First, the chest X-ray shows bilateral infiltrates. "
            "The evidence suggests a bacterial infection. Therefore, the most likely "
            "diagnosis is community-acquired pneumonia, and treatment with antibiotics "
            "is recommended."
        )
        score = process_reward_reasoning_quality(response, "pneumonia")
        assert score > 0.5

    def test_empty_reasoning(self):
        from bioagents.evaluation.rewards import process_reward_reasoning_quality
        assert process_reward_reasoning_quality("", "test") == 0.0

    def test_short_response(self):
        from bioagents.evaluation.rewards import process_reward_reasoning_quality
        score = process_reward_reasoning_quality("A", "A")
        assert score < 0.5


class TestCompositeReward:
    """Tests for compute_composite_reward."""

    def test_all_components(self):
        from bioagents.evaluation.rewards import compute_composite_reward
        result = compute_composite_reward(
            response="The treatment for diabetes involves metformin.",
            correct_answer="metformin",
            tool_call_log=[
                {"tool_name": "search_pubmed", "arguments": {"query": "diabetes treatment"}},
            ],
            expected_actions=[
                {"name": "search_pubmed", "arguments": {}, "compare_args": []},
            ],
            is_final=True,
        )
        assert "total" in result
        assert "accuracy" in result
        assert "format" in result
        assert "process" in result
        assert "weights" in result
        assert 0.0 <= result["total"] <= 1.0

    def test_custom_weights(self):
        from bioagents.evaluation.rewards import compute_composite_reward
        result = compute_composite_reward(
            response="A",
            correct_answer="A",
            weights={"accuracy": 1.0, "format": 0.0, "process": 0.0},
        )
        assert result["total"] == pytest.approx(result["accuracy"])

    def test_zero_score(self):
        from bioagents.evaluation.rewards import compute_composite_reward
        result = compute_composite_reward(
            response="",
            correct_answer="B",
        )
        # Empty response: accuracy=0, format varies, process may be non-zero due to efficiency
        assert result["accuracy"] == 0.0
        assert result["total"] < 0.5  # Should be very low


class TestRewardRegistry:
    """Tests for the reward function registry."""

    def test_get_known_function(self):
        from bioagents.evaluation.rewards import get_reward_function
        fn = get_reward_function("accuracy_exact")
        assert callable(fn)

    def test_get_unknown_function(self):
        from bioagents.evaluation.rewards import get_reward_function
        with pytest.raises(ValueError, match="Unknown reward function"):
            get_reward_function("nonexistent")

    def test_register_custom(self):
        from bioagents.evaluation.rewards import register_reward_function, get_reward_function
        register_reward_function("custom_test", lambda **kw: 0.42)
        fn = get_reward_function("custom_test")
        assert fn() == 0.42


# ============================================================
# GRPO Reward Wrapper Tests
# ============================================================

class TestGRPOAccuracyReward:
    """Tests for grpo_accuracy_reward."""

    def test_mc_correct(self):
        from bioagents.evaluation.grpo_rewards import grpo_accuracy_reward
        completions = [[{"content": "A", "role": "assistant"}]]
        scores = grpo_accuracy_reward(completions, solution=["A"])
        assert scores == [1.0]

    def test_mc_wrong(self):
        from bioagents.evaluation.grpo_rewards import grpo_accuracy_reward
        completions = [[{"content": "B", "role": "assistant"}]]
        scores = grpo_accuracy_reward(completions, solution=["A"])
        assert scores == [0.0]

    def test_mc_from_answer_tag(self):
        from bioagents.evaluation.grpo_rewards import grpo_accuracy_reward
        completions = [[{"content": "<think>reasoning</think> <answer>C</answer>", "role": "assistant"}]]
        scores = grpo_accuracy_reward(completions, solution=["C"])
        assert scores == [1.0]

    def test_open_ended(self):
        from bioagents.evaluation.grpo_rewards import grpo_accuracy_reward
        completions = [
            [{"content": "Metformin reduces hepatic glucose production through AMPK activation"}],
        ]
        scores = grpo_accuracy_reward(
            completions,
            solution=["Metformin reduces hepatic glucose production through AMPK activation"],
        )
        # Should get high score (near 1.0) for exact match
        assert scores[0] > 0.5

    def test_batch(self):
        from bioagents.evaluation.grpo_rewards import grpo_accuracy_reward
        completions = [
            [{"content": "A"}],
            [{"content": "C"}],
            [{"content": "D"}],
        ]
        scores = grpo_accuracy_reward(completions, solution=["A", "B", "D"])
        assert scores[0] == 1.0
        assert scores[1] == 0.0
        assert scores[2] == 1.0


class TestGRPOFormatReward:
    """Tests for grpo_format_reward."""

    def test_valid_tool_call(self):
        from bioagents.evaluation.grpo_rewards import grpo_format_reward
        tool = json.dumps({"name": "search_pubmed", "arguments": {"query": "test"}})
        completions = [[{"content": tool}]]
        scores = grpo_format_reward(completions)
        assert scores == [1.0]

    def test_answer_tags(self):
        from bioagents.evaluation.grpo_rewards import grpo_format_reward
        completions = [[{"content": "<answer>A</answer>"}]]
        scores = grpo_format_reward(completions)
        assert scores == [1.0]

    def test_plain_text(self):
        from bioagents.evaluation.grpo_rewards import grpo_format_reward
        completions = [[{"content": "This is a long enough text response for partial credit testing purposes."}]]
        scores = grpo_format_reward(completions)
        assert scores == [0.3]

    def test_empty(self):
        from bioagents.evaluation.grpo_rewards import grpo_format_reward
        completions = [[{"content": ""}]]
        scores = grpo_format_reward(completions)
        assert scores == [0.0]


class TestGRPOProcessReward:
    """Tests for grpo_process_reward."""

    def test_good_reasoning(self):
        from bioagents.evaluation.grpo_rewards import grpo_process_reward
        completions = [[{
            "content": (
                "Based on the clinical evidence, the patient presents with symptoms "
                "consistent with pneumonia. First, the chest X-ray shows bilateral infiltrates. "
                "Therefore, the diagnosis is community-acquired pneumonia."
            )
        }]]
        scores = grpo_process_reward(completions, solution=["pneumonia"])
        assert scores[0] > 0.3

    def test_empty_reasoning(self):
        from bioagents.evaluation.grpo_rewards import grpo_process_reward
        completions = [[{"content": ""}]]
        scores = grpo_process_reward(completions, solution=["test"])
        assert scores == [0.0]


class TestGRPOCompositeReward:
    """Tests for grpo_composite_reward."""

    def test_combined_scores(self):
        from bioagents.evaluation.grpo_rewards import grpo_composite_reward
        tool = json.dumps({"name": "submit_answer", "arguments": {"answer": "A"}})
        completions = [[{"content": tool}]]
        scores = grpo_composite_reward(completions, solution=["A"])
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_batch_combined(self):
        from bioagents.evaluation.grpo_rewards import grpo_composite_reward
        completions = [
            [{"content": "A"}],
            [{"content": "<answer>B</answer>"}],
        ]
        scores = grpo_composite_reward(completions, solution=["A", "B"])
        assert len(scores) == 2
        # Both should get non-zero scores
        assert scores[0] > 0.0
        assert scores[1] > 0.0


class TestGRPORewardRegistry:
    """Tests for GRPO reward registry."""

    def test_get_known_functions(self):
        from bioagents.evaluation.grpo_rewards import get_grpo_reward_functions
        funcs = get_grpo_reward_functions(["accuracy", "format", "process"])
        assert len(funcs) == 3
        assert all(callable(f) for f in funcs)

    def test_unknown_function(self):
        from bioagents.evaluation.grpo_rewards import get_grpo_reward_functions
        with pytest.raises(ValueError, match="Unknown GRPO reward"):
            get_grpo_reward_functions(["nonexistent"])


# ============================================================
# Integration Tests
# ============================================================

class TestRewardIntegration:
    """Integration tests combining reward functions with environment outputs."""

    def test_clinical_diagnosis_trajectory(self):
        """Test reward computation on a simulated clinical diagnosis trajectory."""
        from bioagents.evaluation.rewards import compute_composite_reward
        
        response = (
            "Based on my assessment, the patient presents with community-acquired pneumonia. "
            "The findings include fever, productive cough, and bilateral infiltrates on imaging. "
            "I recommend starting empiric antibiotics with amoxicillin-clavulanate."
        )
        
        tool_log = [
            {"tool_name": "get_patient_info", "arguments": {"patient_id": "P001"}},
            {"tool_name": "get_vital_signs", "arguments": {"patient_id": "P001"}},
            {"tool_name": "order_lab_test", "arguments": {"patient_id": "P001", "test_name": "CBC"}},
            {"tool_name": "record_diagnosis", "arguments": {"patient_id": "P001", "diagnosis": "pneumonia"}},
        ]
        
        expected = [
            {"name": "get_patient_info", "arguments": {"patient_id": "P001"}, "compare_args": ["patient_id"]},
            {"name": "get_vital_signs", "arguments": {"patient_id": "P001"}, "compare_args": ["patient_id"]},
            {"name": "order_lab_test", "arguments": {"patient_id": "P001", "test_name": "CBC"}, "compare_args": ["patient_id"]},
            {"name": "record_diagnosis", "arguments": {"patient_id": "P001", "diagnosis": "pneumonia"}, "compare_args": ["patient_id", "diagnosis"]},
        ]
        
        result = compute_composite_reward(
            response=response,
            correct_answer="community-acquired pneumonia",
            tool_call_log=tool_log,
            expected_actions=expected,
            is_final=True,
        )
        
        assert result["total"] > 0.4
        assert result["tool_usage"] > 0.9  # All expected tools called (coverage 1.0)
        assert result["format"] == 1.0  # Long final answer

    def test_medical_qa_mc_trajectory(self):
        """Test reward computation on a medical QA multiple-choice task."""
        from bioagents.evaluation.grpo_rewards import grpo_accuracy_reward, grpo_format_reward
        
        # Simulate correct submission
        completions = [
            [{"content": json.dumps({"name": "submit_answer", "arguments": {"answer": "C"}})}],
        ]
        
        acc_scores = grpo_accuracy_reward(completions, solution=["C"])
        fmt_scores = grpo_format_reward(completions)
        
        assert acc_scores == [1.0]
        assert fmt_scores == [1.0]

    def test_medical_qa_wrong_answer(self):
        """Test reward computation on wrong MC answer."""
        from bioagents.evaluation.grpo_rewards import grpo_accuracy_reward
        
        completions = [
            [{"content": json.dumps({"name": "submit_answer", "arguments": {"answer": "A"}})}],
        ]
        
        scores = grpo_accuracy_reward(completions, solution=["C"])
        assert scores == [0.0]


# ============================================================
# Helper Function Tests
# ============================================================

class TestExtractAnswer:
    """Tests for _extract_answer_from_response."""

    def test_single_letter(self):
        from bioagents.evaluation.rewards import _extract_answer_from_response
        assert _extract_answer_from_response("A") == "A"

    def test_answer_prefix(self):
        from bioagents.evaluation.rewards import _extract_answer_from_response
        assert _extract_answer_from_response("Answer: B") == "B"

    def test_option_format(self):
        from bioagents.evaluation.rewards import _extract_answer_from_response
        assert _extract_answer_from_response("Option C") == "C"

    def test_embedded_in_text(self):
        from bioagents.evaluation.rewards import _extract_answer_from_response
        result = _extract_answer_from_response(
            "After careful analysis, the correct answer is D because of..."
        )
        assert result == "D"

    def test_submit_answer_tool(self):
        from bioagents.evaluation.rewards import _extract_answer_from_response
        tool_json = json.dumps({"name": "submit_answer", "arguments": {"answer": "E"}})
        assert _extract_answer_from_response(tool_json) == "E"

    def test_no_answer_found(self):
        from bioagents.evaluation.rewards import _extract_answer_from_response
        result = _extract_answer_from_response("I need more information.")
        assert result == ""
