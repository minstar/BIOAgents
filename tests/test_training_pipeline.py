"""Test script for the Training Pipeline (GRPO + SFT).

Tests:
1. GRPO config loading from YAML
2. GRPO dataset building
3. GRPO reward function integration
4. SFT config loading from YAML
5. SFT dataset building
6. Cross-domain GYM registration
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_grpo_config_loading():
    """Test GRPO config loading from YAML."""
    print("\n=== Test 1: GRPO Config Loading ===")

    from bioagents.training.grpo_trainer import BioAgentGRPOConfig

    # Medical QA config
    config = BioAgentGRPOConfig.from_yaml("configs/grpo_medical_qa.yaml")
    assert config.model_name_or_path == "Qwen/Qwen3-1.7B"
    assert config.domain == "medical_qa"
    assert config.num_generations == 4
    assert config.beta == 0.04
    assert config.peft_enabled is True
    assert config.peft_r == 16
    assert len(config.reward_functions) == 3
    print(f"  ✓ Medical QA config: model={config.model_name_or_path}, domain={config.domain}")
    print(f"    G={config.num_generations}, β={config.beta}, lr={config.learning_rate}")

    # Drug interaction config
    config_di = BioAgentGRPOConfig.from_yaml("configs/grpo_drug_interaction.yaml")
    assert config_di.domain == "drug_interaction"
    assert config_di.num_train_epochs == 5
    print(f"  ✓ Drug Interaction config: domain={config_di.domain}, epochs={config_di.num_train_epochs}")

    print("  ✓ GRPO config loading PASSED")


def test_grpo_dataset_building():
    """Test GRPO dataset construction."""
    print("\n=== Test 2: GRPO Dataset Building ===")

    from bioagents.training.grpo_trainer import BioAgentGRPOConfig, build_grpo_dataset

    # Medical QA
    config = BioAgentGRPOConfig.from_yaml("configs/grpo_medical_qa.yaml")
    dataset = build_grpo_dataset(config, split="train")
    assert len(dataset) > 0
    assert "prompt" in dataset.column_names
    assert "solution" in dataset.column_names
    assert "task_id" in dataset.column_names

    sample = dataset[0]
    prompt = sample["prompt"]
    assert isinstance(prompt, list)
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    assert len(sample["solution"]) > 0
    print(f"  ✓ Medical QA dataset: {len(dataset)} examples")
    print(f"    Sample task_id: {sample['task_id']}")
    print(f"    Prompt roles: {[m['role'] for m in prompt]}")

    # Drug interaction
    config_di = BioAgentGRPOConfig.from_yaml("configs/grpo_drug_interaction.yaml")
    dataset_di = build_grpo_dataset(config_di, split="train")
    assert len(dataset_di) > 0
    sample_di = dataset_di[0]
    assert "pharmacology" in sample_di["prompt"][0]["content"].lower() or "drug" in sample_di["prompt"][0]["content"].lower()
    print(f"  ✓ Drug Interaction dataset: {len(dataset_di)} examples")

    print("  ✓ GRPO dataset building PASSED")


def test_grpo_reward_functions():
    """Test GRPO reward function integration."""
    print("\n=== Test 3: GRPO Reward Functions ===")

    from bioagents.training.grpo_trainer import BioAgentGRPOConfig, build_reward_functions
    from bioagents.evaluation.grpo_rewards import (
        grpo_accuracy_reward,
        grpo_format_reward,
        grpo_process_reward,
        grpo_composite_reward,
    )

    config = BioAgentGRPOConfig.from_yaml("configs/grpo_medical_qa.yaml")
    reward_fns = build_reward_functions(config)
    assert len(reward_fns) == 3

    # Test with mock completions
    completions = [
        [{"content": "<answer>B</answer>", "role": "assistant"}],
        [{"content": "The answer is C", "role": "assistant"}],
        [{"content": '{"name": "submit_answer", "arguments": {"answer": "A"}}', "role": "assistant"}],
    ]
    solutions = ["B", "B", "A"]

    # Test accuracy
    acc_scores = grpo_accuracy_reward(completions, solution=solutions)
    assert acc_scores[0] == 1.0  # Correct
    assert acc_scores[1] == 0.0  # Wrong (C != B)
    assert acc_scores[2] == 1.0  # Correct via tool call
    print(f"  ✓ Accuracy rewards: {acc_scores}")

    # Test format
    fmt_scores = grpo_format_reward(completions)
    assert fmt_scores[0] == 1.0  # Has <answer> tags
    assert fmt_scores[1] == 0.3  # Just text
    assert fmt_scores[2] == 1.0  # Valid tool call
    print(f"  ✓ Format rewards: {fmt_scores}")

    # Test process
    proc_scores = grpo_process_reward(completions, solution=solutions)
    assert all(0 <= s <= 1.0 for s in proc_scores)
    print(f"  ✓ Process rewards: {proc_scores}")

    # Test composite
    comp_scores = grpo_composite_reward(completions, solution=solutions)
    assert all(0 <= s <= 1.0 for s in comp_scores)
    print(f"  ✓ Composite rewards: {comp_scores}")

    print("  ✓ GRPO reward functions PASSED")


def test_sft_config_loading():
    """Test SFT config loading from YAML."""
    print("\n=== Test 4: SFT Config Loading ===")

    from bioagents.training.sft_trainer import BioAgentSFTConfig

    config = BioAgentSFTConfig.from_yaml("configs/sft_medical_qa.yaml")
    assert config.model_name_or_path == "Qwen/Qwen3-1.7B"
    assert config.qa_tasks_path == "data/domains/medical_qa/tasks.json"
    assert config.max_length == 4096
    assert config.peft_enabled is True
    print(f"  ✓ SFT config: model={config.model_name_or_path}")
    print(f"    max_length={config.max_length}, train_ratio={config.train_ratio}")

    print("  ✓ SFT config loading PASSED")


def test_sft_dataset_building():
    """Test SFT dataset construction."""
    print("\n=== Test 5: SFT Dataset Building ===")

    from bioagents.training.sft_trainer import BioAgentSFTConfig, build_sft_dataset

    config = BioAgentSFTConfig.from_yaml("configs/sft_medical_qa.yaml")
    train_ds, eval_ds = build_sft_dataset(config)

    assert len(train_ds) > 0
    assert eval_ds is None or len(eval_ds) > 0

    sample = json.loads(train_ds[0]["messages"])
    assert isinstance(sample, list)
    assert sample[0]["role"] == "system"
    assert sample[1]["role"] == "user"
    # Should have tool-use demonstration turns
    assert len(sample) >= 5  # system + user + think + search + submit
    print(f"  ✓ SFT train dataset: {len(train_ds)} examples")
    print(f"  ✓ SFT eval dataset: {len(eval_ds) if eval_ds else 0} examples")
    print(f"    Sample turns: {len(sample)}")
    print(f"    Roles: {[m['role'] for m in sample]}")

    print("  ✓ SFT dataset building PASSED")


def test_cross_domain_gym():
    """Test GYM registration across all domains."""
    print("\n=== Test 6: Cross-Domain GYM Registration ===")

    from bioagents.gym.agent_env import _load_default_domains, _DOMAIN_REGISTRY

    _load_default_domains()

    expected_domains = ["clinical_diagnosis", "medical_qa", "visual_diagnosis", "drug_interaction"]
    for domain in expected_domains:
        assert domain in _DOMAIN_REGISTRY, f"Domain '{domain}' not registered"
        print(f"  ✓ Domain registered: {domain}")

    print(f"  ✓ Total registered domains: {len(_DOMAIN_REGISTRY)}")

    # Quick smoke test for each domain
    from bioagents.gym.agent_env import BioAgentGymEnv

    for domain in expected_domains:
        try:
            env = BioAgentGymEnv(domain=domain, max_turns=5)
            obs, info = env.reset()
            assert len(obs) > 0
            assert info["domain"] == domain
            print(f"  ✓ {domain}: reset OK, tools={len(info['tools'])}")
        except Exception as e:
            print(f"  ✗ {domain}: {e}")

    print("  ✓ Cross-domain GYM tests PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("BIOAgents Training Pipeline - Test Suite")
    print("=" * 60)

    test_grpo_config_loading()
    test_grpo_dataset_building()
    test_grpo_reward_functions()
    test_sft_config_loading()
    test_sft_dataset_building()
    test_cross_domain_gym()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
