"""Test script for the Visual Diagnosis domain.

Tests:
1. DB loading and schema validation
2. Tool execution (all tools)
3. Environment setup
4. Gym interface
5. Task loading
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bioagents.domains.visual_diagnosis.data_model import (
    VisualDiagnosisDB, DB_PATH, POLICY_PATH, TASKS_PATH,
)
from bioagents.domains.visual_diagnosis.tools import VisualDiagnosisTools
from bioagents.domains.visual_diagnosis.environment import get_environment, get_tasks


def test_db_loading():
    """Test database loading and validation."""
    print("\n=== Test 1: DB Loading ===")
    db = VisualDiagnosisDB.load(DB_PATH)

    assert len(db.images) >= 3, f"Expected >=3 images, got {len(db.images)}"
    assert len(db.reports) >= 2, f"Expected >=2 reports, got {len(db.reports)}"
    assert len(db.questions) >= 3, f"Expected >=3 visual questions, got {len(db.questions)}"

    assert "IMG001" in db.images
    assert "RPT001" in db.reports
    assert "VQ001" in db.questions

    # Validate image fields
    img = db.images["IMG001"]
    assert img.modality == "xray"
    assert img.body_part == "chest"

    # Test hash
    h1 = db.get_hash()
    h2 = db.get_hash()
    assert h1 == h2

    # Test dump and reload
    test_path = "/tmp/test_visual_db.json"
    db.dump(test_path)
    db_reloaded = VisualDiagnosisDB.load(test_path)
    assert db.get_hash() == db_reloaded.get_hash()
    os.remove(test_path)

    print(f"  ✓ Loaded {len(db.images)} images")
    print(f"  ✓ {len(db.reports)} reports")
    print(f"  ✓ {len(db.questions)} visual questions")
    print(f"  ✓ DB hash: {h1}")
    print("  ✓ DB loading test PASSED")


def test_tools():
    """Test all visual diagnosis tools."""
    print("\n=== Test 2: Tool Execution ===")
    db = VisualDiagnosisDB.load(DB_PATH)
    tools = VisualDiagnosisTools(db)

    # 2a. get_image_report
    print("  Testing get_image_report...")
    report = tools.get_image_report(image_id="IMG001")
    assert isinstance(report, (dict, str))
    print(f"    ✓ Report for IMG001 retrieved")

    # 2b. analyze_medical_image
    print("  Testing analyze_medical_image...")
    analysis = tools.analyze_medical_image(image_id="IMG001")
    assert isinstance(analysis, (dict, str))
    print(f"    ✓ Image analysis completed")

    # 2c. search_similar_cases
    print("  Testing search_similar_cases...")
    similar = tools.search_similar_cases(image_id="IMG001")
    assert isinstance(similar, (list, dict, str))
    print(f"    ✓ Similar cases found")

    # 2d. search_imaging_knowledge
    print("  Testing search_imaging_knowledge...")
    knowledge = tools.search_imaging_knowledge(query="pneumonia chest xray")
    assert isinstance(knowledge, (list, dict, str))
    print(f"    ✓ Knowledge search completed")

    # 2e. think
    print("  Testing think...")
    result = tools.think(thought="Analyzing the chest X-ray findings...")
    assert result == ""
    print(f"    ✓ Think tool works")

    # 2f. submit_answer
    print("  Testing submit_answer...")
    answer = tools.submit_answer(
        answer="Right lower lobe consolidation consistent with pneumonia",
        reasoning="Based on the report findings"
    )
    assert isinstance(answer, str)
    print(f"    ✓ Answer submitted")

    # 2g. assertion helpers
    print("  Testing assert_correct_answer...")
    q = db.questions["VQ001"]
    is_correct = tools.assert_correct_answer("VQ001", q.answer)
    assert is_correct, f"Expected correct, answer='{q.answer}'"
    assert not tools.assert_correct_answer("VQ001", "Wrong answer")
    print(f"    ✓ Assertion helpers verified")

    # Tool statistics
    stats = tools.get_statistics()
    print(f"\n  Tool Statistics: {stats['num_tools']} total")
    print(f"    Tools: {stats['tool_names']}")
    print("  ✓ All tool tests PASSED")


def test_environment():
    """Test environment setup and interaction."""
    print("\n=== Test 3: Environment ===")

    env = get_environment()
    assert env.domain_name == "visual_diagnosis"
    assert env.tools is not None
    assert len(env.policy) > 0

    obs, info = env.reset()
    print(f"  ✓ Environment created: {env.domain_name}")
    print(f"  ✓ Policy length: {len(env.policy)} chars")
    print(f"  ✓ Tools available: {len(info['tools'])}")

    # Test step
    action = json.dumps({"name": "get_image_report", "arguments": {"image_id": "IMG001"}})
    obs, reward, terminated, truncated, info = env.step(action)
    assert len(obs) > 0
    print(f"  ✓ Environment step works. Turn: {info['turn_count']}")

    print("  ✓ Environment tests PASSED")


def test_tasks():
    """Test task loading."""
    print("\n=== Test 4: Tasks ===")

    tasks = get_tasks()
    assert len(tasks) >= 2, f"Expected >=2 tasks, got {len(tasks)}"
    print(f"  ✓ Loaded {len(tasks)} tasks")

    for task in tasks:
        tid = task["id"]
        desc = task.get("description", {})
        eval_criteria = task.get("evaluation_criteria", {})
        print(f"    - {tid}: difficulty={desc.get('difficulty', '?')}, "
              f"actions={len(eval_criteria.get('actions', []))}")

    print("  ✓ Task loading tests PASSED")


def test_gym_interface():
    """Test Gymnasium-compatible interface."""
    print("\n=== Test 5: GYM Interface ===")

    from bioagents.gym.agent_env import BioAgentGymEnv

    env = BioAgentGymEnv(
        domain="visual_diagnosis",
        task_id="vdx_chest_pneumonia_001",
        max_turns=10,
    )

    obs, info = env.reset()
    assert info["domain"] == "visual_diagnosis"
    assert len(info["tools"]) > 0
    print(f"  ✓ GYM env created. Task: {info['task_id']}")

    # Simulate visual diagnosis workflow
    actions = [
        json.dumps({"name": "get_image_report", "arguments": {"image_id": "IMG001"}}),
        json.dumps({"name": "analyze_medical_image", "arguments": {"image_id": "IMG001"}}),
        json.dumps({"name": "think", "arguments": {"thought": "The X-ray shows consolidation"}}),
        json.dumps({"name": "submit_answer", "arguments": {
            "answer": "Right lower lobe consolidation consistent with pneumonia",
            "reasoning": "Report findings"
        }}),
    ]

    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        tool_name = json.loads(action)["name"]
        print(f"  Step {i+1}: {tool_name} → reward={reward:.2f}")

    trajectory = env.get_trajectory()
    print(f"\n  ✓ Trajectory: {trajectory['total_turns']} turns")
    print(f"  ✓ Final reward: {trajectory['final_reward']:.3f}")
    print("  ✓ GYM interface tests PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("BIOAgents Visual Diagnosis Domain - Test Suite")
    print("=" * 60)

    test_db_loading()
    test_tools()
    test_environment()
    test_tasks()
    test_gym_interface()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
