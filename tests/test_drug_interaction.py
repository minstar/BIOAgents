"""Test script for the Drug Interaction domain.

Tests:
1. DB loading and schema validation
2. Tool execution (all tools)
3. Environment setup
4. Gym interface
5. Task loading and basic reward computation
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bioagents.domains.drug_interaction.data_model import (
    DrugInteractionDB, DB_PATH, POLICY_PATH, TASKS_PATH,
)
from bioagents.domains.drug_interaction.tools import DrugInteractionTools
from bioagents.domains.drug_interaction.environment import get_environment, get_tasks


def test_db_loading():
    """Test database loading and validation."""
    print("\n=== Test 1: DB Loading ===")
    db = DrugInteractionDB.load(DB_PATH)

    assert len(db.drugs) == 12, f"Expected 12 drugs, got {len(db.drugs)}"
    assert len(db.interactions) == 10, f"Expected 10 interactions, got {len(db.interactions)}"
    assert len(db.patient_profiles) == 4, f"Expected 4 patient profiles, got {len(db.patient_profiles)}"
    assert len(db.alternatives) == 4, f"Expected 4 alternative groups, got {len(db.alternatives)}"

    assert "warfarin" in db.drugs
    assert "fluoxetine" in db.drugs
    assert "DI_P001" in db.patient_profiles

    # Test hash
    h1 = db.get_hash()
    h2 = db.get_hash()
    assert h1 == h2, "Hash should be deterministic"

    # Test dump and reload
    test_path = "/tmp/test_drug_interaction_db.json"
    db.dump(test_path)
    db_reloaded = DrugInteractionDB.load(test_path)
    assert db.get_hash() == db_reloaded.get_hash()
    os.remove(test_path)

    print(f"  ✓ Loaded {len(db.drugs)} drugs")
    print(f"  ✓ {len(db.interactions)} interactions")
    print(f"  ✓ {len(db.patient_profiles)} patient profiles")
    print(f"  ✓ {len(db.alternatives)} alternative groups")
    print(f"  ✓ DB hash: {h1}")
    print("  ✓ DB loading test PASSED")


def test_tools():
    """Test all drug interaction tools."""
    print("\n=== Test 2: Tool Execution ===")
    db = DrugInteractionDB.load(DB_PATH)
    tools = DrugInteractionTools(db)

    # 2a. get_drug_info
    print("  Testing get_drug_info...")
    info = tools.get_drug_info("warfarin")
    assert info["name"] == "warfarin"
    assert "Anticoagulant" in info["drug_class"]
    assert "CYP2C9" in info["metabolism"]
    print(f"    ✓ warfarin: {info['drug_class']}, metabolism={info['metabolism'][:30]}")

    # Test brand name lookup
    info_brand = tools.get_drug_info("Prozac")
    assert info_brand["name"] == "fluoxetine"
    print(f"    ✓ Brand lookup 'Prozac' → {info_brand['name']}")

    # Test not found
    info_missing = tools.get_drug_info("FakeDrug123")
    assert "error" in info_missing
    print(f"    ✓ Not found: {info_missing['error'][:50]}")

    # 2b. search_drugs_by_class
    print("  Testing search_drugs_by_class...")
    ssris = tools.search_drugs_by_class("SSRI")
    assert len(ssris) >= 1
    assert any(d["name"] == "fluoxetine" for d in ssris)
    print(f"    ✓ SSRI class: {[d['name'] for d in ssris]}")

    # 2c. check_interaction
    print("  Testing check_interaction...")
    result = tools.check_interaction("warfarin", "aspirin")
    assert result["severity"] == "major"
    assert "bleeding" in result["effect"].lower()
    print(f"    ✓ warfarin + aspirin: {result['severity']} — {result['effect'][:60]}")

    result2 = tools.check_interaction("fluoxetine", "tramadol")
    assert result2["severity"] == "major"
    assert "serotonin" in result2["effect"].lower()
    print(f"    ✓ fluoxetine + tramadol: {result2['severity']} — {result2['effect'][:60]}")

    result3 = tools.check_interaction("metformin", "aspirin")
    assert result3["severity"] == "none"
    print(f"    ✓ metformin + aspirin: {result3['severity']}")

    # 2d. check_all_interactions
    print("  Testing check_all_interactions...")
    all_int = tools.check_all_interactions("DI_P003")
    major_count = sum(1 for r in all_int if r.get("severity") in ("major", "contraindicated"))
    print(f"    ✓ DI_P003: {len(all_int)} interactions, {major_count} major/contraindicated")
    for r in all_int:
        if "drug_a" in r:
            print(f"      - {r['drug_a']} + {r['drug_b']}: {r['severity']}")

    # 2e. get_patient_medications
    print("  Testing get_patient_medications...")
    profile = tools.get_patient_medications("DI_P001")
    assert profile["patient_id"] == "DI_P001"
    assert "warfarin" in profile["current_medications"]
    print(f"    ✓ DI_P001: {profile['current_medications']}, age={profile['age']}")

    # 2f. search_alternatives
    print("  Testing search_alternatives...")
    alts = tools.search_alternatives("warfarin")
    assert len(alts) >= 1
    alt_names = [a.get("drug_name", "") for a in alts]
    assert "apixaban" in alt_names
    print(f"    ✓ Alternatives for warfarin: {alt_names}")

    alts_class = tools.search_alternatives("simvastatin")
    print(f"    ✓ Alternatives for simvastatin: {[a.get('drug_name', '') for a in alts_class]}")

    # 2g. check_dosage
    print("  Testing check_dosage...")
    dosage = tools.check_dosage("metformin", patient_id="DI_P001")
    assert dosage["drug_name"] == "metformin"
    assert "renal_adjustment" in dosage
    print(f"    ✓ metformin dosage: {dosage['typical_dosage'][:50]}")
    print(f"    ✓ Renal adjustment: {dosage.get('renal_adjustment', 'N/A')[:60]}")

    # 2h. think
    print("  Testing think...")
    assert tools.think("Analyzing pharmacological interactions...") == ""
    print(f"    ✓ Think tool works")

    # 2i. submit_answer
    print("  Testing submit_answer...")
    answer = tools.submit_answer("Contraindicated due to bleeding risk", "Based on CYP interaction")
    assert "submitted" in answer.lower()
    print(f"    ✓ Answer submitted")

    # 2j. assertion helpers
    print("  Testing assertion helpers...")
    assert tools.assert_interaction_found("warfarin", "aspirin")
    assert tools.assert_interaction_found("fluoxetine", "tramadol")
    print(f"    ✓ Assertions verified")

    # Tool statistics
    stats = tools.get_statistics()
    print(f"\n  Tool Statistics: {stats['num_tools']} total")
    print(f"    Tools: {stats['tool_names']}")
    print("  ✓ All tool tests PASSED")


def test_environment():
    """Test environment setup and interaction."""
    print("\n=== Test 3: Environment ===")

    env = get_environment()
    assert env.domain_name == "drug_interaction"
    assert env.tools is not None
    assert len(env.policy) > 0

    obs, info = env.reset()
    assert "policy" in info
    assert "tools" in info
    print(f"  ✓ Environment created: {env.domain_name}")
    print(f"  ✓ Policy length: {len(env.policy)} chars")
    print(f"  ✓ Tools available: {len(info['tools'])}")

    # Test tool execution via environment
    result = env.execute_tool("get_drug_info", drug_name="warfarin")
    assert not result.error
    print(f"  ✓ Tool execution through environment works")

    # Test step with tool call
    action = json.dumps({"name": "check_interaction", "arguments": {"drug_a": "warfarin", "drug_b": "aspirin"}})
    obs, reward, terminated, truncated, info = env.step(action)
    assert "major" in obs.lower() or "bleeding" in obs.lower()
    print(f"  ✓ Environment step works. Turn: {info['turn_count']}")

    print("  ✓ Environment tests PASSED")


def test_tasks():
    """Test task loading."""
    print("\n=== Test 4: Tasks ===")

    tasks = get_tasks()
    assert len(tasks) == 5, f"Expected 5 tasks, got {len(tasks)}"
    print(f"  ✓ Loaded {len(tasks)} tasks")

    for task in tasks:
        tid = task["id"]
        desc = task.get("description", {})
        eval_criteria = task.get("evaluation_criteria", {})
        n_actions = len(eval_criteria.get("actions", []))
        n_assertions = len(eval_criteria.get("nl_assertions", []))
        print(f"    - {tid}: difficulty={desc.get('difficulty', '?')}, "
              f"actions={n_actions}, assertions={n_assertions}")

    print("  ✓ Task loading tests PASSED")


def test_gym_interface():
    """Test Gymnasium-compatible interface."""
    print("\n=== Test 5: GYM Interface ===")

    from bioagents.gym.agent_env import BioAgentGymEnv

    env = BioAgentGymEnv(
        domain="drug_interaction",
        task_id="di_warfarin_aspirin_001",
        max_turns=10,
    )

    obs, info = env.reset()
    assert "di_warfarin_aspirin_001" in obs
    assert info["domain"] == "drug_interaction"
    assert len(info["tools"]) > 0
    print(f"  ✓ GYM env created. Task: {info['task_id']}")
    print(f"  ✓ Tools: {len(info['tools'])}")

    # Simulate drug interaction workflow
    actions = [
        json.dumps({"name": "get_patient_medications", "arguments": {"patient_id": "DI_P001"}}),
        json.dumps({"name": "get_drug_info", "arguments": {"drug_name": "warfarin"}}),
        json.dumps({"name": "check_interaction", "arguments": {"drug_a": "warfarin", "drug_b": "aspirin"}}),
        json.dumps({"name": "submit_answer", "arguments": {"answer": "Major bleeding risk", "reasoning": "warfarin+aspirin"}}),
    ]

    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        tool_name = json.loads(action)["name"]
        print(f"  Step {i+1}: {tool_name} → reward={reward:.2f}, done={terminated or truncated}")

    trajectory = env.get_trajectory()
    print(f"\n  ✓ Trajectory: {trajectory['total_turns']} turns, "
          f"{len(trajectory['tool_call_log'])} tool calls")
    print(f"  ✓ Final reward: {trajectory['final_reward']:.3f}")
    print("  ✓ GYM interface tests PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("BIOAgents Drug Interaction Domain - Test Suite")
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
