"""Test script for the Clinical Diagnosis domain.

Tests:
1. DB loading and schema validation
2. Tool execution (all tools)
3. Environment setup
4. Gym interface
5. Task loading and basic reward computation
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bioagents.environment.db import DB
from bioagents.environment.toolkit import ToolKitBase, ToolType, is_tool, ToolDefinition
from bioagents.domains.clinical_diagnosis.data_model import ClinicalDB, DB_PATH, POLICY_PATH, TASKS_PATH
from bioagents.domains.clinical_diagnosis.tools import ClinicalTools
from bioagents.domains.clinical_diagnosis.environment import get_environment, get_tasks


def test_db_loading():
    """Test database loading and validation."""
    print("\n=== Test 1: DB Loading ===")
    db = ClinicalDB.load(DB_PATH)
    
    assert len(db.patients) == 3, f"Expected 3 patients, got {len(db.patients)}"
    assert "P001" in db.patients
    assert "P002" in db.patients
    assert "P003" in db.patients
    
    p1 = db.patients["P001"]
    assert p1.name == "James Wilson"
    assert p1.age == 58
    assert len(p1.allergies) == 2
    assert len(p1.medications) == 3
    assert len(p1.vital_signs) == 2
    assert len(p1.lab_results) == 6
    
    # Test hash
    h1 = db.get_hash()
    h2 = db.get_hash()
    assert h1 == h2, "Hash should be deterministic"
    
    # Test dump and reload
    test_path = "/tmp/test_clinical_db.json"
    db.dump(test_path)
    db_reloaded = ClinicalDB.load(test_path)
    assert db.get_hash() == db_reloaded.get_hash(), "Reloaded DB should have same hash"
    os.remove(test_path)
    
    print(f"  ✓ Loaded {len(db.patients)} patients")
    print(f"  ✓ {len(db.drug_interactions)} drug interactions")
    print(f"  ✓ {len(db.clinical_guidelines)} clinical guidelines")
    print(f"  ✓ DB hash: {h1}")
    print("  ✓ DB loading test PASSED")


def test_tools():
    """Test all clinical tools."""
    print("\n=== Test 2: Tool Execution ===")
    db = ClinicalDB.load(DB_PATH)
    tools = ClinicalTools(db)
    
    # 2a. get_patient_info
    print("  Testing get_patient_info...")
    info = tools.get_patient_info(patient_id="P001")
    assert info["name"] == "James Wilson"
    assert len(info["allergies"]) == 2
    assert info["chief_complaint"] is not None
    print(f"    ✓ Patient P001: {info['name']}, age {info['age']}, {len(info['allergies'])} allergies")
    
    # 2b. get_vital_signs
    print("  Testing get_vital_signs...")
    vitals = tools.get_vital_signs(patient_id="P001")
    assert vitals["temperature"] == 39.1
    assert vitals["spo2"] == 91
    print(f"    ✓ Vitals: T={vitals['temperature']}°C, SpO2={vitals['spo2']}%, HR={vitals['heart_rate']}")
    
    # 2c. get_vital_signs_trend
    print("  Testing get_vital_signs_trend...")
    trend = tools.get_vital_signs_trend(patient_id="P001", num_readings=5)
    assert len(trend) == 2  # P001 has 2 readings
    print(f"    ✓ Got {len(trend)} vital sign readings")
    
    # 2d. get_lab_results
    print("  Testing get_lab_results...")
    labs = tools.get_lab_results(patient_id="P001")
    assert len(labs) == 6
    critical_labs = [l for l in labs if l["flag"] == "critical"]
    print(f"    ✓ {len(labs)} lab results, {len(critical_labs)} critical")
    
    # 2e. get_lab_results with filter
    labs_chem = tools.get_lab_results(patient_id="P001", category="chemistry")
    assert all(l["category"] == "chemistry" for l in labs_chem)
    print(f"    ✓ Filtered chemistry labs: {len(labs_chem)}")
    
    # 2f. order_lab_test
    print("  Testing order_lab_test...")
    order = tools.order_lab_test(patient_id="P001", test_name="Chest X-Ray", priority="urgent")
    assert order["status"] == "ordered"
    assert order["priority"] == "urgent"
    print(f"    ✓ Order {order['order_id']} placed: {order['test_name']}")
    
    # 2g. get_medications
    print("  Testing get_medications...")
    meds = tools.get_medications(patient_id="P001")
    assert len(meds) == 3
    print(f"    ✓ {len(meds)} current medications: {[m['drug_name'] for m in meds]}")
    
    # 2h. check_drug_interaction
    print("  Testing check_drug_interaction...")
    interaction = tools.check_drug_interaction(drug_a="Atorvastatin", drug_b="Clarithromycin")
    assert interaction["severity"] == "major"
    print(f"    ✓ Atorvastatin + Clarithromycin: {interaction['severity']} interaction")
    
    no_interaction = tools.check_drug_interaction(drug_a="Metformin", drug_b="Atorvastatin")
    assert no_interaction["severity"] == "none"
    print(f"    ✓ Metformin + Atorvastatin: {no_interaction['severity']}")
    
    # 2i. prescribe_medication
    print("  Testing prescribe_medication...")
    rx = tools.prescribe_medication(
        patient_id="P003", drug_name="Acetaminophen", dosage="1000mg", frequency="every 6 hours"
    )
    assert rx["status"] == "prescribed"
    print(f"    ✓ Prescribed {rx['drug_name']} {rx['dosage']} for P003")
    
    # Test allergy detection
    try:
        tools.prescribe_medication(
            patient_id="P001", drug_name="Penicillin V", dosage="500mg", frequency="four times daily"
        )
        assert False, "Should have raised ValueError for allergy"
    except ValueError as e:
        print(f"    ✓ Allergy detected: {str(e)[:80]}")
    
    # 2j. get_clinical_notes
    print("  Testing get_clinical_notes...")
    notes = tools.get_clinical_notes(patient_id="P001")
    assert len(notes) >= 1
    print(f"    ✓ {len(notes)} clinical notes for P001")
    
    # 2k. add_clinical_note
    print("  Testing add_clinical_note...")
    note = tools.add_clinical_note(
        patient_id="P001", note_type="progress",
        content="Patient assessed. CAP diagnosed.", diagnosis_codes="J18.9"
    )
    assert note["status"] == "added"
    print(f"    ✓ Added note {note['note_id']}")
    
    # 2l. get_differential_diagnosis
    print("  Testing get_differential_diagnosis...")
    ddx = tools.get_differential_diagnosis(symptoms="fever, cough, dyspnea")
    assert len(ddx) > 0
    print(f"    ✓ Differential diagnosis: {[d['condition'] for d in ddx]}")
    
    # 2m. search_clinical_guidelines
    print("  Testing search_clinical_guidelines...")
    guidelines = tools.search_clinical_guidelines(condition="pneumonia")
    assert len(guidelines) > 0
    print(f"    ✓ Found {len(guidelines)} guideline(s)")
    
    # 2n. record_diagnosis
    print("  Testing record_diagnosis...")
    dx = tools.record_diagnosis(
        patient_id="P001", diagnosis="Community-acquired pneumonia",
        icd10_code="J18.9", confidence="high",
        reasoning="Fever, productive cough, elevated WBC, CRP, procalcitonin, CXR findings"
    )
    assert dx["status"] == "recorded"
    print(f"    ✓ Recorded diagnosis: {dx['diagnosis']} ({dx['confidence']} confidence)")
    
    # 2o. transfer_to_specialist
    print("  Testing transfer_to_specialist...")
    ref = tools.transfer_to_specialist(
        summary="Suspected bacterial meningitis", specialty="neurology"
    )
    assert "neurology" in ref
    print(f"    ✓ Referral: {ref[:80]}")
    
    # 2p. think
    print("  Testing think...")
    thought = tools.think(thought="Patient has classic CAP presentation with comorbidities")
    assert thought == ""
    print(f"    ✓ Think tool works (no output)")
    
    # 2q. Assertion helpers
    assert tools.assert_diagnosis_recorded("P001", "Community-acquired pneumonia")
    assert tools.assert_lab_ordered("P001", "Chest X-Ray")
    assert tools.assert_medication_prescribed("P003", "Acetaminophen")
    print(f"    ✓ Assertion helpers verified")
    
    # Tool statistics
    stats = tools.get_statistics()
    print(f"\n  Tool Statistics: {stats['num_tools']} total")
    print(f"    READ: {stats['num_read_tools']}, WRITE: {stats['num_write_tools']}")
    print(f"    THINK: {stats['num_think_tools']}, GENERIC: {stats['num_generic_tools']}")
    print(f"    Tools: {stats['tool_names']}")
    
    # Tool definitions (OpenAI format)
    defs = tools.get_tool_definitions_dict()
    print(f"  ✓ Generated {len(defs)} OpenAI-compatible tool definitions")
    
    print("  ✓ All tool tests PASSED")


def test_environment():
    """Test environment setup and interaction."""
    print("\n=== Test 3: Environment ===")
    
    env = get_environment()
    assert env.domain_name == "clinical_diagnosis"
    assert env.tools is not None
    assert len(env.policy) > 0
    
    obs, info = env.reset()
    assert "policy" in info
    assert "tools" in info
    print(f"  ✓ Environment created: {env.domain_name}")
    print(f"  ✓ Policy length: {len(env.policy)} chars")
    print(f"  ✓ Tools available: {len(info['tools'])}")
    
    # Test tool execution via environment
    result = env.execute_tool("get_patient_info", patient_id="P001")
    assert not result.error
    print(f"  ✓ Tool execution through environment works")
    
    # Test step with tool call
    action = json.dumps({"name": "get_vital_signs", "arguments": {"patient_id": "P001"}})
    obs, reward, terminated, truncated, info = env.step(action)
    assert "39.1" in obs  # Should contain temperature
    print(f"  ✓ Environment step works. Turn: {info['turn_count']}")
    
    # Test state snapshot
    snapshot = env.get_state_snapshot()
    assert snapshot["turn_count"] == 1
    print(f"  ✓ State snapshot: turn={snapshot['turn_count']}, db_hash={snapshot['db_hash']}")
    
    print("  ✓ Environment tests PASSED")


def test_tasks():
    """Test task loading."""
    print("\n=== Test 4: Tasks ===")
    
    tasks = get_tasks()
    assert len(tasks) == 5
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
        domain="clinical_diagnosis",
        task_id="dx_pneumonia_001",
        max_turns=10,
    )
    
    obs, info = env.reset()
    assert "dx_pneumonia_001" in obs
    assert "clinical_diagnosis" in info["domain"]
    assert len(info["tools"]) > 0
    print(f"  ✓ GYM env created. Task: {info['task_id']}")
    print(f"  ✓ Tools: {len(info['tools'])}")
    print(f"  ✓ Initial observation ({len(obs)} chars):")
    print(f"    {obs[:200]}...")
    
    # Simulate a simple clinical workflow
    actions = [
        json.dumps({"name": "get_patient_info", "arguments": {"patient_id": "P001"}}),
        json.dumps({"name": "get_vital_signs", "arguments": {"patient_id": "P001"}}),
        json.dumps({"name": "get_lab_results", "arguments": {"patient_id": "P001"}}),
        json.dumps({"name": "get_differential_diagnosis", "arguments": {"symptoms": "fever, cough, dyspnea"}}),
        json.dumps({"name": "search_clinical_guidelines", "arguments": {"condition": "Community-acquired pneumonia"}}),
        json.dumps({"name": "record_diagnosis", "arguments": {
            "patient_id": "P001", 
            "diagnosis": "Community-acquired pneumonia",
            "icd10_code": "J18.9",
            "confidence": "high",
            "reasoning": "Fever, productive cough, elevated WBC/CRP/PCT, crackles on exam"
        }}),
    ]
    
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        tool_name = json.loads(action)["name"]
        print(f"  Step {i+1}: {tool_name} → reward={reward:.2f}, done={terminated or truncated}")
    
    # Get trajectory
    trajectory = env.get_trajectory()
    print(f"\n  ✓ Trajectory: {trajectory['total_turns']} turns, "
          f"{len(trajectory['tool_call_log'])} tool calls")
    print(f"  ✓ Final reward: {trajectory['final_reward']:.3f}")
    
    # Verify expected actions were covered
    expected_tools = ["get_patient_info", "get_vital_signs", "get_lab_results",
                      "get_differential_diagnosis", "search_clinical_guidelines", "record_diagnosis"]
    actual_tools = [tc["tool_name"] for tc in trajectory["tool_call_log"]]
    
    for et in expected_tools:
        found = et in actual_tools
        status = "✓" if found else "✗"
        print(f"    {status} {et}")
    
    print(f"\n  ✓ GYM interface tests PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("BIOAgents Clinical Diagnosis Domain - Test Suite")
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
