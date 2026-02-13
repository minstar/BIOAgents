"""Test script for the EHR Management domain.

Tests:
1. DB loading and schema validation
2. Tool execution (all tools)
3. Environment setup
4. Task loading and split filtering
5. Gym interface
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bioagents.domains.ehr_management.data_model import (
    EHRDB, DB_PATH, POLICY_PATH, TASKS_PATH,
)
from bioagents.domains.ehr_management.tools import EHRTools
from bioagents.domains.ehr_management.environment import get_environment, get_tasks


def test_db_loading():
    """Test database loading and validation."""
    print("\n=== Test 1: DB Loading ===")
    db = EHRDB.load(DB_PATH)

    assert len(db.records) == 4, f"Expected 4 records, got {len(db.records)}"
    assert len(db.patient_index) == 3, f"Expected 3 patients, got {len(db.patient_index)}"
    assert len(db.lab_reference_ranges) == 14
    assert len(db.icd_descriptions) == 12

    # Check specific records
    assert "HADM_10001" in db.records
    assert "HADM_10002" in db.records
    assert "HADM_10003" in db.records
    assert "HADM_9001" in db.records

    # Check patient index
    assert "P2001" in db.patient_index
    assert len(db.patient_index["P2001"]) == 2  # Two admissions

    # Check record structure
    rec = db.records["HADM_10001"]
    assert rec.demographics.name == "Robert Chen"
    assert rec.demographics.age == 68
    assert rec.admission.diagnosis_at_admission == "Acute decompensated heart failure"
    assert len(rec.lab_events) == 20
    assert len(rec.vital_events) == 6
    assert len(rec.medication_orders) == 7
    assert len(rec.procedures) == 2
    assert len(rec.icu_stays) == 1
    assert rec.discharge_summary is not None
    assert rec.admission.is_readmission is True

    # Check still-admitted patient
    rec_icu = db.records["HADM_10002"]
    assert rec_icu.discharge_summary is None
    assert rec_icu.admission.discharge_time is None

    # Hash determinism
    h1 = db.get_hash()
    h2 = db.get_hash()
    assert h1 == h2

    # Dump and reload
    test_path = "/tmp/test_ehr_db.json"
    db.dump(test_path)
    db_reloaded = EHRDB.load(test_path)
    assert db.get_hash() == db_reloaded.get_hash()
    os.remove(test_path)

    print(f"  ✓ Loaded {len(db.records)} admission records")
    print(f"  ✓ {len(db.patient_index)} patients")
    print(f"  ✓ {len(db.lab_reference_ranges)} lab reference ranges")
    print(f"  ✓ DB hash: {h1}")
    print("  ✓ DB loading test PASSED")


def test_tools():
    """Test all EHR tools."""
    print("\n=== Test 2: Tool Execution ===")
    db = EHRDB.load(DB_PATH)
    tools = EHRTools(db)

    stats = tools.get_statistics()
    assert stats["num_tools"] == 14
    print(f"  Total tools: {stats['num_tools']}")
    print(f"  Tools: {stats['tool_names']}")

    # 2a. get_patient_summary
    print("  Testing get_patient_summary...")
    summary = tools.get_patient_summary("HADM_10001")
    assert summary["demographics"]["name"] == "Robert Chen"
    assert summary["admission"]["is_readmission"] is True
    assert "furosemide" in summary["active_medications"]
    assert "carvedilol" in summary["active_medications"]
    print(f"    ✓ HADM_10001: {summary['demographics']['name']}, "
          f"active meds: {summary['active_medications']}")

    # 2b. get_admission_history
    print("  Testing get_admission_history...")
    history = tools.get_admission_history("P2001")
    assert len(history) == 2
    assert history[0]["hadm_id"] == "HADM_9001"  # Earlier admission first
    assert history[1]["hadm_id"] == "HADM_10001"
    print(f"    ✓ P2001: {len(history)} admissions")

    # 2c. get_lab_results (filtered)
    print("  Testing get_lab_results...")
    bnp_labs = tools.get_lab_results("HADM_10001", lab_name="BNP")
    assert len(bnp_labs) == 5
    assert all("BNP" in l["label"] for l in bnp_labs)
    print(f"    ✓ HADM_10001 BNP: {len(bnp_labs)} results")

    # get_lab_results (all)
    all_labs = tools.get_lab_results("HADM_10002")
    assert len(all_labs) == 10  # last_n=10 default
    print(f"    ✓ HADM_10002 all labs: {len(all_labs)} (capped at 10)")

    # 2d. get_lab_trend
    print("  Testing get_lab_trend...")
    trend = tools.get_lab_trend("HADM_10001", "BNP")
    assert trend["trend"] == "falling"
    assert trend["max_value"] == 1850.0
    assert trend["min_value"] == 320.0
    assert len(trend["values"]) == 5
    print(f"    ✓ BNP trend: {trend['trend']}, {trend['values']}")

    trend_cr = tools.get_lab_trend("HADM_10002", "Creatinine")
    # Cr goes 2.3→2.8→3.1→2.6 (peak-and-resolve), classified as stable
    assert trend_cr["trend"] in ("rising", "stable")
    assert trend_cr["max_value"] == 3.1
    print(f"    ✓ Creatinine trend: {trend_cr['trend']}, {trend_cr['values']}")

    # 2e. get_vital_signs
    print("  Testing get_vital_signs...")
    vitals = tools.get_vital_signs("HADM_10002")
    assert len(vitals) == 5
    assert vitals[0]["charttime"] > vitals[-1]["charttime"]  # Reversed (most recent first)
    print(f"    ✓ HADM_10002 vitals: {len(vitals)} readings")

    # 2f. detect_vital_alerts
    print("  Testing detect_vital_alerts...")
    alerts = tools.detect_vital_alerts("HADM_10002")
    # Latest vitals for 10002 should be stable (day 3)
    assert len(alerts) >= 1
    print(f"    ✓ HADM_10002 alerts: {len(alerts)} alert(s)")

    # 2g. get_medication_orders
    print("  Testing get_medication_orders...")
    all_meds = tools.get_medication_orders("HADM_10003")
    assert len(all_meds) == 6
    print(f"    ✓ HADM_10003 all meds: {len(all_meds)}")

    active_meds = tools.get_medication_orders("HADM_10003", active_only=True)
    active_count = sum(1 for m in all_meds if m["status"] == "active")
    assert len(active_meds) == active_count
    print(f"    ✓ HADM_10003 active meds: {len(active_meds)}")

    # 2h. get_clinical_scores
    print("  Testing get_clinical_scores...")
    scores = tools.get_clinical_scores("HADM_10002")
    assert len(scores) == 4
    score_names = [s["score_name"] for s in scores]
    assert "SOFA" in score_names
    assert "qSOFA" in score_names
    assert "NEWS2" in score_names
    print(f"    ✓ HADM_10002 scores: {score_names}")

    # No scores case
    no_scores = tools.get_clinical_scores("HADM_9001")
    assert "message" in no_scores[0]
    print(f"    ✓ HADM_9001: no scores (as expected)")

    # 2i. get_quality_indicators
    print("  Testing get_quality_indicators...")
    qi = tools.get_quality_indicators("HADM_10001")
    assert qi["readmission_risk"] == 0.35
    assert qi["aki_stage"] == 2
    print(f"    ✓ HADM_10001: readmission_risk={qi['readmission_risk']}, AKI stage={qi['aki_stage']}")

    # 2j. get_procedures
    print("  Testing get_procedures...")
    procs = tools.get_procedures("HADM_10003")
    assert len(procs) == 2
    proc_names = [p["procedure_name"] for p in procs]
    assert "Primary PCI — LAD stenting" in proc_names
    print(f"    ✓ HADM_10003 procedures: {proc_names}")

    # 2k. get_discharge_summary
    print("  Testing get_discharge_summary...")
    disch = tools.get_discharge_summary("HADM_10003")
    assert "STEMI" in disch["text"]
    assert len(disch["discharge_medications"]) == 6
    print(f"    ✓ HADM_10003 discharge: {len(disch['diagnoses'])} diagnoses, "
          f"{len(disch['discharge_medications'])} meds")

    # Not yet discharged
    disch_pending = tools.get_discharge_summary("HADM_10002")
    assert "message" in disch_pending
    print(f"    ✓ HADM_10002: not yet discharged")

    # 2l. lookup_icd_code
    print("  Testing lookup_icd_code...")
    icd = tools.lookup_icd_code("I50.31")
    assert icd["description"] != ""
    print(f"    ✓ I50.31: {icd['description']}")

    icd_miss = tools.lookup_icd_code("Z99.99")
    assert "not found" in icd_miss["description"].lower() or "related" in str(icd_miss).lower()
    print(f"    ✓ Z99.99: not found (expected)")

    # 2m. think
    print("  Testing think...")
    assert tools.think("Analyzing BNP trend for heart failure prognosis.") == ""
    print(f"    ✓ Think tool works")

    # 2n. submit_answer
    print("  Testing submit_answer...")
    ans = tools.submit_answer("Patient improving", "BNP trend falling, vitals stable")
    assert "submitted" in ans.lower()
    print(f"    ✓ Answer submitted")

    print("  ✓ All tool tests PASSED")


def test_environment():
    """Test environment setup and interaction."""
    print("\n=== Test 3: Environment ===")

    env = get_environment()
    assert env.domain_name == "ehr_management"
    assert env.tools is not None
    assert len(env.policy) > 0

    obs, info = env.reset()
    assert "policy" in info
    assert "tools" in info
    assert len(info["tools"]) == 14
    print(f"  ✓ Environment created: {env.domain_name}")
    print(f"  ✓ Policy length: {len(env.policy)} chars")
    print(f"  ✓ Tools available: {len(info['tools'])}")

    # Test tool execution via environment
    result = env.execute_tool("get_patient_summary", hadm_id="HADM_10001")
    assert not result.error
    data = json.loads(result.content)
    assert data["demographics"]["name"] == "Robert Chen"
    print(f"  ✓ Tool execution through environment works")

    # Test step with tool call
    action = json.dumps({
        "name": "get_lab_results",
        "arguments": {"hadm_id": "HADM_10002", "lab_name": "WBC"}
    })
    obs, reward, terminated, truncated, info = env.step(action)
    assert "WBC" in obs
    print(f"  ✓ Environment step works. Turn: {info['turn_count']}")

    print("  ✓ Environment tests PASSED")


def test_tasks():
    """Test task loading and split filtering."""
    print("\n=== Test 4: Tasks ===")

    tasks = get_tasks()
    assert len(tasks) == 15, f"Expected 15 tasks, got {len(tasks)}"
    print(f"  ✓ Loaded {len(tasks)} tasks total")

    # Test splits
    train_tasks = get_tasks("train")
    test_tasks = get_tasks("test")
    assert len(train_tasks) == 8, f"Expected 8 train tasks, got {len(train_tasks)}"
    assert len(test_tasks) == 7, f"Expected 7 test tasks, got {len(test_tasks)}"
    assert len(train_tasks) + len(test_tasks) == len(tasks)
    print(f"  ✓ Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Verify task structure
    for task in tasks:
        assert "id" in task
        assert "domain" in task
        assert "category" in task
        assert "difficulty" in task
        assert "ticket" in task
        assert "expected_answer" in task
        assert "rubric" in task
        assert task["domain"] == "ehr_management"

    # Check category diversity
    categories = set(t["category"] for t in tasks)
    assert len(categories) >= 8, f"Expected ≥8 categories, got {len(categories)}"
    print(f"  ✓ Categories: {categories}")

    # Check difficulty distribution
    difficulties = [t["difficulty"] for t in tasks]
    assert "medium" in difficulties
    assert "hard" in difficulties
    print(f"  ✓ Difficulty distribution: medium={difficulties.count('medium')}, hard={difficulties.count('hard')}")

    # Show tasks
    for t in tasks:
        print(f"    - {t['id']}: [{t['category']}] [{t['difficulty']}] {t['split']}")

    print("  ✓ Task tests PASSED")


def test_gym_interface():
    """Test Gymnasium-compatible interface."""
    print("\n=== Test 5: GYM Interface ===")

    from bioagents.gym.agent_env import BioAgentGymEnv

    env = BioAgentGymEnv(
        domain="ehr_management",
        task_id="ehr_001",
        max_turns=10,
    )

    obs, info = env.reset()
    assert "ehr_001" in obs
    assert "EHR" in obs
    assert info["domain"] == "ehr_management"
    assert len(info["tools"]) == 14
    print(f"  ✓ GYM env created. Task: {info['task_id']}")
    print(f"  ✓ Tools: {len(info['tools'])}")

    # Simulate EHR chart review workflow
    actions = [
        json.dumps({"name": "get_patient_summary", "arguments": {"hadm_id": "HADM_10001"}}),
        json.dumps({"name": "get_lab_results", "arguments": {"hadm_id": "HADM_10001"}}),
        json.dumps({"name": "get_vital_signs", "arguments": {"hadm_id": "HADM_10001"}}),
        json.dumps({"name": "get_medication_orders", "arguments": {"hadm_id": "HADM_10001"}}),
        json.dumps({"name": "think", "arguments": {"thought": "BNP trending down, vitals improving."}}),
        json.dumps({"name": "submit_answer", "arguments": {
            "answer": "Patient improving — BNP falling, vitals stable",
            "reasoning": "Based on BNP trend and vital sign improvement"
        }}),
    ]

    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        tool_name = json.loads(action)["name"]
        print(f"  Step {i+1}: {tool_name} → reward={reward:.2f}, "
              f"done={terminated or truncated}")

    trajectory = env.get_trajectory()
    print(f"\n  ✓ Trajectory: {trajectory['total_turns']} turns, "
          f"{len(trajectory['tool_call_log'])} tool calls")
    print(f"  ✓ Final reward: {trajectory['final_reward']:.3f}")

    # Test random task selection
    env2 = BioAgentGymEnv(domain="ehr_management", max_turns=5)
    obs2, info2 = env2.reset(seed=42)
    assert info2["domain"] == "ehr_management"
    print(f"  ✓ Random task: {info2['task_id']}")

    print("  ✓ GYM interface tests PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("BIOAgents EHR Management Domain - Test Suite")
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
