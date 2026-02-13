"""Cross-domain Clinical Pathway Engine.

Orchestrates multi-phase patient journeys that cross domain boundaries.
Each pathway defines a realistic clinical workflow:

  Phase 1 (Triage):        Patient arrives at ED → ESI assessment
  Phase 2 (Diagnosis):     History, physical, labs → differential diagnosis
  Phase 3 (Imaging):       Order and interpret imaging studies
  Phase 4 (Drug Mgmt):     Medication reconciliation, check interactions
  Phase 5 (EHR):           Document findings, update records
  Phase 6 (Disposition):   Admit/discharge planning, follow-up

The engine:
1. Loads tools from multiple domain environments
2. Merges databases so cross-domain references resolve
3. Tracks phase transitions and evaluates per-phase + overall
4. Provides a unified Gymnasium interface
"""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "domains"


# ============================================================
# 1. Pathway Phase Definition
# ============================================================

@dataclass
class PathwayPhase:
    """A single phase in a cross-domain clinical pathway."""
    phase_id: str
    domain: str                     # Which domain's tools are active
    name: str                       # Human-readable name
    description: str                # What the agent should accomplish
    required_actions: list[dict]    # Expected tool calls for this phase
    assertions: list[str]           # NL assertions to evaluate
    transition_condition: str       # How to determine phase completion
    max_turns: int = 5              # Max turns for this phase
    time_pressure: bool = False     # Whether this is time-critical


@dataclass
class ClinicalPathway:
    """A complete cross-domain clinical pathway."""
    pathway_id: str
    title: str
    patient_summary: str
    clinical_context: str
    phases: list[PathwayPhase]
    overall_assertions: list[str]
    difficulty: str                 # "moderate", "hard", "expert"
    estimated_turns: int
    safety_critical: bool = False
    patient_data: dict = field(default_factory=dict)


# ============================================================
# 2. Pathway Definitions
# ============================================================

def get_pathway_definitions() -> list[ClinicalPathway]:
    """Return all defined cross-domain clinical pathways."""
    return [
        _pathway_chest_pain_ed(),
        _pathway_diabetic_emergency(),
        _pathway_stroke_code(),
        _pathway_sepsis_bundle(),
        _pathway_postop_complication(),
        _pathway_pediatric_fever(),
    ]


def _pathway_chest_pain_ed() -> ClinicalPathway:
    """ED Chest Pain: Triage → Dx → Imaging → Drugs → EHR → Disposition."""
    return ClinicalPathway(
        pathway_id="xd_chest_pain_001",
        title="Acute Chest Pain in the Emergency Department",
        patient_summary=(
            "Robert Chen, 62M, presents to the ED with acute onset substernal "
            "chest pain radiating to the left arm, diaphoresis, nausea. "
            "PMH: HTN, HLD, DM2, smoking 30 pack-years. "
            "Current meds: metformin 1000mg BID, lisinopril 20mg daily, "
            "atorvastatin 40mg daily."
        ),
        clinical_context=(
            "This is a high-acuity ED presentation requiring rapid triage, "
            "immediate ECG and troponin, possible cath lab activation, "
            "and careful medication management considering drug interactions."
        ),
        phases=[
            PathwayPhase(
                phase_id="triage",
                domain="triage_emergency",
                name="ED Triage Assessment",
                description="Assess vital signs, determine ESI level, identify time-critical condition",
                required_actions=[
                    {"name": "get_patient_presentation", "arguments": {"patient_id": "EP001"}},
                    {"name": "get_vital_signs", "arguments": {"patient_id": "EP001"}},
                    {"name": "calculate_esi_level", "arguments": {"patient_id": "EP001"}},
                ],
                assertions=[
                    "Agent should assign ESI-1 or ESI-2 for acute chest pain with cardiac features",
                    "Agent should recognize this as a potential STEMI/ACS",
                    "Agent should flag as time-critical",
                ],
                transition_condition="ESI level assigned and urgency communicated",
                max_turns=3,
                time_pressure=True,
            ),
            PathwayPhase(
                phase_id="diagnosis",
                domain="clinical_diagnosis",
                name="Clinical Assessment & Diagnosis",
                description="Perform history, physical, order labs, establish differential",
                required_actions=[
                    {"name": "get_patient_info", "arguments": {"patient_id": "P001"}},
                    {"name": "get_lab_results", "arguments": {"patient_id": "P001"}},
                    {"name": "get_vital_signs", "arguments": {"patient_id": "P001"}},
                ],
                assertions=[
                    "Agent should identify acute coronary syndrome as primary diagnosis",
                    "Agent should order troponin, ECG, CBC, BMP, coagulation studies",
                    "Agent should consider differential: ACS, PE, aortic dissection, pericarditis",
                ],
                transition_condition="Diagnosis established with supporting evidence",
                max_turns=5,
            ),
            PathwayPhase(
                phase_id="imaging",
                domain="visual_diagnosis",
                name="Imaging Interpretation",
                description="Review chest X-ray and ECG findings",
                required_actions=[
                    {"name": "analyze_medical_image", "arguments": {"image_id": "IMG001"}},
                    {"name": "get_image_report", "arguments": {"image_id": "IMG001"}},
                ],
                assertions=[
                    "Agent should interpret chest X-ray findings correctly",
                    "Agent should correlate imaging with clinical picture",
                ],
                transition_condition="Imaging interpreted and correlated with diagnosis",
                max_turns=3,
            ),
            PathwayPhase(
                phase_id="drug_management",
                domain="drug_interaction",
                name="Medication Management",
                description="Check current medications, assess interactions with new drugs",
                required_actions=[
                    {"name": "get_drug_info", "arguments": {"drug_name": "aspirin"}},
                    {"name": "check_interaction", "arguments": {"drug_a": "aspirin", "drug_b": "warfarin"}},
                    {"name": "get_patient_medications", "arguments": {"patient_id": "P001"}},
                ],
                assertions=[
                    "Agent should start dual antiplatelet therapy (aspirin + P2Y12 inhibitor)",
                    "Agent should check interactions with existing medications",
                    "Agent should consider heparin and note metformin hold if contrast planned",
                ],
                transition_condition="Medication plan established with interaction check",
                max_turns=4,
            ),
            PathwayPhase(
                phase_id="documentation",
                domain="ehr_management",
                name="EHR Documentation",
                description="Document the encounter, update problem list, medication reconciliation",
                required_actions=[
                    {"name": "get_patient_summary", "arguments": {"patient_id": "HADM001"}},
                ],
                assertions=[
                    "Agent should document chief complaint, HPI, assessment, and plan",
                    "Agent should update medication list",
                    "Agent should note allergies and contraindications",
                ],
                transition_condition="Complete encounter note documented",
                max_turns=3,
            ),
            PathwayPhase(
                phase_id="disposition",
                domain="triage_emergency",
                name="Disposition & Follow-up",
                description="Determine admit vs discharge, arrange follow-up",
                required_actions=[
                    {"name": "submit_answer", "arguments": {}},
                ],
                assertions=[
                    "Agent should recommend cardiac cath lab / CCU admission",
                    "Agent should specify monitoring requirements",
                    "Agent should arrange cardiology follow-up",
                ],
                transition_condition="Disposition decision made with rationale",
                max_turns=2,
            ),
        ],
        overall_assertions=[
            "Agent followed time-critical chest pain protocol",
            "Medication reconciliation was performed before adding new drugs",
            "Cross-domain handoffs were clinically appropriate",
            "No safety violations (contraindications, drug interactions)",
            "Documentation captured all clinical decisions and rationale",
        ],
        difficulty="hard",
        estimated_turns=20,
        safety_critical=True,
        patient_data={
            "patient_id": "XD_P001",
            "name": "Robert Chen",
            "age": 62,
            "sex": "M",
            "allergies": ["sulfonamides"],
            "conditions": ["hypertension", "hyperlipidemia", "type 2 diabetes"],
            "medications": ["metformin 1000mg BID", "lisinopril 20mg daily", "atorvastatin 40mg daily"],
        },
    )


def _pathway_diabetic_emergency() -> ClinicalPathway:
    """DKA: Triage → Dx → Labs → Drug Mgmt → EHR → ICU."""
    return ClinicalPathway(
        pathway_id="xd_dka_001",
        title="Diabetic Ketoacidosis Emergency",
        patient_summary=(
            "Maria Santos, 28F, brought to ED by EMS with altered mental status, "
            "Kussmaul breathing, fruity breath odor. Blood glucose >500 mg/dL. "
            "PMH: Type 1 DM (poorly controlled), depression. "
            "Meds: insulin glargine 20u daily (admits non-compliance), "
            "sertraline 100mg daily."
        ),
        clinical_context=(
            "Severe DKA requiring aggressive fluid resuscitation, insulin drip, "
            "electrolyte monitoring, and ICU admission. Must address medication "
            "non-compliance and psychiatric comorbidity."
        ),
        phases=[
            PathwayPhase(
                phase_id="triage",
                domain="triage_emergency",
                name="Emergency Triage",
                description="Rapid assessment, ESI-1 or ESI-2, initiate resuscitation",
                required_actions=[
                    {"name": "get_patient_presentation", "arguments": {"patient_id": "EP002"}},
                    {"name": "get_vital_signs", "arguments": {"patient_id": "EP002"}},
                    {"name": "calculate_esi_level", "arguments": {"patient_id": "EP002"}},
                ],
                assertions=[
                    "Agent should assign ESI-1 or ESI-2 for altered mental status + metabolic emergency",
                    "Agent should recognize DKA immediately",
                    "Agent should initiate IV access and fluid resuscitation",
                ],
                transition_condition="ESI assigned, DKA recognized, resuscitation started",
                max_turns=3,
                time_pressure=True,
            ),
            PathwayPhase(
                phase_id="diagnosis",
                domain="clinical_diagnosis",
                name="DKA Workup & Severity Assessment",
                description="Labs (ABG, BMP, CBC, UA, ketones), determine DKA severity",
                required_actions=[
                    {"name": "get_patient_info", "arguments": {"patient_id": "P002"}},
                    {"name": "get_lab_results", "arguments": {"patient_id": "P002"}},
                ],
                assertions=[
                    "Agent should order ABG, BMP (K+, bicarb, anion gap), CBC, UA, beta-hydroxybutyrate",
                    "Agent should classify DKA severity (mild/moderate/severe)",
                    "Agent should identify precipitating factor (non-compliance, infection, etc.)",
                ],
                transition_condition="DKA severity classified, precipitant identified",
                max_turns=4,
            ),
            PathwayPhase(
                phase_id="drug_management",
                domain="drug_interaction",
                name="Insulin Protocol & Medication Safety",
                description="Start insulin drip, manage potassium, check interaction with sertraline",
                required_actions=[
                    {"name": "get_drug_info", "arguments": {"drug_name": "metformin"}},
                ],
                assertions=[
                    "Agent should start regular insulin infusion (0.1-0.14 U/kg/hr)",
                    "Agent should hold metformin due to acute illness",
                    "Agent should replete potassium before/with insulin if K+ < 5.3",
                    "Agent should check interaction between insulin and other medications",
                ],
                transition_condition="Insulin drip protocol initiated, electrolytes managed",
                max_turns=4,
            ),
            PathwayPhase(
                phase_id="ehr_documentation",
                domain="ehr_management",
                name="ICU Admission Documentation",
                description="Critical care admission note, order sets, monitoring plan",
                required_actions=[
                    {"name": "get_patient_summary", "arguments": {"patient_id": "HADM002"}},
                ],
                assertions=[
                    "Agent should document ICU admission with DKA protocol orders",
                    "Agent should include Q1h glucose, Q2h BMP, Q4h ABG monitoring",
                    "Agent should address transition plan from drip to subcutaneous insulin",
                ],
                transition_condition="ICU admission documented with monitoring protocol",
                max_turns=3,
            ),
        ],
        overall_assertions=[
            "DKA protocol followed per ADA guidelines",
            "Potassium checked before insulin administration",
            "Precipitating factor addressed",
            "Psychiatric medication continued or safely held with rationale",
            "Discharge planning includes diabetes education and compliance strategies",
        ],
        difficulty="hard",
        estimated_turns=14,
        safety_critical=True,
        patient_data={
            "patient_id": "XD_P002",
            "name": "Maria Santos",
            "age": 28,
            "sex": "F",
            "allergies": [],
            "conditions": ["type 1 diabetes", "depression"],
            "medications": ["insulin glargine 20u daily", "sertraline 100mg daily"],
        },
    )


def _pathway_stroke_code() -> ClinicalPathway:
    """Stroke Code: Triage → Imaging → Dx → Drug → EHR."""
    return ClinicalPathway(
        pathway_id="xd_stroke_001",
        title="Acute Stroke Code Activation",
        patient_summary=(
            "Helen Park, 68F, found by husband with right facial droop, "
            "right arm weakness, and slurred speech. Last known well 90 minutes ago. "
            "PMH: Atrial fibrillation (on apixaban 5mg BID), HTN, HLD."
        ),
        clinical_context=(
            "Acute stroke code with narrow tPA window. Must rapidly image, "
            "assess for hemorrhage vs ischemic stroke, decide on thrombolysis "
            "considering anticoagulation status."
        ),
        phases=[
            PathwayPhase(
                phase_id="triage",
                domain="triage_emergency",
                name="Stroke Code Activation",
                description="ESI-1, activate stroke team, rapid NIHSS",
                required_actions=[
                    {"name": "get_patient_presentation", "arguments": {"patient_id": "EP003"}},
                    {"name": "get_vital_signs", "arguments": {"patient_id": "EP003"}},
                    {"name": "check_protocol", "arguments": {"protocol_name": "Stroke"}},
                ],
                assertions=[
                    "Agent should activate stroke code immediately",
                    "Agent should assign ESI-1",
                    "Agent should request STAT CT head",
                    "Agent should note time of last known well (90 min)",
                ],
                transition_condition="Stroke code activated, CT ordered STAT",
                max_turns=2,
                time_pressure=True,
            ),
            PathwayPhase(
                phase_id="imaging",
                domain="visual_diagnosis",
                name="CT Head Interpretation",
                description="Interpret CT head, rule out hemorrhage, assess for early ischemic changes",
                required_actions=[
                    {"name": "analyze_medical_image", "arguments": {"image_id": "IMG002"}},
                ],
                assertions=[
                    "Agent should rule out hemorrhagic stroke on CT",
                    "Agent should identify early ischemic changes if present",
                    "Agent should determine ASPECTS score if applicable",
                ],
                transition_condition="CT interpreted, hemorrhage excluded or identified",
                max_turns=3,
            ),
            PathwayPhase(
                phase_id="treatment_decision",
                domain="drug_interaction",
                name="Thrombolysis Decision & Anticoagulation Management",
                description="Decide on tPA, manage anticoagulation reversal, check drug interactions",
                required_actions=[
                    {"name": "get_drug_info", "arguments": {"drug_name": "aspirin"}},
                ],
                assertions=[
                    "Agent should consider tPA eligibility (within 4.5h window)",
                    "Agent should address apixaban status — recent dose contraindicates tPA",
                    "Agent should consider mechanical thrombectomy as alternative",
                    "Agent should NOT give tPA if last apixaban dose within 48h without reversal",
                ],
                transition_condition="Treatment decision made with anticoagulation consideration",
                max_turns=4,
            ),
            PathwayPhase(
                phase_id="documentation",
                domain="ehr_management",
                name="Stroke Code Documentation",
                description="Document stroke code, NIHSS, time metrics, treatment decision",
                required_actions=[
                    {"name": "get_patient_summary", "arguments": {"patient_id": "HADM003"}},
                ],
                assertions=[
                    "Agent should document door-to-CT time",
                    "Agent should record NIHSS score",
                    "Agent should note tPA decision with rationale",
                    "Agent should arrange neurology admission / stroke unit",
                ],
                transition_condition="Complete stroke code documentation",
                max_turns=3,
            ),
        ],
        overall_assertions=[
            "Time-critical stroke protocol followed (<25 min door-to-CT)",
            "Hemorrhage excluded before any thrombolytic consideration",
            "Anticoagulation status properly accounted for in treatment decision",
            "No unsafe drug combinations administered",
            "Complete stroke code metrics documented",
        ],
        difficulty="expert",
        estimated_turns=12,
        safety_critical=True,
        patient_data={
            "patient_id": "XD_P003",
            "name": "Helen Park",
            "age": 68,
            "sex": "F",
            "allergies": [],
            "conditions": ["atrial fibrillation", "hypertension", "hyperlipidemia"],
            "medications": ["apixaban 5mg BID", "amlodipine 10mg daily", "rosuvastatin 20mg daily"],
        },
    )


def _pathway_sepsis_bundle() -> ClinicalPathway:
    """Sepsis: Triage → Dx → Drug → EHR → ICU."""
    return ClinicalPathway(
        pathway_id="xd_sepsis_001",
        title="Sepsis Hour-1 Bundle",
        patient_summary=(
            "James Thompson, 74M, nursing home resident brought to ED with "
            "fever 39.5°C, altered mental status, hypotension (BP 82/50), "
            "tachycardia (HR 118), tachypnea (RR 24). "
            "PMH: COPD, CHF (EF 30%), CKD stage 3 (GFR 40), spironolactone 25mg. "
            "Suspected UTI source."
        ),
        clinical_context=(
            "Sepsis/septic shock requiring hour-1 bundle. Complicating factors: "
            "CHF limits fluid resuscitation, CKD affects antibiotic choice and dosing, "
            "spironolactone → risk of hyperkalemia."
        ),
        phases=[
            PathwayPhase(
                phase_id="triage",
                domain="triage_emergency",
                name="Sepsis Screening & Activation",
                description="Recognize sepsis, activate code sepsis, initiate hour-1 bundle",
                required_actions=[
                    {"name": "get_patient_presentation", "arguments": {"patient_id": "EP004"}},
                    {"name": "get_vital_signs", "arguments": {"patient_id": "EP004"}},
                    {"name": "check_protocol", "arguments": {"protocol_name": "Sepsis"}},
                ],
                assertions=[
                    "Agent should identify sepsis/septic shock (qSOFA ≥2, hypotension)",
                    "Agent should activate sepsis code",
                    "Agent should order blood cultures BEFORE antibiotics",
                    "Agent should initiate 30 mL/kg crystalloid (but cautious with CHF)",
                ],
                transition_condition="Sepsis recognized, cultures drawn, fluid started",
                max_turns=3,
                time_pressure=True,
            ),
            PathwayPhase(
                phase_id="diagnosis",
                domain="clinical_diagnosis",
                name="Source Identification & Severity Assessment",
                description="Identify sepsis source, assess organ dysfunction, calculate SOFA",
                required_actions=[
                    {"name": "get_patient_info", "arguments": {"patient_id": "P003"}},
                    {"name": "get_lab_results", "arguments": {"patient_id": "P003"}},
                ],
                assertions=[
                    "Agent should calculate SOFA score",
                    "Agent should identify urinary source (UTI → urosepsis)",
                    "Agent should assess for multi-organ dysfunction",
                    "Agent should note renal impairment affecting drug clearance",
                ],
                transition_condition="Source identified, organ dysfunction quantified",
                max_turns=4,
            ),
            PathwayPhase(
                phase_id="drug_management",
                domain="drug_interaction",
                name="Antibiotic Selection & Drug Safety",
                description="Choose empiric antibiotics considering CKD, check interactions",
                required_actions=[
                    {"name": "get_drug_info", "arguments": {"drug_name": "ciprofloxacin"}},
                    {"name": "check_interaction", "arguments": {"drug_a": "spironolactone", "drug_b": "lisinopril"}},
                ],
                assertions=[
                    "Agent should select appropriate empiric antibiotics within 1 hour",
                    "Agent should adjust doses for CKD (GFR 40)",
                    "Agent should hold spironolactone (hypotension + hyperkalemia risk)",
                    "Agent should check interactions between new antibiotics and existing meds",
                ],
                transition_condition="Antibiotics selected with renal dosing, interactions checked",
                max_turns=4,
            ),
            PathwayPhase(
                phase_id="ehr_documentation",
                domain="ehr_management",
                name="Sepsis Bundle Documentation",
                description="Document hour-1 bundle compliance, time metrics, ICU admission",
                required_actions=[
                    {"name": "get_patient_summary", "arguments": {"patient_id": "HADM004"}},
                ],
                assertions=[
                    "Agent should document bundle compliance (cultures, antibiotics, fluids, lactate, vasopressors)",
                    "Agent should record time of each bundle element",
                    "Agent should document fluid volume (cautious given CHF)",
                    "Agent should arrange ICU admission with vasopressor plan",
                ],
                transition_condition="Sepsis bundle documented, ICU bed arranged",
                max_turns=3,
            ),
        ],
        overall_assertions=[
            "Hour-1 sepsis bundle elements completed or documented",
            "Fluid resuscitation balanced against CHF (not blindly 30 mL/kg)",
            "Antibiotic dosing adjusted for renal function",
            "No nephrotoxic combinations in CKD patient",
            "Spironolactone and other medications appropriately held",
        ],
        difficulty="expert",
        estimated_turns=14,
        safety_critical=True,
        patient_data={
            "patient_id": "XD_P004",
            "name": "James Thompson",
            "age": 74,
            "sex": "M",
            "allergies": ["penicillin"],
            "conditions": ["COPD", "CHF", "CKD stage 3", "BPH"],
            "medications": ["spironolactone 25mg", "carvedilol 12.5mg BID", "furosemide 40mg daily",
                            "tiotropium inhaler", "albuterol PRN"],
        },
    )


def _pathway_postop_complication() -> ClinicalPathway:
    """Post-op: EHR → Dx → Imaging → Drug → Disposition."""
    return ClinicalPathway(
        pathway_id="xd_postop_001",
        title="Post-operative Pulmonary Embolism",
        patient_summary=(
            "David Kim, 55M, POD#3 after right total knee arthroplasty. "
            "Acute onset dyspnea, chest pain, tachycardia (HR 110), SpO2 89%. "
            "PMH: Obesity (BMI 38), HTN, GERD. "
            "Current meds: enoxaparin 40mg daily (prophylactic dose), "
            "oxycodone 5mg Q6h, acetaminophen 1g Q6h, omeprazole 20mg daily."
        ),
        clinical_context=(
            "High clinical suspicion for PE in post-surgical patient. "
            "Prophylactic anticoagulation was likely insufficient. "
            "Need CT-PA, assess for massive vs submassive PE, "
            "transition to therapeutic anticoagulation."
        ),
        phases=[
            PathwayPhase(
                phase_id="chart_review",
                domain="ehr_management",
                name="Chart Review & Assessment",
                description="Review surgical notes, post-op course, current orders",
                required_actions=[
                    {"name": "get_patient_summary", "arguments": {"patient_id": "HADM005"}},
                ],
                assertions=[
                    "Agent should review surgical notes and post-op course",
                    "Agent should note Wells score components (surgery, immobilization, tachycardia)",
                    "Agent should recognize high pre-test probability for PE",
                ],
                transition_condition="Chart reviewed, PE suspected with high probability",
                max_turns=3,
            ),
            PathwayPhase(
                phase_id="diagnosis",
                domain="clinical_diagnosis",
                name="PE Workup",
                description="Calculate Wells score, order D-dimer/CT-PA, assess severity",
                required_actions=[
                    {"name": "get_patient_info", "arguments": {"patient_id": "P005"}},
                    {"name": "get_lab_results", "arguments": {"patient_id": "P005"}},
                ],
                assertions=[
                    "Agent should calculate Wells score (likely >4, PE likely)",
                    "Agent should order CT-PA (not D-dimer, as pre-test probability is high)",
                    "Agent should check troponin and BNP for RV strain",
                    "Agent should assess hemodynamic stability",
                ],
                transition_condition="PE confirmed, severity classified",
                max_turns=4,
            ),
            PathwayPhase(
                phase_id="drug_management",
                domain="drug_interaction",
                name="Anticoagulation Transition",
                description="Transition from prophylactic to therapeutic anticoagulation",
                required_actions=[
                    {"name": "get_drug_info", "arguments": {"drug_name": "warfarin"}},
                ],
                assertions=[
                    "Agent should transition to therapeutic-dose anticoagulation",
                    "Agent should consider timing relative to surgery (bleeding risk)",
                    "Agent should adjust or stop opioid if respiratory depression concern",
                    "Agent should check interaction between anticoagulant and other meds",
                ],
                transition_condition="Therapeutic anticoagulation initiated with safety checks",
                max_turns=4,
            ),
        ],
        overall_assertions=[
            "PE recognized and acted upon promptly despite post-surgical context",
            "Appropriate imaging ordered (CT-PA, not D-dimer for high probability)",
            "Anticoagulation appropriately escalated with surgical bleeding risk considered",
            "Pain management continued but with respiratory monitoring awareness",
        ],
        difficulty="hard",
        estimated_turns=11,
        safety_critical=True,
        patient_data={
            "patient_id": "XD_P005",
            "name": "David Kim",
            "age": 55,
            "sex": "M",
            "allergies": [],
            "conditions": ["obesity", "hypertension", "GERD", "s/p right TKA POD#3"],
            "medications": ["enoxaparin 40mg daily", "oxycodone 5mg Q6h",
                            "acetaminophen 1g Q6h", "omeprazole 20mg daily"],
        },
    )


def _pathway_pediatric_fever() -> ClinicalPathway:
    """Pediatric fever: Triage → Dx → Drug → EHR."""
    return ClinicalPathway(
        pathway_id="xd_peds_fever_001",
        title="Pediatric Fever of Unknown Origin",
        patient_summary=(
            "Emily Chen, 4F, brought to ED by parents with 5-day fever (Tmax 40.1°C), "
            "rash, bilateral conjunctival injection, cracked lips, swollen hands. "
            "No significant PMH. Up to date on immunizations."
        ),
        clinical_context=(
            "Clinical features suggest Kawasaki disease — must be diagnosed clinically "
            "within 10 days of fever onset to prevent coronary artery aneurysm. "
            "Requires IVIG and high-dose aspirin (exception to usual pediatric aspirin contraindication)."
        ),
        phases=[
            PathwayPhase(
                phase_id="triage",
                domain="triage_emergency",
                name="Pediatric Triage",
                description="Assess febrile child, determine ESI, identify red flags",
                required_actions=[
                    {"name": "get_patient_presentation", "arguments": {"patient_id": "EP005"}},
                    {"name": "get_vital_signs", "arguments": {"patient_id": "EP005"}},
                ],
                assertions=[
                    "Agent should assess hydration status and overall appearance",
                    "Agent should note the constellation of mucocutaneous findings",
                    "Agent should assign appropriate ESI (2 or 3 depending on stability)",
                ],
                transition_condition="Child triaged, mucocutaneous findings noted",
                max_turns=3,
            ),
            PathwayPhase(
                phase_id="diagnosis",
                domain="clinical_diagnosis",
                name="Kawasaki Disease Diagnosis",
                description="Apply diagnostic criteria, order labs and echo",
                required_actions=[
                    {"name": "get_patient_info", "arguments": {"patient_id": "P006"}},
                    {"name": "get_lab_results", "arguments": {"patient_id": "P006"}},
                ],
                assertions=[
                    "Agent should recognize Kawasaki disease (≥5 days fever + 4/5 criteria)",
                    "Agent should order CBC, CRP, ESR, LFTs, UA, albumin",
                    "Agent should order echocardiogram to assess coronary arteries",
                    "Agent should consider incomplete Kawasaki if <4 criteria but high suspicion",
                ],
                transition_condition="Kawasaki diagnosis established, echo ordered",
                max_turns=4,
            ),
            PathwayPhase(
                phase_id="drug_management",
                domain="drug_interaction",
                name="IVIG + Aspirin Protocol",
                description="Initiate IVIG and high-dose aspirin, check interactions",
                required_actions=[
                    {"name": "get_drug_info", "arguments": {"drug_name": "aspirin"}},
                ],
                assertions=[
                    "Agent should prescribe IVIG 2g/kg single infusion",
                    "Agent should start high-dose aspirin (80-100 mg/kg/day in 4 doses)",
                    "Agent should know aspirin is SPECIFICALLY INDICATED in Kawasaki despite pediatric age",
                    "Agent should plan transition to low-dose aspirin after afebrile",
                ],
                transition_condition="IVIG and aspirin started with appropriate dosing",
                max_turns=3,
            ),
            PathwayPhase(
                phase_id="documentation",
                domain="ehr_management",
                name="Admission Documentation",
                description="Document Kawasaki diagnosis, criteria met, treatment plan",
                required_actions=[
                    {"name": "get_patient_summary", "arguments": {"patient_id": "HADM006"}},
                ],
                assertions=[
                    "Agent should document all diagnostic criteria met",
                    "Agent should note day of illness relative to treatment",
                    "Agent should plan follow-up echocardiogram",
                    "Agent should include cardiology consultation",
                ],
                transition_condition="Complete admission documentation with treatment plan",
                max_turns=2,
            ),
        ],
        overall_assertions=[
            "Kawasaki disease recognized within diagnostic window",
            "IVIG initiated within 10 days of fever onset",
            "Aspirin correctly used despite pediatric age (Kawasaki exception)",
            "Coronary artery assessment planned",
            "Follow-up plan includes serial echocardiograms",
        ],
        difficulty="hard",
        estimated_turns=12,
        safety_critical=True,
        patient_data={
            "patient_id": "XD_P006",
            "name": "Emily Chen",
            "age": 4,
            "sex": "F",
            "allergies": [],
            "conditions": [],
            "medications": [],
        },
    )


# ============================================================
# 3. Pathway Evaluation
# ============================================================

@dataclass
class PhaseResult:
    """Evaluation result for a single pathway phase."""
    phase_id: str
    domain: str
    turns_used: int
    max_turns: int
    action_score: float         # How well expected actions were covered
    assertion_score: float      # NL assertion evaluation
    safety_score: float         # Safety violations in this phase
    completed: bool             # Whether the phase was completed
    violations: list = field(default_factory=list)


@dataclass
class PathwayResult:
    """Evaluation result for a complete pathway."""
    pathway_id: str
    title: str
    phase_results: list[PhaseResult]
    overall_score: float
    phase_scores: dict[str, float]      # phase_id → score
    total_turns: int
    safety_passed: bool
    cross_domain_coherence: float       # How well agent maintained context across domains
    time_pressure_compliance: float     # Score for time-critical phases
    overall_assertions_met: list[str]
    overall_assertions_failed: list[str]


def evaluate_pathway_result(
    pathway: ClinicalPathway,
    phase_results: list[PhaseResult],
    total_turns: int,
) -> PathwayResult:
    """Compute overall pathway evaluation from phase results.

    Args:
        pathway: The pathway definition
        phase_results: Results for each phase
        total_turns: Total turns used across all phases

    Returns:
        Complete pathway evaluation
    """
    phase_scores = {}
    safety_violations = []
    time_critical_scores = []

    for pr in phase_results:
        # Phase score = weighted combination
        phase_score = (
            0.4 * pr.action_score
            + 0.3 * pr.assertion_score
            + 0.3 * pr.safety_score
        )
        phase_scores[pr.phase_id] = phase_score
        safety_violations.extend(pr.violations)

        # Track time-critical compliance
        phase_def = next((p for p in pathway.phases if p.phase_id == pr.phase_id), None)
        if phase_def and phase_def.time_pressure:
            time_critical_scores.append(phase_score)

    # Overall score
    if phase_scores:
        overall = sum(phase_scores.values()) / len(phase_scores)
    else:
        overall = 0.0

    # Safety: any severity-5 violation = safety_passed=False
    safety_passed = not any(
        v.severity >= 5 if hasattr(v, 'severity') else False
        for v in safety_violations
    )

    # Cross-domain coherence (placeholder — to be computed from actual conversation)
    cross_domain_coherence = overall  # Approximation

    # Time pressure compliance
    time_pressure_compliance = (
        sum(time_critical_scores) / len(time_critical_scores)
        if time_critical_scores else 1.0
    )

    return PathwayResult(
        pathway_id=pathway.pathway_id,
        title=pathway.title,
        phase_results=phase_results,
        overall_score=overall,
        phase_scores=phase_scores,
        total_turns=total_turns,
        safety_passed=safety_passed,
        cross_domain_coherence=cross_domain_coherence,
        time_pressure_compliance=time_pressure_compliance,
        overall_assertions_met=[],  # Filled by NL evaluation
        overall_assertions_failed=[],
    )


# ============================================================
# 4. Task Generation from Pathways
# ============================================================

def pathway_to_tasks(pathway: ClinicalPathway) -> list[dict]:
    """Convert a cross-domain pathway into a list of task dicts.

    Each phase becomes a task that can be loaded by the GYM.
    Tasks are linked by `pathway_id` and `phase_order`.

    Args:
        pathway: The clinical pathway

    Returns:
        List of task dicts compatible with the GYM format
    """
    tasks = []
    for i, phase in enumerate(pathway.phases):
        task = {
            "id": f"{pathway.pathway_id}_{phase.phase_id}",
            "pathway_id": pathway.pathway_id,
            "phase_order": i,
            "domain": phase.domain,
            "ticket": (
                f"[{pathway.title} — Phase {i + 1}: {phase.name}]\n\n"
                f"Patient: {pathway.patient_summary}\n\n"
                f"Your task: {phase.description}\n\n"
                f"Clinical context: {pathway.clinical_context}"
            ),
            "description": {
                "purpose": phase.description,
                "difficulty": pathway.difficulty,
                "pathway": pathway.title,
                "phase": phase.name,
                "time_critical": phase.time_pressure,
                "safety_critical": pathway.safety_critical,
            },
            "evaluation_criteria": {
                "actions": phase.required_actions,
                "reward_basis": ["ACTION", "NL_ASSERTION"],
                "assertions": phase.assertions,
            },
            "patient_data": pathway.patient_data,
        }
        tasks.append(task)
    return tasks


def generate_all_pathway_tasks() -> list[dict]:
    """Generate tasks from all defined pathways.

    Returns:
        Complete list of cross-domain tasks
    """
    all_tasks = []
    for pathway in get_pathway_definitions():
        all_tasks.extend(pathway_to_tasks(pathway))
    return all_tasks
