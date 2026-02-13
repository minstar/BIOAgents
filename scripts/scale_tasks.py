#!/usr/bin/env python3
"""Task Scaler: Automatically generate 50+ tasks per domain.

Generates diverse medical task scenarios by:
1. Varying patient demographics (age, gender, comorbidities)
2. Varying disease severity and presentation
3. Combining multiple conditions
4. Creating cross-domain scenarios
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────
#  Clinical Diagnosis Task Templates
# ─────────────────────────────────────────────────────────────

CLINICAL_CONDITIONS = [
    {"name": "pneumonia", "symptoms": ["cough", "fever", "dyspnea", "chest_pain"], "key_tests": ["chest_xray", "CBC", "sputum_culture", "CRP"], "severity": ["mild", "moderate", "severe"]},
    {"name": "meningitis", "symptoms": ["headache", "neck_stiffness", "fever", "photophobia"], "key_tests": ["lumbar_puncture", "blood_culture", "CBC", "CT_head"], "severity": ["moderate", "severe"]},
    {"name": "appendicitis", "symptoms": ["RLQ_pain", "nausea", "fever", "anorexia"], "key_tests": ["CT_abdomen", "CBC", "CRP", "urinalysis"], "severity": ["uncomplicated", "complicated"]},
    {"name": "myocardial_infarction", "symptoms": ["chest_pain", "diaphoresis", "dyspnea", "nausea"], "key_tests": ["ECG", "troponin", "CBC", "BMP"], "severity": ["STEMI", "NSTEMI"]},
    {"name": "pulmonary_embolism", "symptoms": ["dyspnea", "pleuritic_chest_pain", "tachycardia", "hemoptysis"], "key_tests": ["CT_angiography", "D-dimer", "ABG", "ECG"], "severity": ["submassive", "massive"]},
    {"name": "diabetic_ketoacidosis", "symptoms": ["polyuria", "polydipsia", "abdominal_pain", "Kussmaul_breathing"], "key_tests": ["BMP", "ABG", "serum_ketones", "urinalysis"], "severity": ["mild", "moderate", "severe"]},
    {"name": "stroke", "symptoms": ["hemiparesis", "aphasia", "facial_droop", "vision_loss"], "key_tests": ["CT_head", "MRI_brain", "carotid_doppler", "ECG"], "severity": ["TIA", "ischemic", "hemorrhagic"]},
    {"name": "sepsis", "symptoms": ["fever", "tachycardia", "hypotension", "altered_mental_status"], "key_tests": ["blood_culture", "lactate", "CBC", "procalcitonin"], "severity": ["sepsis", "severe_sepsis", "septic_shock"]},
    {"name": "heart_failure", "symptoms": ["dyspnea", "orthopnea", "edema", "fatigue"], "key_tests": ["BNP", "echocardiogram", "chest_xray", "BMP"], "severity": ["NYHA_II", "NYHA_III", "NYHA_IV"]},
    {"name": "pancreatitis", "symptoms": ["epigastric_pain", "nausea", "vomiting", "fever"], "key_tests": ["lipase", "amylase", "CT_abdomen", "CBC"], "severity": ["mild", "moderately_severe", "severe"]},
    {"name": "urinary_tract_infection", "symptoms": ["dysuria", "frequency", "urgency", "suprapubic_pain"], "key_tests": ["urinalysis", "urine_culture", "CBC", "BMP"], "severity": ["uncomplicated", "complicated", "pyelonephritis"]},
    {"name": "COPD_exacerbation", "symptoms": ["dyspnea", "increased_sputum", "wheezing", "cough"], "key_tests": ["ABG", "chest_xray", "CBC", "spirometry"], "severity": ["mild", "moderate", "severe"]},
    {"name": "gastrointestinal_bleeding", "symptoms": ["hematemesis", "melena", "tachycardia", "lightheadedness"], "key_tests": ["CBC", "BMP", "coagulation", "upper_endoscopy"], "severity": ["minor", "major", "massive"]},
    {"name": "acute_kidney_injury", "symptoms": ["oliguria", "edema", "nausea", "fatigue"], "key_tests": ["BMP", "urinalysis", "renal_ultrasound", "CBC"], "severity": ["stage_1", "stage_2", "stage_3"]},
    {"name": "asthma_exacerbation", "symptoms": ["wheezing", "dyspnea", "chest_tightness", "cough"], "key_tests": ["peak_flow", "ABG", "chest_xray", "CBC"], "severity": ["mild", "moderate", "severe", "life_threatening"]},
]

PATIENT_TEMPLATES = [
    {"age_range": (18, 35), "gender": "M", "comorbidities": [], "prefix": "young_male"},
    {"age_range": (18, 35), "gender": "F", "comorbidities": [], "prefix": "young_female"},
    {"age_range": (45, 65), "gender": "M", "comorbidities": ["hypertension", "diabetes"], "prefix": "middle_age_male"},
    {"age_range": (45, 65), "gender": "F", "comorbidities": ["obesity", "hypothyroidism"], "prefix": "middle_age_female"},
    {"age_range": (65, 85), "gender": "M", "comorbidities": ["COPD", "atrial_fibrillation", "CKD"], "prefix": "elderly_male"},
    {"age_range": (65, 85), "gender": "F", "comorbidities": ["osteoporosis", "hypertension", "heart_failure"], "prefix": "elderly_female"},
    {"age_range": (25, 45), "gender": "F", "comorbidities": ["pregnancy"], "prefix": "pregnant"},
    {"age_range": (2, 12), "gender": "M", "comorbidities": [], "prefix": "pediatric"},
]


def generate_clinical_tasks(target_count: int = 60) -> list[dict]:
    """Generate clinical diagnosis tasks."""
    tasks = []
    task_idx = 0

    for condition in CLINICAL_CONDITIONS:
        for severity in condition["severity"]:
            for patient in random.sample(PATIENT_TEMPLATES, min(3, len(PATIENT_TEMPLATES))):
                if task_idx >= target_count:
                    break

                task_idx += 1
                age = random.randint(*patient["age_range"])
                gender_str = "male" if patient["gender"] == "M" else "female"

                # Build symptom presentation
                num_symptoms = random.randint(2, len(condition["symptoms"]))
                presented_symptoms = random.sample(condition["symptoms"], num_symptoms)

                comorbidity_str = ""
                if patient["comorbidities"]:
                    comorbidity_str = f" Past medical history includes {', '.join(patient['comorbidities'])}."

                ticket = (
                    f"A {age}-year-old {gender_str} presents to the emergency department with "
                    f"{', '.join(s.replace('_', ' ') for s in presented_symptoms[:-1])} "
                    f"and {presented_symptoms[-1].replace('_', ' ')}. "
                    f"Symptoms began {random.choice(['2 hours', '6 hours', '1 day', '3 days', '1 week'])} ago."
                    f"{comorbidity_str} "
                    f"The presentation appears to be {severity.replace('_', ' ')} in nature."
                )

                # Expected actions
                expected_actions = [
                    {"name": "get_patient_info", "arguments": {}, "compare_args": []},
                    {"name": "get_vital_signs", "arguments": {}, "compare_args": []},
                ]
                for test in condition["key_tests"][:3]:
                    expected_actions.append(
                        {"name": "order_lab_test", "arguments": {"test_name": test}, "compare_args": []}
                    )

                task = {
                    "id": f"dx_{condition['name']}_{severity}_{patient['prefix']}_{task_idx:03d}",
                    "ticket": ticket,
                    "description": {
                        "condition": condition["name"],
                        "severity": severity,
                        "patient_type": patient["prefix"],
                    },
                    "evaluation_criteria": {
                        "actions": expected_actions,
                        "reward_basis": ["ACTION", "NL_ASSERTION"],
                        "assertions": [
                            f"Agent should consider {condition['name']} in differential",
                            f"Agent should order appropriate diagnostic tests",
                            f"Agent should account for severity level: {severity}",
                        ],
                    },
                }
                tasks.append(task)

            if task_idx >= target_count:
                break
        if task_idx >= target_count:
            break

    return tasks


# ─────────────────────────────────────────────────────────────
#  Drug Interaction Task Templates
# ─────────────────────────────────────────────────────────────

DRUG_SCENARIOS = [
    {"drugs": ["warfarin", "aspirin"], "interaction": "bleeding_risk", "severity": "high"},
    {"drugs": ["metformin", "contrast_dye"], "interaction": "lactic_acidosis", "severity": "high"},
    {"drugs": ["fluoxetine", "tramadol"], "interaction": "serotonin_syndrome", "severity": "high"},
    {"drugs": ["simvastatin", "amiodarone"], "interaction": "myopathy_risk", "severity": "moderate"},
    {"drugs": ["lisinopril", "spironolactone"], "interaction": "hyperkalemia", "severity": "moderate"},
    {"drugs": ["clopidogrel", "omeprazole"], "interaction": "reduced_efficacy", "severity": "moderate"},
    {"drugs": ["phenytoin", "warfarin"], "interaction": "variable_anticoagulation", "severity": "high"},
    {"drugs": ["methotrexate", "NSAIDs"], "interaction": "methotrexate_toxicity", "severity": "high"},
    {"drugs": ["digoxin", "amiodarone"], "interaction": "digoxin_toxicity", "severity": "high"},
    {"drugs": ["lithium", "NSAIDs"], "interaction": "lithium_toxicity", "severity": "high"},
    {"drugs": ["ciprofloxacin", "theophylline"], "interaction": "theophylline_toxicity", "severity": "moderate"},
    {"drugs": ["rifampin", "oral_contraceptives"], "interaction": "reduced_efficacy", "severity": "moderate"},
    {"drugs": ["ACE_inhibitor", "potassium_supplements"], "interaction": "hyperkalemia", "severity": "moderate"},
    {"drugs": ["SSRIs", "MAOIs"], "interaction": "serotonin_syndrome", "severity": "severe"},
    {"drugs": ["grapefruit", "statins"], "interaction": "increased_statin_levels", "severity": "mild"},
]


def generate_drug_interaction_tasks(target_count: int = 50) -> list[dict]:
    """Generate drug interaction tasks."""
    tasks = []
    task_idx = 0

    scenarios_extended = []
    for scenario in DRUG_SCENARIOS:
        # Add polypharmacy variants
        for extra_drugs in [
            [],
            ["metformin"],
            ["lisinopril", "amlodipine"],
            ["levothyroxine"],
            ["atorvastatin", "metoprolol"],
        ]:
            new_scenario = {**scenario, "extra_meds": extra_drugs}
            scenarios_extended.append(new_scenario)

    for scenario in scenarios_extended:
        if task_idx >= target_count:
            break

        task_idx += 1
        all_drugs = scenario["drugs"] + scenario.get("extra_meds", [])

        patient = random.choice(PATIENT_TEMPLATES)
        age = random.randint(*patient["age_range"])
        gender_str = "male" if patient["gender"] == "M" else "female"

        ticket = (
            f"A {age}-year-old {gender_str} is currently taking: {', '.join(all_drugs)}. "
            f"The physician wants to add {scenario['drugs'][-1]} to the regimen. "
            f"Please review for potential drug-drug interactions and provide recommendations."
        )

        task = {
            "id": f"di_{scenario['interaction']}_{task_idx:03d}",
            "ticket": ticket,
            "description": {
                "interaction_type": scenario["interaction"],
                "severity": scenario["severity"],
                "drugs": scenario["drugs"],
            },
            "evaluation_criteria": {
                "actions": [
                    {"name": "get_drug_info", "arguments": {}, "compare_args": []},
                    {"name": "check_interaction", "arguments": {}, "compare_args": []},
                ],
                "reward_basis": ["ACTION", "NL_ASSERTION"],
                "assertions": [
                    f"Agent should identify {scenario['interaction']} interaction",
                    f"Agent should assess severity as {scenario['severity']}",
                    "Agent should provide alternative recommendations",
                ],
            },
        }
        tasks.append(task)

    return tasks


# ─────────────────────────────────────────────────────────────
#  Visual Diagnosis Task Templates
# ─────────────────────────────────────────────────────────────

IMAGING_SCENARIOS = [
    {"modality": "chest_xray", "finding": "pneumothorax", "conditions": ["tension", "simple", "small"]},
    {"modality": "chest_xray", "finding": "cardiomegaly", "conditions": ["mild", "moderate", "severe"]},
    {"modality": "chest_xray", "finding": "pleural_effusion", "conditions": ["unilateral", "bilateral"]},
    {"modality": "CT_head", "finding": "intracranial_hemorrhage", "conditions": ["subdural", "epidural", "subarachnoid"]},
    {"modality": "CT_abdomen", "finding": "bowel_obstruction", "conditions": ["partial", "complete"]},
    {"modality": "MRI_brain", "finding": "demyelination", "conditions": ["multiple_sclerosis", "ADEM"]},
    {"modality": "mammography", "finding": "breast_mass", "conditions": ["benign", "suspicious", "malignant"]},
    {"modality": "fundoscopy", "finding": "diabetic_retinopathy", "conditions": ["NPDR_mild", "NPDR_moderate", "PDR"]},
    {"modality": "dermoscopy", "finding": "pigmented_lesion", "conditions": ["benign_nevus", "atypical", "melanoma"]},
    {"modality": "echocardiogram", "finding": "valvular_disease", "conditions": ["mitral_regurgitation", "aortic_stenosis"]},
    {"modality": "CT_chest", "finding": "lung_nodule", "conditions": ["ground_glass", "solid", "part_solid"]},
    {"modality": "ultrasound", "finding": "gallstones", "conditions": ["asymptomatic", "cholecystitis"]},
]


def generate_visual_diagnosis_tasks(target_count: int = 50) -> list[dict]:
    """Generate visual diagnosis tasks."""
    tasks = []
    task_idx = 0

    for scenario in IMAGING_SCENARIOS:
        for condition in scenario["conditions"]:
            if task_idx >= target_count:
                break

            task_idx += 1
            patient = random.choice(PATIENT_TEMPLATES)
            age = random.randint(*patient["age_range"])
            gender_str = "male" if patient["gender"] == "M" else "female"

            ticket = (
                f"A {age}-year-old {gender_str} presents for {scenario['modality'].replace('_', ' ')} evaluation. "
                f"Clinical concern: {scenario['finding'].replace('_', ' ')}. "
                f"Image ID: IMG_{task_idx:03d}. "
                f"Please analyze the image and provide your diagnostic assessment."
            )

            task = {
                "id": f"vdx_{scenario['modality']}_{condition}_{task_idx:03d}",
                "ticket": ticket,
                "description": {
                    "modality": scenario["modality"],
                    "finding": scenario["finding"],
                    "condition": condition,
                },
                "evaluation_criteria": {
                    "actions": [
                        {"name": "analyze_medical_image", "arguments": {"image_id": f"IMG_{task_idx:03d}"}, "compare_args": ["image_id"]},
                    ],
                    "reward_basis": ["ACTION", "NL_ASSERTION"],
                    "assertions": [
                        f"Agent should identify {scenario['finding'].replace('_', ' ')}",
                        f"Agent should characterize the condition as {condition.replace('_', ' ')}",
                    ],
                },
            }
            tasks.append(task)

        if task_idx >= target_count:
            break

    return tasks


# ─────────────────────────────────────────────────────────────
#  EHR Management Task Templates
# ─────────────────────────────────────────────────────────────

EHR_SCENARIOS = [
    {"type": "chart_review", "focus": "admission_assessment"},
    {"type": "chart_review", "focus": "pre_operative"},
    {"type": "medication_reconciliation", "focus": "admission"},
    {"type": "medication_reconciliation", "focus": "discharge"},
    {"type": "critical_value", "focus": "lab_alert"},
    {"type": "critical_value", "focus": "vital_alert"},
    {"type": "readmission_risk", "focus": "30_day"},
    {"type": "clinical_scoring", "focus": "SOFA"},
    {"type": "clinical_scoring", "focus": "NEWS2"},
    {"type": "clinical_scoring", "focus": "GRACE"},
    {"type": "discharge_planning", "focus": "complex"},
    {"type": "antibiotic_stewardship", "focus": "de_escalation"},
    {"type": "icu_assessment", "focus": "daily_rounds"},
    {"type": "quality_measure", "focus": "core_measures"},
    {"type": "longitudinal_analysis", "focus": "trend_review"},
]


def generate_ehr_tasks(target_count: int = 50) -> list[dict]:
    """Generate EHR management tasks."""
    tasks = []
    task_idx = 0

    hadm_ids = ["HADM_10001", "HADM_9001", "HADM_10002", "HADM_10003"]

    for scenario in EHR_SCENARIOS:
        for hadm_id in hadm_ids:
            if task_idx >= target_count:
                break

            task_idx += 1

            if scenario["type"] == "chart_review":
                ticket = (
                    f"Please perform a comprehensive chart review for admission {hadm_id}. "
                    f"Focus: {scenario['focus'].replace('_', ' ')}. "
                    f"Review labs, vitals, medications, and clinical scores."
                )
            elif scenario["type"] == "medication_reconciliation":
                ticket = (
                    f"Perform medication reconciliation for {hadm_id} at {scenario['focus']}. "
                    f"Check for drug interactions, appropriate dosing, and missing medications."
                )
            elif scenario["type"] == "critical_value":
                ticket = (
                    f"Review {hadm_id} for critical values ({scenario['focus'].replace('_', ' ')}). "
                    f"Identify any concerning trends and recommend immediate actions."
                )
            elif scenario["type"] == "clinical_scoring":
                ticket = (
                    f"Calculate and interpret the {scenario['focus']} score for {hadm_id}. "
                    f"Provide recommendations based on the score."
                )
            else:
                ticket = (
                    f"Perform {scenario['type'].replace('_', ' ')} for {hadm_id}. "
                    f"Focus area: {scenario['focus'].replace('_', ' ')}."
                )

            task = {
                "id": f"ehr_{scenario['type']}_{scenario['focus']}_{task_idx:03d}",
                "ticket": ticket,
                "description": {
                    "task_type": scenario["type"],
                    "focus": scenario["focus"],
                    "hadm_id": hadm_id,
                },
                "evaluation_criteria": {
                    "actions": [
                        {"name": "get_patient_summary", "arguments": {"hadm_id": hadm_id}, "compare_args": ["hadm_id"]},
                    ],
                    "reward_basis": ["ACTION"],
                },
            }
            tasks.append(task)

        if task_idx >= target_count:
            break

    return tasks


# ─────────────────────────────────────────────────────────────
#  Triage & Emergency Task Templates
# ─────────────────────────────────────────────────────────────

TRIAGE_COMPLAINTS = [
    {"complaint": "chest_pain", "esi_range": [1, 2, 3], "protocols": ["STEMI", "ACS"]},
    {"complaint": "shortness_of_breath", "esi_range": [1, 2, 3], "protocols": ["PE", "asthma"]},
    {"complaint": "altered_mental_status", "esi_range": [1, 2], "protocols": ["stroke", "sepsis"]},
    {"complaint": "abdominal_pain", "esi_range": [2, 3, 4], "protocols": []},
    {"complaint": "headache", "esi_range": [2, 3, 4], "protocols": ["stroke"]},
    {"complaint": "seizure", "esi_range": [1, 2], "protocols": []},
    {"complaint": "trauma_fall", "esi_range": [1, 2, 3], "protocols": ["trauma"]},
    {"complaint": "overdose", "esi_range": [1, 2], "protocols": []},
    {"complaint": "allergic_reaction", "esi_range": [1, 2, 3], "protocols": ["anaphylaxis"]},
    {"complaint": "back_pain", "esi_range": [3, 4], "protocols": []},
    {"complaint": "fever_immunocompromised", "esi_range": [2, 3], "protocols": ["sepsis"]},
    {"complaint": "laceration", "esi_range": [3, 4, 5], "protocols": []},
    {"complaint": "eye_injury", "esi_range": [2, 3, 4], "protocols": []},
    {"complaint": "syncope", "esi_range": [2, 3], "protocols": []},
    {"complaint": "gi_bleeding", "esi_range": [1, 2, 3], "protocols": []},
]


def generate_triage_tasks(target_count: int = 60) -> list[dict]:
    """Generate triage & emergency tasks."""
    tasks = []
    task_idx = 0

    for complaint_info in TRIAGE_COMPLAINTS:
        for esi in complaint_info["esi_range"]:
            for patient in random.sample(PATIENT_TEMPLATES, min(2, len(PATIENT_TEMPLATES))):
                if task_idx >= target_count:
                    break

                task_idx += 1
                age = random.randint(*patient["age_range"])
                gender_str = "male" if patient["gender"] == "M" else "female"
                complaint = complaint_info["complaint"].replace("_", " ")

                vitals_text = _random_vitals_text(esi)

                ticket = (
                    f"A {age}-year-old {gender_str} presents to the ED with {complaint}. "
                    f"Onset: {random.choice(['10 minutes', '1 hour', '3 hours', '12 hours', '2 days'])} ago. "
                    f"Vitals: {vitals_text}. "
                )

                if patient["comorbidities"]:
                    ticket += f"PMH: {', '.join(patient['comorbidities'])}. "

                actions = [
                    {"name": "get_patient_presentation", "arguments": {}, "compare_args": []},
                    {"name": "get_vital_signs", "arguments": {}, "compare_args": []},
                    {"name": "assess_airway_breathing", "arguments": {}, "compare_args": []},
                    {"name": "calculate_esi_level", "arguments": {}, "compare_args": []},
                ]

                if complaint_info["protocols"]:
                    actions.append({"name": "check_protocol", "arguments": {}, "compare_args": []})

                actions.append({"name": "submit_answer", "arguments": {"esi_level": esi}, "compare_args": ["esi_level"]})

                task = {
                    "id": f"triage_{complaint_info['complaint']}_esi{esi}_{patient['prefix']}_{task_idx:03d}",
                    "ticket": ticket,
                    "correct_answer": f"ESI {esi}",
                    "description": {
                        "complaint": complaint_info["complaint"],
                        "esi_level": esi,
                        "patient_type": patient["prefix"],
                    },
                    "evaluation_criteria": {
                        "actions": actions,
                        "nl_assertions": [
                            f"Agent should assign ESI level {esi}",
                            f"Agent should identify {complaint} as the chief complaint",
                            "Agent should assess ABC stability",
                        ],
                        "reward_basis": ["ACTION", "NL_ASSERTION"],
                    },
                }
                tasks.append(task)

            if task_idx >= target_count:
                break
        if task_idx >= target_count:
            break

    return tasks


def _random_vitals_text(esi: int) -> str:
    """Generate random vital signs text appropriate for ESI level."""
    if esi == 1:
        hr = random.randint(120, 160)
        sbp = random.randint(60, 85)
        rr = random.randint(28, 40)
        spo2 = random.randint(75, 88)
    elif esi == 2:
        hr = random.randint(100, 130)
        sbp = random.randint(85, 110)
        rr = random.randint(22, 30)
        spo2 = random.randint(88, 94)
    elif esi == 3:
        hr = random.randint(80, 110)
        sbp = random.randint(110, 140)
        rr = random.randint(16, 24)
        spo2 = random.randint(94, 98)
    else:
        hr = random.randint(65, 90)
        sbp = random.randint(115, 140)
        rr = random.randint(14, 20)
        spo2 = random.randint(97, 100)

    temp = round(random.uniform(36.5, 39.5 if esi <= 2 else 38.0), 1)
    return f"HR {hr}, BP {sbp}/{random.randint(50, 90)}, RR {rr}, SpO2 {spo2}%, Temp {temp}C"


# ─────────────────────────────────────────────────────────────
#  Radiology Report Task Templates
# ─────────────────────────────────────────────────────────────

RADIOLOGY_SCENARIOS = [
    {"modality": "XR", "body_part": "chest", "findings": ["pneumonia", "pleural_effusion", "pneumothorax", "cardiomegaly", "normal"]},
    {"modality": "CT", "body_part": "head", "findings": ["ischemic_stroke", "hemorrhagic_stroke", "subdural_hematoma", "normal"]},
    {"modality": "CT", "body_part": "abdomen", "findings": ["appendicitis", "bowel_obstruction", "renal_stone", "pancreatitis"]},
    {"modality": "CT", "body_part": "chest", "findings": ["pulmonary_embolism", "lung_mass", "mediastinal_mass"]},
    {"modality": "MRI", "body_part": "brain", "findings": ["brain_tumor", "multiple_sclerosis", "abscess"]},
    {"modality": "MRI", "body_part": "spine", "findings": ["disc_herniation", "spinal_stenosis", "cord_compression"]},
    {"modality": "US", "body_part": "abdomen", "findings": ["cholecystitis", "liver_mass", "aortic_aneurysm"]},
    {"modality": "MG", "body_part": "breast", "findings": ["mass_suspicious", "calcifications", "normal"]},
]


def generate_radiology_report_tasks(target_count: int = 50) -> list[dict]:
    """Generate radiology report generation tasks."""
    tasks = []
    task_idx = 0

    for scenario in RADIOLOGY_SCENARIOS:
        for finding in scenario["findings"]:
            if task_idx >= target_count:
                break

            task_idx += 1
            patient = random.choice(PATIENT_TEMPLATES)
            age = random.randint(*patient["age_range"])
            gender_str = "male" if patient["gender"] == "M" else "female"

            modality_name = {
                "XR": "X-ray", "CT": "CT scan", "MRI": "MRI",
                "US": "Ultrasound", "MG": "Mammogram",
            }.get(scenario["modality"], scenario["modality"])

            ticket = (
                f"{modality_name} {scenario['body_part']} for a {age}-year-old {gender_str}. "
                f"Study ID: RAD_GEN_{task_idx:03d}. "
                f"Generate a complete structured radiology report."
            )

            task = {
                "id": f"rad_{scenario['modality'].lower()}_{finding}_{task_idx:03d}",
                "ticket": ticket,
                "study_id": f"RAD_GEN_{task_idx:03d}",
                "description": {
                    "modality": scenario["modality"],
                    "body_part": scenario["body_part"],
                    "finding": finding,
                },
                "evaluation_criteria": {
                    "actions": [
                        {"name": "get_study_info", "arguments": {}, "compare_args": []},
                        {"name": "get_clinical_history", "arguments": {}, "compare_args": []},
                        {"name": "analyze_findings", "arguments": {}, "compare_args": []},
                        {"name": "get_report_template", "arguments": {}, "compare_args": []},
                    ],
                    "nl_assertions": [
                        f"Report should identify {finding.replace('_', ' ')}",
                        "Report follows structured format (indication, technique, findings, impression)",
                        "Report includes comparison with prior if available",
                    ],
                    "reward_basis": ["ACTION", "NL_ASSERTION"],
                },
            }
            tasks.append(task)

        if task_idx >= target_count:
            break

    return tasks


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scale up tasks per domain")
    parser.add_argument("--target", type=int, default=50, help="Target tasks per domain")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: data/domains/{domain})")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    generators = {
        "clinical_diagnosis": generate_clinical_tasks,
        "drug_interaction": generate_drug_interaction_tasks,
        "visual_diagnosis": generate_visual_diagnosis_tasks,
        "ehr_management": generate_ehr_tasks,
        "triage_emergency": generate_triage_tasks,
        "radiology_report": generate_radiology_report_tasks,
    }

    total = 0
    for domain, gen_fn in generators.items():
        tasks = gen_fn(target_count=args.target)

        # Split: 70% train, 30% test
        random.shuffle(tasks)
        split_idx = int(len(tasks) * 0.7)
        train_tasks = tasks[:split_idx]
        test_tasks = tasks[split_idx:]

        # Save
        if args.output_dir:
            out_dir = Path(args.output_dir) / domain
        else:
            out_dir = PROJECT_ROOT / "data" / "domains" / domain

        out_dir.mkdir(parents=True, exist_ok=True)

        # Save scaled tasks (don't overwrite originals)
        tasks_path = out_dir / "tasks_scaled.json"
        with open(tasks_path, "w") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        split_path = out_dir / "split_tasks_scaled.json"
        with open(split_path, "w") as f:
            json.dump(
                {"train": [t["id"] for t in train_tasks], "test": [t["id"] for t in test_tasks]},
                f,
                indent=2,
            )

        total += len(tasks)
        logger.info(f"  {domain}: {len(tasks)} tasks ({len(train_tasks)} train / {len(test_tasks)} test) → {tasks_path}")

    logger.info(f"\nTotal: {total} tasks across {len(generators)} domains")


if __name__ == "__main__":
    main()
