"""Clinical Guidelines Knowledge Base for BIOAgents Healthcare AI GYM.

Provides structured access to evidence-based clinical practice guidelines.
Used for:
1. Agent training — guidelines as part of the policy/context
2. Evaluation — check if agent actions comply with guidelines
3. Reward signal — guideline compliance as a reward component

Reference guidelines included:
- AHA/ACC STEMI (2023)
- AHA/ASA Acute Ischemic Stroke (2019)
- Surviving Sepsis Campaign (2021)
- ADA DKA (2024)
- ACEP HEART Score (2020)
- AHA Kawasaki Disease (2017/2024)
- WSES Appendicitis (2020)
- ESC Pulmonary Embolism (2019)
- ACP Acute Low Back Pain (2017)
- IDSA Antimicrobial Stewardship (2023)
"""

import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GUIDELINES_PATH = PROJECT_ROOT / "data" / "guidelines" / "clinical_guidelines.json"


# ============================================================
# 1. Guidelines Database (embedded for reliability)
# ============================================================

_GUIDELINES_DB: dict = {
    "stemi": {
        "guideline_id": "aha_stemi_2023",
        "organization": "AHA/ACC",
        "title": "2023 AHA/ACC Guidelines for Management of ST-Elevation MI",
        "condition": "STEMI",
        "year": 2023,
        "category": "cardiology",
        "key_recommendations": [
            {"id": "R1", "class": "I", "level": "A",
             "text": "Perform primary PCI within 90 minutes of first medical contact (door-to-balloon)",
             "time_target": "90 minutes", "critical": True},
            {"id": "R2", "class": "I", "level": "A",
             "text": "Administer aspirin 162-325mg as soon as possible, chewed",
             "medications": ["aspirin"]},
            {"id": "R3", "class": "I", "level": "A",
             "text": "Administer P2Y12 inhibitor (ticagrelor or prasugrel preferred over clopidogrel) as loading dose",
             "medications": ["ticagrelor", "prasugrel", "clopidogrel"]},
            {"id": "R4", "class": "I", "level": "A",
             "text": "Anticoagulation with unfractionated heparin during PCI",
             "medications": ["heparin"]},
            {"id": "R5", "class": "I", "level": "B-R",
             "text": "Obtain 12-lead ECG within 10 minutes of ED arrival",
             "time_target": "10 minutes", "critical": True},
            {"id": "R6", "class": "I", "level": "C-LD",
             "text": "If PCI cannot be performed within 120 min, administer fibrinolytic therapy within 30 min",
             "time_target": "30 minutes for fibrinolysis if no PCI"},
            {"id": "R7", "class": "III", "level": "B-R",
             "text": "Do NOT administer fibrinolytics if symptom onset >12 hours without ongoing ischemia",
             "contraindication": True},
        ],
        "contraindications_fibrinolysis": [
            "Active internal bleeding", "History of hemorrhagic stroke",
            "Intracranial neoplasm", "Suspected aortic dissection",
            "Significant head/facial trauma within 3 months",
        ],
        "quality_metrics": [
            "Door-to-ECG time ≤10 min",
            "Door-to-balloon time ≤90 min",
            "Aspirin at arrival",
            "Statin prescribed at discharge",
            "DAPT prescribed at discharge",
        ],
    },
    "acute_ischemic_stroke": {
        "guideline_id": "aha_stroke_2019",
        "organization": "AHA/ASA",
        "title": "2019 AHA/ASA Guidelines for Early Management of Acute Ischemic Stroke",
        "condition": "Acute Ischemic Stroke",
        "year": 2019,
        "category": "neurology",
        "key_recommendations": [
            {"id": "R1", "class": "I", "level": "A",
             "text": "Non-contrast CT head within 25 minutes of ED arrival (door-to-CT)",
             "time_target": "25 minutes", "critical": True},
            {"id": "R2", "class": "I", "level": "A",
             "text": "IV alteplase (0.9 mg/kg, max 90mg) for eligible patients within 4.5 hours of symptom onset",
             "time_target": "4.5 hours from onset", "medications": ["alteplase"]},
            {"id": "R3", "class": "I", "level": "A",
             "text": "Door-to-needle time ≤60 minutes for tPA administration",
             "time_target": "60 minutes", "critical": True},
            {"id": "R4", "class": "I", "level": "A",
             "text": "Mechanical thrombectomy for LVO within 24 hours if ASPECTS ≥6 and favorable perfusion imaging",
             "time_target": "6-24 hours with perfusion mismatch"},
            {"id": "R5", "class": "I", "level": "B-R",
             "text": "BP management: permissive hypertension <220/120 if no tPA; <185/110 before and <180/105 after tPA"},
            {"id": "R6", "class": "III", "level": "B-R",
             "text": "Do NOT give tPA if INR >1.7 or on DOAC with last dose <48h (unless specific reversal available)",
             "contraindication": True},
            {"id": "R7", "class": "I", "level": "B-R",
             "text": "Assess NIHSS score for stroke severity and guide treatment decisions"},
        ],
        "tpa_exclusion_criteria": [
            "Symptom onset >4.5 hours", "Active bleeding", "Plt <100,000",
            "INR >1.7", "DOAC within 48h", "Major surgery within 14 days",
            "GI/urinary bleeding within 21 days", "BP >185/110 despite treatment",
        ],
        "quality_metrics": [
            "Door-to-CT ≤25 min", "Door-to-needle ≤60 min",
            "Dysphagia screen before PO", "DVT prophylaxis within 48h",
            "Antithrombotic at discharge", "Statin at discharge",
        ],
    },
    "sepsis": {
        "guideline_id": "ssc_2021",
        "organization": "Surviving Sepsis Campaign",
        "title": "2021 Surviving Sepsis Campaign Guidelines",
        "condition": "Sepsis and Septic Shock",
        "year": 2021,
        "category": "critical_care",
        "key_recommendations": [
            {"id": "R1", "class": "strong", "level": "moderate",
             "text": "Obtain blood cultures BEFORE initiating antibiotics (do not delay antibiotics >45 min for cultures)",
             "time_target": "Before antibiotics, but don't delay >45 min", "critical": True},
            {"id": "R2", "class": "strong", "level": "moderate",
             "text": "Administer broad-spectrum IV antibiotics within 1 hour of sepsis recognition",
             "time_target": "1 hour", "critical": True},
            {"id": "R3", "class": "strong", "level": "low",
             "text": "Administer ≥30 mL/kg IV crystalloid within first 3 hours for sepsis-induced hypoperfusion",
             "time_target": "3 hours"},
            {"id": "R4", "class": "strong", "level": "moderate",
             "text": "Use norepinephrine as first-line vasopressor for septic shock",
             "medications": ["norepinephrine"]},
            {"id": "R5", "class": "strong", "level": "moderate",
             "text": "Target MAP ≥65 mmHg in septic shock"},
            {"id": "R6", "class": "strong", "level": "low",
             "text": "Measure serum lactate; if elevated (>2 mmol/L), remeasure within 2-4 hours to guide resuscitation"},
            {"id": "R7", "class": "weak", "level": "low",
             "text": "Use dynamic measures (passive leg raise, fluid challenge) rather than static measures to guide fluid resuscitation"},
            {"id": "R8", "class": "strong", "level": "moderate",
             "text": "De-escalate antibiotics based on culture results and clinical improvement"},
        ],
        "hour_1_bundle": [
            "Measure lactate (remeasure if >2)",
            "Obtain blood cultures before antibiotics",
            "Administer broad-spectrum antibiotics",
            "Rapid IV crystalloid 30 mL/kg for hypotension or lactate ≥4",
            "Apply vasopressors for MAP <65 despite fluids",
        ],
        "quality_metrics": [
            "Time to antibiotics ≤1 hour",
            "Blood cultures before antibiotics",
            "Lactate measured within 1 hour",
            "30 mL/kg crystalloid within 3 hours",
            "Vasopressor within 1 hour if refractory hypotension",
            "Lactate remeasured within 2-4 hours if initially elevated",
        ],
    },
    "dka": {
        "guideline_id": "ada_dka_2024",
        "organization": "ADA",
        "title": "2024 ADA Standards: Management of Diabetic Ketoacidosis",
        "condition": "Diabetic Ketoacidosis",
        "year": 2024,
        "category": "endocrinology",
        "key_recommendations": [
            {"id": "R1", "class": "strong", "level": "A",
             "text": "IV normal saline 1-1.5 L/hr for first 1-2 hours, then 250-500 mL/hr adjusted by hydration status"},
            {"id": "R2", "class": "strong", "level": "A",
             "text": "Regular insulin infusion 0.1-0.14 U/kg/hr after potassium is confirmed ≥3.3 mEq/L",
             "medications": ["regular insulin"]},
            {"id": "R3", "class": "strong", "level": "A",
             "text": "If serum K+ <3.3: hold insulin, replete K+ 20-40 mEq/hr until K+ ≥3.3",
             "critical": True},
            {"id": "R4", "class": "strong", "level": "A",
             "text": "If K+ 3.3-5.3: give 20-30 mEq K+ in each liter of IV fluid"},
            {"id": "R5", "class": "strong", "level": "B",
             "text": "When glucose reaches 200 mg/dL: reduce insulin to 0.02-0.05 U/kg/hr, add D5 to maintain glucose 150-200"},
            {"id": "R6", "class": "strong", "level": "C",
             "text": "Bicarbonate only if pH <6.9 (give 100 mEq NaHCO3 in 400 mL with 20 mEq KCl)"},
            {"id": "R7", "class": "strong", "level": "A",
             "text": "Monitor BMP q2h, glucose q1h, ABG q2-4h until resolved"},
            {"id": "R8", "class": "strong", "level": "B",
             "text": "DKA resolution criteria: glucose <200, bicarb ≥15, pH >7.3, anion gap ≤12"},
        ],
        "severity_classification": {
            "mild": "pH 7.25-7.30, bicarb 15-18, alert",
            "moderate": "pH 7.00-7.24, bicarb 10-15, alert/drowsy",
            "severe": "pH <7.00, bicarb <10, stupor/coma",
        },
        "quality_metrics": [
            "Potassium checked before insulin",
            "Insulin infusion started within 1 hour (if K+ ≥3.3)",
            "Glucose monitored hourly",
            "BMP monitored q2h",
            "Precipitating factor identified",
        ],
    },
    "chest_pain_heart_score": {
        "guideline_id": "acep_heart_2020",
        "organization": "ACEP",
        "title": "HEART Score for Chest Pain Risk Stratification",
        "condition": "Acute Chest Pain",
        "year": 2020,
        "category": "emergency_medicine",
        "scoring": {
            "History": {"0": "Slightly suspicious", "1": "Moderately suspicious", "2": "Highly suspicious"},
            "ECG": {"0": "Normal", "1": "Non-specific changes", "2": "Significant ST deviation"},
            "Age": {"0": "<45", "1": "45-64", "2": "≥65"},
            "Risk_factors": {"0": "None", "1": "1-2 factors", "2": "≥3 factors or known CAD/PAD"},
            "Troponin": {"0": "Normal", "1": "1-3x normal", "2": ">3x normal"},
        },
        "interpretation": [
            {"score": "0-3", "risk": "Low (1.7% MACE)", "action": "Consider discharge with PCP follow-up"},
            {"score": "4-6", "risk": "Moderate (12-17% MACE)", "action": "Observation, serial troponins, cardiology consult"},
            {"score": "7-10", "risk": "High (50-65% MACE)", "action": "Admission, invasive strategy, cardiology consult"},
        ],
        "quality_metrics": [
            "HEART score documented",
            "Serial troponins for moderate risk",
            "Cardiology consultation for high risk",
        ],
    },
    "kawasaki": {
        "guideline_id": "aha_kawasaki_2017",
        "organization": "AHA",
        "title": "2017 AHA Scientific Statement: Kawasaki Disease (2024 update)",
        "condition": "Kawasaki Disease",
        "year": 2017,
        "category": "pediatrics",
        "diagnostic_criteria": {
            "required": "Fever ≥5 days",
            "plus_4_of_5": [
                "Bilateral conjunctival injection (non-purulent)",
                "Oral mucosal changes (strawberry tongue, cracked lips)",
                "Polymorphous rash",
                "Extremity changes (edema, erythema, desquamation)",
                "Cervical lymphadenopathy (≥1.5 cm, usually unilateral)",
            ],
            "incomplete_kawasaki": "Fever ≥5 days + 2-3 criteria + elevated CRP/ESR → echocardiogram",
        },
        "key_recommendations": [
            {"id": "R1", "class": "I", "level": "A",
             "text": "IVIG 2 g/kg as single infusion within 10 days of fever onset",
             "time_target": "Within 10 days of fever onset", "critical": True},
            {"id": "R2", "class": "I", "level": "A",
             "text": "High-dose aspirin 80-100 mg/kg/day divided q6h until afebrile 48-72 hours",
             "medications": ["aspirin"]},
            {"id": "R3", "class": "I", "level": "A",
             "text": "Transition to low-dose aspirin 3-5 mg/kg/day after afebrile, continue 6-8 weeks"},
            {"id": "R4", "class": "I", "level": "B",
             "text": "Echocardiogram at diagnosis, 2 weeks, and 6-8 weeks"},
            {"id": "R5", "class": "I", "level": "C",
             "text": "If IVIG-resistant (fever >36h after first IVIG): repeat IVIG or consider IV methylprednisolone"},
            {"id": "R6", "class": "I", "level": "C",
             "text": "Long-term follow-up with cardiology if coronary artery abnormalities develop"},
        ],
        "quality_metrics": [
            "IVIG within 10 days of fever onset",
            "Echocardiogram performed at diagnosis",
            "Aspirin initiated",
            "Follow-up echo scheduled",
        ],
    },
    "appendicitis": {
        "guideline_id": "wses_appendicitis_2020",
        "organization": "WSES",
        "title": "2020 WSES Jerusalem Guidelines for Diagnosis and Treatment of Acute Appendicitis",
        "condition": "Acute Appendicitis",
        "year": 2020,
        "category": "surgery",
        "key_recommendations": [
            {"id": "R1", "class": "strong", "level": "moderate",
             "text": "CT abdomen/pelvis is the gold standard for diagnosis in adults (sensitivity >95%)"},
            {"id": "R2", "class": "strong", "level": "moderate",
             "text": "Appendectomy (laparoscopic preferred) for uncomplicated appendicitis"},
            {"id": "R3", "class": "weak", "level": "moderate",
             "text": "Antibiotic-first strategy may be considered for uncomplicated appendicitis in selected patients"},
            {"id": "R4", "class": "strong", "level": "moderate",
             "text": "Preoperative antibiotics (single dose of cefoxitin or ceftriaxone + metronidazole)"},
            {"id": "R5", "class": "strong", "level": "high",
             "text": "Pregnancy test (beta-hCG) mandatory in women of childbearing age before CT"},
        ],
        "quality_metrics": [
            "Pregnancy test before CT in women of childbearing age",
            "Surgical consultation within 60 minutes of diagnosis",
            "Preoperative antibiotics administered",
        ],
    },
    "pulmonary_embolism": {
        "guideline_id": "esc_pe_2019",
        "organization": "ESC",
        "title": "2019 ESC Guidelines on Acute Pulmonary Embolism",
        "condition": "Pulmonary Embolism",
        "year": 2019,
        "category": "pulmonology",
        "wells_score": {
            "Clinical signs DVT": 3.0,
            "PE most likely diagnosis": 3.0,
            "Heart rate >100": 1.5,
            "Immobilization/surgery within 4 weeks": 1.5,
            "Previous PE/DVT": 1.5,
            "Hemoptysis": 1.0,
            "Active cancer": 1.0,
            "interpretation": {"≤4": "PE unlikely → D-dimer", ">4": "PE likely → CT-PA"},
        },
        "key_recommendations": [
            {"id": "R1", "class": "I", "level": "A",
             "text": "Use validated clinical prediction rule (Wells or revised Geneva) for pre-test probability"},
            {"id": "R2", "class": "I", "level": "A",
             "text": "D-dimer for PE-unlikely patients; CT-PA for PE-likely patients"},
            {"id": "R3", "class": "I", "level": "A",
             "text": "Initiate therapeutic anticoagulation immediately when PE confirmed"},
            {"id": "R4", "class": "I", "level": "B",
             "text": "Systemic thrombolysis for hemodynamically unstable (massive) PE"},
            {"id": "R5", "class": "IIa", "level": "B",
             "text": "Consider thrombolysis for submassive PE with RV dysfunction and clinical deterioration"},
            {"id": "R6", "class": "I", "level": "A",
             "text": "Risk stratify with sPESI score, troponin, BNP, and RV function assessment"},
        ],
        "quality_metrics": [
            "Validated clinical prediction rule used",
            "Appropriate imaging ordered (D-dimer vs CT-PA based on pre-test probability)",
            "Therapeutic anticoagulation within 1 hour of diagnosis",
            "Risk stratification documented",
        ],
    },
    "low_back_pain": {
        "guideline_id": "acp_lbp_2017",
        "organization": "ACP",
        "title": "2017 ACP Guidelines for Noninvasive Treatment of Low Back Pain",
        "condition": "Acute Low Back Pain",
        "year": 2017,
        "category": "primary_care",
        "red_flags": [
            "Cauda equina syndrome (saddle anesthesia, bowel/bladder dysfunction)",
            "Progressive neurological deficit",
            "Fever with back pain (spinal infection)",
            "History of cancer with new back pain",
            "Significant trauma (or minor trauma if osteoporosis)",
            "IV drug use (epidural abscess)",
            "Weight loss + back pain",
            "Age >50 with new onset",
        ],
        "key_recommendations": [
            {"id": "R1", "class": "strong", "level": "moderate",
             "text": "Initial treatment: superficial heat, massage, acupuncture, or spinal manipulation (non-pharmacologic first)"},
            {"id": "R2", "class": "strong", "level": "moderate",
             "text": "If pharmacologic therapy needed: NSAIDs first-line, skeletal muscle relaxants second-line"},
            {"id": "R3", "class": "strong", "level": "low",
             "text": "Do NOT routinely obtain imaging for acute low back pain without red flags"},
            {"id": "R4", "class": "strong", "level": "moderate",
             "text": "Avoid opioids for acute low back pain; consider only if NSAIDs/muscle relaxants fail"},
            {"id": "R5", "class": "strong", "level": "moderate",
             "text": "For chronic LBP: exercise, multidisciplinary rehab, CBT, mindfulness"},
        ],
        "quality_metrics": [
            "Red flags screened and documented",
            "No imaging ordered without red flags (for acute LBP <6 weeks)",
            "Non-pharmacologic therapy recommended first",
            "Opioids not prescribed as first-line",
        ],
    },
    "antimicrobial_stewardship": {
        "guideline_id": "idsa_stewardship_2023",
        "organization": "IDSA",
        "title": "Antimicrobial Stewardship Principles",
        "condition": "Antimicrobial Selection",
        "year": 2023,
        "category": "infectious_disease",
        "key_recommendations": [
            {"id": "R1", "class": "strong", "level": "high",
             "text": "Obtain cultures before antibiotics whenever possible"},
            {"id": "R2", "class": "strong", "level": "high",
             "text": "Start empiric broad-spectrum coverage, de-escalate based on cultures within 48-72 hours"},
            {"id": "R3", "class": "strong", "level": "moderate",
             "text": "Adjust doses for renal and hepatic function"},
            {"id": "R4", "class": "strong", "level": "moderate",
             "text": "Use shortest effective duration (e.g., 5-7 days for uncomplicated pneumonia, not 10-14)"},
            {"id": "R5", "class": "strong", "level": "high",
             "text": "Check for drug allergies — true penicillin allergy is rare (<5% cross-react with cephalosporins)"},
            {"id": "R6", "class": "strong", "level": "moderate",
             "text": "Procalcitonin-guided discontinuation reduces unnecessary antibiotic duration"},
        ],
        "cross_reactivity": {
            "penicillin_cephalosporin": "1-2% true cross-reactivity; safe to use 3rd/4th gen cephalosporins in most PCN-allergic patients",
            "sulfonamide_antibiotics_vs_nonantibiotic": "No cross-reactivity between sulfa antibiotics and non-antibiotic sulfonamides",
        },
        "quality_metrics": [
            "Blood cultures obtained before antibiotics",
            "Antibiotic de-escalation within 72 hours",
            "Renal dose adjustment documented",
            "Duration of therapy appropriate for indication",
        ],
    },
}

# Condition name aliases for flexible lookup
_CONDITION_ALIASES = {
    "stemi": "stemi",
    "mi": "stemi",
    "myocardial infarction": "stemi",
    "heart attack": "stemi",
    "st elevation": "stemi",
    "stroke": "acute_ischemic_stroke",
    "ischemic stroke": "acute_ischemic_stroke",
    "cva": "acute_ischemic_stroke",
    "tpa": "acute_ischemic_stroke",
    "sepsis": "sepsis",
    "septic shock": "sepsis",
    "urosepsis": "sepsis",
    "dka": "dka",
    "diabetic ketoacidosis": "dka",
    "chest pain": "chest_pain_heart_score",
    "heart score": "chest_pain_heart_score",
    "acs": "chest_pain_heart_score",
    "kawasaki": "kawasaki",
    "appendicitis": "appendicitis",
    "pe": "pulmonary_embolism",
    "pulmonary embolism": "pulmonary_embolism",
    "dvt": "pulmonary_embolism",
    "back pain": "low_back_pain",
    "low back pain": "low_back_pain",
    "lbp": "low_back_pain",
    "antibiotic": "antimicrobial_stewardship",
    "antimicrobial": "antimicrobial_stewardship",
    "stewardship": "antimicrobial_stewardship",
}


# ============================================================
# 2. API Functions
# ============================================================

def load_guidelines() -> dict:
    """Load the complete guidelines database.
    
    Returns:
        Dict of all guidelines keyed by condition ID.
    """
    return _GUIDELINES_DB.copy()


def get_guideline(condition: str) -> Optional[dict]:
    """Get guidelines for a specific condition.
    
    Args:
        condition: Condition name (flexible matching via aliases)
        
    Returns:
        Guideline dict or None if not found.
    """
    condition_lower = condition.lower().strip()
    
    # Direct match
    if condition_lower in _GUIDELINES_DB:
        return _GUIDELINES_DB[condition_lower]
    
    # Alias match
    key = _CONDITION_ALIASES.get(condition_lower)
    if key and key in _GUIDELINES_DB:
        return _GUIDELINES_DB[key]
    
    # Fuzzy match
    for alias, key in _CONDITION_ALIASES.items():
        if alias in condition_lower or condition_lower in alias:
            return _GUIDELINES_DB.get(key)
    
    return None


def get_quality_metrics(condition: str) -> list[str]:
    """Get measurable quality metrics for a condition.
    
    Args:
        condition: Condition name
        
    Returns:
        List of quality metric strings.
    """
    guideline = get_guideline(condition)
    if not guideline:
        return []
    return guideline.get("quality_metrics", [])


def check_compliance(
    actions: list[str],
    condition: str,
    time_metrics: dict = None,
) -> dict:
    """Check if clinical actions comply with guidelines.
    
    Args:
        actions: List of actions taken (descriptions)
        condition: The clinical condition
        time_metrics: Optional dict of time measurements (e.g., {"door_to_ecg": 8})
        
    Returns:
        Compliance report dict.
    """
    guideline = get_guideline(condition)
    if not guideline:
        return {"error": f"No guideline found for '{condition}'", "score": 0.0}
    
    if time_metrics is None:
        time_metrics = {}
    
    actions_lower = [a.lower() for a in actions]
    all_actions_text = " ".join(actions_lower)
    
    recommendations = guideline.get("key_recommendations", [])
    metrics = guideline.get("quality_metrics", [])
    
    # Check each recommendation
    rec_compliance = []
    for rec in recommendations:
        rec_text = rec.get("text", "").lower()
        # Check if any action addresses this recommendation
        addressed = False
        for action in actions_lower:
            # Simple keyword overlap check
            rec_words = set(rec_text.split())
            action_words = set(action.split())
            overlap = len(rec_words & action_words) / max(len(rec_words), 1)
            if overlap > 0.2:
                addressed = True
                break
        
        # Check medications if specified
        meds = rec.get("medications", [])
        if meds:
            for med in meds:
                if med.lower() in all_actions_text:
                    addressed = True
                    break
        
        rec_compliance.append({
            "recommendation_id": rec.get("id", ""),
            "text": rec.get("text", ""),
            "class": rec.get("class", ""),
            "addressed": addressed,
            "critical": rec.get("critical", False),
        })
    
    # Check quality metrics
    metric_compliance = []
    for metric in metrics:
        metric_lower = metric.lower()
        met = any(
            any(word in action for word in metric_lower.split()[:3])
            for action in actions_lower
        )
        metric_compliance.append({
            "metric": metric,
            "met": met,
        })
    
    # Scoring
    total_recs = len(rec_compliance)
    addressed_recs = sum(1 for r in rec_compliance if r["addressed"])
    critical_recs = [r for r in rec_compliance if r["critical"]]
    critical_addressed = sum(1 for r in critical_recs if r["addressed"])
    
    # Critical recommendations have higher weight
    if critical_recs:
        critical_score = critical_addressed / len(critical_recs)
    else:
        critical_score = 1.0
    
    general_score = addressed_recs / max(total_recs, 1)
    
    # Composite: 60% critical, 40% general
    compliance_score = 0.6 * critical_score + 0.4 * general_score
    
    return {
        "condition": condition,
        "guideline_id": guideline.get("guideline_id", ""),
        "compliance_score": compliance_score,
        "critical_compliance": critical_score,
        "general_compliance": general_score,
        "recommendations": rec_compliance,
        "quality_metrics": metric_compliance,
        "total_recommendations": total_recs,
        "addressed_recommendations": addressed_recs,
        "critical_total": len(critical_recs),
        "critical_addressed": critical_addressed,
    }


def get_guideline_context(condition: str, max_length: int = 2000) -> str:
    """Generate a text context string from guidelines for agent prompts.
    
    This can be injected into the agent's system prompt or observation
    to provide evidence-based guidance.
    
    Args:
        condition: The clinical condition
        max_length: Maximum character length
        
    Returns:
        Formatted guideline text for agent context.
    """
    guideline = get_guideline(condition)
    if not guideline:
        return ""
    
    lines = [
        f"=== Clinical Guidelines: {guideline.get('title', condition)} ===",
        f"Organization: {guideline.get('organization', 'N/A')} ({guideline.get('year', 'N/A')})",
        "",
        "Key Recommendations:",
    ]
    
    for rec in guideline.get("key_recommendations", []):
        cls = rec.get("class", "")
        text = rec.get("text", "")
        time_target = rec.get("time_target", "")
        critical = " [CRITICAL]" if rec.get("critical") else ""
        time_str = f" (Target: {time_target})" if time_target else ""
        lines.append(f"  [{cls}] {text}{time_str}{critical}")
    
    lines.append("")
    lines.append("Quality Metrics:")
    for metric in guideline.get("quality_metrics", []):
        lines.append(f"  - {metric}")
    
    text = "\n".join(lines)
    if len(text) > max_length:
        text = text[:max_length - 3] + "..."
    
    return text
