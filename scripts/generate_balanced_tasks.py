#!/usr/bin/env python3
"""Generate balanced tasks for all underrepresented domains.

Targets ~500 tasks per domain by expanding template diversity.
Outputs tasks_balanced.json per domain.
"""
import json
import random
from pathlib import Path
from itertools import product

random.seed(42)
PROJECT_ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────────
#  Shared patient templates (expanded)
# ─────────────────────────────────────────────────────────────
PATIENTS = [
    {"age_range": (18, 25), "gender": "M", "comorbidities": [], "ctx": "college student"},
    {"age_range": (18, 25), "gender": "F", "comorbidities": [], "ctx": "college student"},
    {"age_range": (25, 35), "gender": "M", "comorbidities": ["smoking"], "ctx": "office worker"},
    {"age_range": (25, 35), "gender": "F", "comorbidities": [], "ctx": "nurse"},
    {"age_range": (35, 50), "gender": "M", "comorbidities": ["hypertension", "obesity"], "ctx": "truck driver"},
    {"age_range": (35, 50), "gender": "F", "comorbidities": ["diabetes"], "ctx": "teacher"},
    {"age_range": (50, 65), "gender": "M", "comorbidities": ["COPD", "hypertension"], "ctx": "retired engineer"},
    {"age_range": (50, 65), "gender": "F", "comorbidities": ["osteoarthritis", "hypothyroidism"], "ctx": "homemaker"},
    {"age_range": (65, 80), "gender": "M", "comorbidities": ["atrial_fibrillation", "CKD", "diabetes"], "ctx": "retired"},
    {"age_range": (65, 80), "gender": "F", "comorbidities": ["heart_failure", "osteoporosis"], "ctx": "nursing home resident"},
    {"age_range": (3, 10), "gender": "M", "comorbidities": [], "ctx": "pediatric"},
    {"age_range": (3, 10), "gender": "F", "comorbidities": ["asthma"], "ctx": "pediatric"},
    {"age_range": (12, 17), "gender": "M", "comorbidities": [], "ctx": "adolescent"},
    {"age_range": (12, 17), "gender": "F", "comorbidities": [], "ctx": "adolescent"},
]

ONSETS = ["30 minutes", "2 hours", "6 hours", "1 day", "2 days", "3 days", "1 week", "2 weeks", "1 month"]
LETTERS = "ABCDE"


def make_mcqa(question: str, correct: str, distractors: list, domain: str, tid: str, condition: str) -> dict:
    """Create an MCQA task with shuffled options."""
    options_list = [correct] + distractors[:4]
    if len(options_list) < 5:
        options_list.append("None of the above")
    options_list = options_list[:5]
    random.shuffle(options_list)
    correct_letter = LETTERS[options_list.index(correct)]
    options = {LETTERS[i]: o for i, o in enumerate(options_list)}
    return {
        "id": tid,
        "ticket": question,
        "options": options,
        "correct_answer": correct_letter,
        "_source_domain": domain,
        "description": {"condition": condition, "task_type": "mcqa"},
        "evaluation_criteria": {"reward_basis": ["MCQA"]},
    }


def make_open(ticket: str, answer: str, domain: str, tid: str, condition: str, actions: list = None) -> dict:
    """Create an open-ended task."""
    t = {
        "id": tid,
        "ticket": ticket,
        "correct_answer": answer,
        "_source_domain": domain,
        "description": {"condition": condition, "task_type": "clinical_scenario"},
        "evaluation_criteria": {"reward_basis": ["ACTION", "NL_ASSERTION"]},
    }
    if actions:
        t["evaluation_criteria"]["actions"] = actions
    return t


def patient_str(p):
    age = random.randint(*p["age_range"])
    g = "male" if p["gender"] == "M" else "female"
    cmb = f" PMH: {', '.join(p['comorbidities'])}." if p["comorbidities"] else ""
    return age, g, cmb


# ─────────────────────────────────────────────────────────────
#  Drug Interaction — expanded (target: +184)
# ─────────────────────────────────────────────────────────────

DRUG_PAIRS = [
    ("warfarin", "aspirin", "increased bleeding risk", "high", ["monitoring INR", "consider PPI", "use lowest aspirin dose"]),
    ("warfarin", "fluconazole", "CYP2C9 inhibition increases warfarin levels", "high", ["reduce warfarin dose", "frequent INR monitoring"]),
    ("warfarin", "rifampin", "CYP induction decreases warfarin levels", "high", ["increase warfarin dose", "monitor INR closely"]),
    ("metformin", "IV contrast", "risk of lactic acidosis", "high", ["hold metformin 48h", "check renal function"]),
    ("fluoxetine", "tramadol", "serotonin syndrome risk", "high", ["avoid combination", "use alternative analgesic"]),
    ("fluoxetine", "MAOIs", "serotonin syndrome risk", "severe", ["contraindicated", "14-day washout required"]),
    ("simvastatin", "amiodarone", "increased myopathy/rhabdomyolysis risk", "high", ["limit simvastatin to 20mg", "monitor CK"]),
    ("simvastatin", "clarithromycin", "CYP3A4 inhibition increases statin levels", "high", ["hold statin during antibiotic course"]),
    ("lisinopril", "spironolactone", "hyperkalemia risk", "moderate", ["monitor potassium", "avoid K supplements"]),
    ("lisinopril", "NSAIDs", "reduced ACEi efficacy + AKI risk", "moderate", ["monitor renal function", "limit NSAID use"]),
    ("clopidogrel", "omeprazole", "CYP2C19 inhibition reduces clopidogrel efficacy", "moderate", ["switch to pantoprazole"]),
    ("phenytoin", "valproate", "displaced protein binding increases free phenytoin", "high", ["monitor free phenytoin levels"]),
    ("methotrexate", "trimethoprim", "reduced methotrexate clearance", "high", ["avoid combination", "monitor CBC"]),
    ("digoxin", "amiodarone", "increased digoxin levels", "high", ["reduce digoxin dose by 50%", "monitor levels"]),
    ("digoxin", "verapamil", "increased digoxin levels + AV block", "high", ["reduce digoxin dose", "monitor ECG"]),
    ("lithium", "thiazide diuretics", "decreased lithium clearance", "high", ["monitor lithium levels", "reduce dose"]),
    ("lithium", "NSAIDs", "decreased lithium clearance", "high", ["avoid NSAIDs", "monitor lithium levels"]),
    ("ciprofloxacin", "theophylline", "CYP1A2 inhibition increases theophylline", "moderate", ["reduce theophylline dose", "monitor levels"]),
    ("ciprofloxacin", "antacids", "chelation reduces ciprofloxacin absorption", "moderate", ["separate by 2 hours"]),
    ("rifampin", "oral contraceptives", "CYP induction reduces OCP efficacy", "moderate", ["use alternative contraception"]),
    ("carbamazepine", "erythromycin", "CYP3A4 inhibition increases carbamazepine", "high", ["monitor carbamazepine levels"]),
    ("sildenafil", "nitrates", "severe hypotension", "severe", ["absolutely contraindicated"]),
    ("potassium", "ACE inhibitors", "hyperkalemia", "moderate", ["monitor potassium closely"]),
    ("grapefruit", "cyclosporine", "CYP3A4 inhibition increases cyclosporine", "moderate", ["avoid grapefruit"]),
    ("SSRIs", "triptans", "serotonin syndrome risk", "moderate", ["monitor for symptoms", "use lowest doses"]),
    ("clonidine", "beta-blockers", "rebound hypertension if clonidine stopped", "high", ["taper clonidine first"]),
    ("metronidazole", "alcohol", "disulfiram-like reaction", "moderate", ["avoid alcohol during treatment"]),
    ("azithromycin", "amiodarone", "QT prolongation risk", "high", ["ECG monitoring", "avoid if possible"]),
    ("quetiapine", "carbamazepine", "CYP3A4 induction reduces quetiapine levels", "moderate", ["increase quetiapine dose"]),
    ("tamoxifen", "paroxetine", "CYP2D6 inhibition reduces tamoxifen efficacy", "high", ["switch to venlafaxine or citalopram"]),
]

DRUG_QUESTIONS = [
    ("What is the PRIMARY mechanism of this drug interaction?", lambda d: d[2]),
    ("What is the MOST appropriate management strategy?", lambda d: d[4][0]),
    ("How would you classify the SEVERITY of this interaction?", lambda d: d[3]),
    ("Which monitoring parameter is MOST important?", lambda d: d[4][-1] if len(d[4]) > 1 else d[4][0]),
]


def gen_drug_interaction(target: int) -> list:
    tasks = []
    idx = 0
    contexts = [
        "new prescription review", "medication reconciliation at admission",
        "discharge medication check", "outpatient follow-up",
        "pharmacist consultation", "pre-operative medication review",
    ]

    for drug_pair in DRUG_PAIRS:
        d1, d2, mechanism, severity, mgmt = drug_pair
        for patient in random.sample(PATIENTS[:10], min(4, len(PATIENTS))):
            if idx >= target:
                return tasks
            idx += 1
            age, gender, cmb = patient_str(patient)
            context = random.choice(contexts)

            # Open-ended
            ticket = (
                f"During {context}, a {age}-year-old {gender} is found to be taking {d1}. "
                f"The physician wants to prescribe {d2}.{cmb} "
                f"Evaluate the potential drug interaction and provide recommendations."
            )
            tasks.append(make_open(ticket, f"Interaction: {mechanism}. Management: {', '.join(mgmt)}.",
                                   "drug_interaction", f"di_bal_{d1}_{d2}_{idx:03d}", f"{d1}_{d2}",
                                   [{"name": "get_drug_info", "arguments": {}, "compare_args": []},
                                    {"name": "check_interaction", "arguments": {}, "compare_args": []}]))

        # MCQA per pair
        for qtype in DRUG_QUESTIONS:
            if idx >= target:
                return tasks
            idx += 1
            q_text, answer_fn = qtype
            patient = random.choice(PATIENTS)
            age, gender, cmb = patient_str(patient)
            presentation = random.choice(drug_pair[4])

            question = f"A {age}-year-old {gender} is taking {d1} and {d2}.{cmb} {q_text}"
            correct = answer_fn(drug_pair)
            distractors = ["No interaction expected", "Dose adjustment unnecessary",
                           "Monitor liver function only", "Switch to alternative medication",
                           "Discontinue both medications"]
            tasks.append(make_mcqa(question, correct, distractors, "drug_interaction",
                                   f"di_mcqa_{d1}_{d2}_{idx:03d}", f"{d1}_{d2}"))

    return tasks


# ─────────────────────────────────────────────────────────────
#  Triage Emergency — expanded (target: +278)
# ─────────────────────────────────────────────────────────────

TRIAGE_SCENARIOS = [
    {"complaint": "chest pain", "esi": [1,2,3], "workup": ["ECG", "troponin", "CXR"], "critical": ["STEMI", "aortic dissection", "PE"]},
    {"complaint": "shortness of breath", "esi": [1,2,3], "workup": ["CXR", "ABG", "D-dimer"], "critical": ["PE", "tension pneumothorax", "anaphylaxis"]},
    {"complaint": "altered mental status", "esi": [1,2], "workup": ["glucose", "CT head", "tox screen"], "critical": ["stroke", "hypoglycemia", "meningitis"]},
    {"complaint": "abdominal pain", "esi": [2,3,4], "workup": ["CBC", "lipase", "CT abdomen"], "critical": ["ruptured AAA", "perforated viscus"]},
    {"complaint": "headache", "esi": [2,3,4], "workup": ["CT head", "LP", "ESR"], "critical": ["SAH", "meningitis", "temporal arteritis"]},
    {"complaint": "seizure", "esi": [1,2], "workup": ["glucose", "electrolytes", "CT head"], "critical": ["status epilepticus", "eclampsia"]},
    {"complaint": "major trauma", "esi": [1,2], "workup": ["FAST exam", "CT", "type and screen"], "critical": ["hemorrhagic shock", "tension pneumothorax"]},
    {"complaint": "overdose/ingestion", "esi": [1,2], "workup": ["tox screen", "ECG", "acetaminophen level"], "critical": ["respiratory depression", "cardiac arrest"]},
    {"complaint": "allergic reaction", "esi": [1,2,3], "workup": ["vitals", "airway assessment"], "critical": ["anaphylaxis", "angioedema"]},
    {"complaint": "syncope", "esi": [2,3], "workup": ["ECG", "orthostatics", "glucose"], "critical": ["cardiac arrhythmia", "PE", "aortic stenosis"]},
    {"complaint": "GI bleeding", "esi": [1,2,3], "workup": ["CBC", "type and screen", "BMP"], "critical": ["massive hemorrhage", "variceal bleeding"]},
    {"complaint": "fever with neutropenia", "esi": [2], "workup": ["CBC", "blood cultures", "CXR"], "critical": ["sepsis", "fungal infection"]},
    {"complaint": "acute vision loss", "esi": [2,3], "workup": ["visual acuity", "fundoscopy", "CT/MRI"], "critical": ["CRAO", "retinal detachment", "stroke"]},
    {"complaint": "severe back pain", "esi": [2,3], "workup": ["MRI spine", "UA", "BMP"], "critical": ["cauda equina", "aortic dissection", "epidural abscess"]},
    {"complaint": "palpitations", "esi": [2,3,4], "workup": ["ECG", "electrolytes", "TSH"], "critical": ["VT", "SVT", "Afib with RVR"]},
    {"complaint": "stroke symptoms", "esi": [1,2], "workup": ["CT head", "CTA", "glucose"], "critical": ["ischemic stroke", "hemorrhagic stroke"]},
    {"complaint": "testicular pain", "esi": [2,3], "workup": ["doppler US", "UA"], "critical": ["testicular torsion", "Fournier's gangrene"]},
    {"complaint": "postpartum bleeding", "esi": [1,2], "workup": ["CBC", "coags", "uterine assessment"], "critical": ["uterine atony", "DIC"]},
    {"complaint": "burns", "esi": [1,2,3], "workup": ["TBSA assessment", "airway eval", "IV access"], "critical": ["inhalation injury", "circumferential burns"]},
    {"complaint": "diabetic emergency", "esi": [1,2], "workup": ["glucose", "BMP", "ABG", "ketones"], "critical": ["DKA", "HHS", "severe hypoglycemia"]},
]


def _vitals(esi):
    if esi == 1:
        return f"HR {random.randint(120,160)}, BP {random.randint(60,85)}/{random.randint(30,50)}, RR {random.randint(28,40)}, SpO2 {random.randint(75,88)}%"
    elif esi == 2:
        return f"HR {random.randint(100,130)}, BP {random.randint(85,110)}/{random.randint(50,70)}, RR {random.randint(22,30)}, SpO2 {random.randint(88,94)}%"
    elif esi == 3:
        return f"HR {random.randint(80,110)}, BP {random.randint(110,140)}/{random.randint(60,85)}, RR {random.randint(16,24)}, SpO2 {random.randint(94,98)}%"
    else:
        return f"HR {random.randint(65,90)}, BP {random.randint(115,140)}/{random.randint(65,85)}, RR {random.randint(14,20)}, SpO2 {random.randint(97,100)}%"


def gen_triage(target: int) -> list:
    tasks = []
    idx = 0

    for scenario in TRIAGE_SCENARIOS:
        for esi in scenario["esi"]:
            for patient in random.sample(PATIENTS, min(3, len(PATIENTS))):
                if idx >= target:
                    return tasks
                idx += 1
                age, gender, cmb = patient_str(patient)
                onset = random.choice(ONSETS[:6])
                vitals = _vitals(esi)

                ticket = (
                    f"A {age}-year-old {gender} arrives at the ED with {scenario['complaint']}. "
                    f"Onset: {onset} ago. Vitals: {vitals}.{cmb} "
                    f"Perform triage assessment and assign ESI level."
                )
                tasks.append(make_open(ticket, f"ESI {esi}",
                                       "triage_emergency", f"tri_bal_{scenario['complaint'].replace(' ','_')}_{esi}_{idx:03d}",
                                       scenario["complaint"]))

            # MCQA: what workup?
            if idx >= target:
                return tasks
            idx += 1
            patient = random.choice(PATIENTS)
            age, gender, cmb = patient_str(patient)
            question = (
                f"A {age}-year-old {gender} presents to the ED with {scenario['complaint']}. "
                f"Vitals: {_vitals(scenario['esi'][0])}.{cmb} "
                f"Which diagnostic study should be ordered FIRST?"
            )
            tasks.append(make_mcqa(question, scenario["workup"][0],
                                   scenario["workup"][1:] + ["observation only", "discharge home"],
                                   "triage_emergency", f"tri_mcqa_{idx:03d}", scenario["complaint"]))

            # MCQA: critical diagnosis
            if scenario["critical"] and idx < target:
                idx += 1
                question2 = (
                    f"A {age}-year-old {gender} presents with acute {scenario['complaint']} and hemodynamic instability. "
                    f"Which life-threatening condition must be ruled out FIRST?"
                )
                tasks.append(make_mcqa(question2, scenario["critical"][0],
                                       scenario["critical"][1:] + ["viral syndrome", "musculoskeletal strain", "anxiety"],
                                       "triage_emergency", f"tri_mcqa2_{idx:03d}", scenario["complaint"]))

    return tasks


# ─────────────────────────────────────────────────────────────
#  EHR Management — expanded (target: +296)
# ─────────────────────────────────────────────────────────────

EHR_TASKS = [
    {"type": "chart_review", "focus": "admission assessment", "q": "Perform comprehensive admission chart review"},
    {"type": "chart_review", "focus": "pre-operative clearance", "q": "Evaluate pre-operative fitness"},
    {"type": "chart_review", "focus": "ICU transfer assessment", "q": "Assess patient for ICU transfer"},
    {"type": "medication_reconciliation", "focus": "admission", "q": "Reconcile home medications with admission orders"},
    {"type": "medication_reconciliation", "focus": "discharge", "q": "Prepare discharge medication reconciliation"},
    {"type": "medication_reconciliation", "focus": "transfer", "q": "Reconcile medications for unit transfer"},
    {"type": "critical_value", "focus": "lab critical", "q": "Review and respond to critical lab values"},
    {"type": "critical_value", "focus": "vital sign alert", "q": "Assess and respond to vital sign alerts"},
    {"type": "clinical_scoring", "focus": "SOFA", "q": "Calculate and interpret SOFA score"},
    {"type": "clinical_scoring", "focus": "NEWS2", "q": "Calculate NEWS2 score and escalation plan"},
    {"type": "clinical_scoring", "focus": "APACHE II", "q": "Calculate APACHE II score for ICU prognosis"},
    {"type": "clinical_scoring", "focus": "CURB-65", "q": "Calculate CURB-65 for pneumonia disposition"},
    {"type": "clinical_scoring", "focus": "Wells PE", "q": "Calculate Wells score for PE probability"},
    {"type": "clinical_scoring", "focus": "CHA2DS2-VASc", "q": "Calculate CHA2DS2-VASc for stroke risk"},
    {"type": "clinical_scoring", "focus": "MELD", "q": "Calculate MELD score for liver disease severity"},
    {"type": "discharge_planning", "focus": "complex discharge", "q": "Plan complex discharge with multiple services"},
    {"type": "discharge_planning", "focus": "AMA discharge", "q": "Document against medical advice discharge"},
    {"type": "quality_measure", "focus": "sepsis bundle", "q": "Verify sepsis bundle compliance within 3 hours"},
    {"type": "quality_measure", "focus": "VTE prophylaxis", "q": "Assess VTE prophylaxis adequacy"},
    {"type": "quality_measure", "focus": "fall risk", "q": "Evaluate fall risk and prevention measures"},
    {"type": "antibiotic_stewardship", "focus": "de-escalation", "q": "Review antibiotic therapy for de-escalation"},
    {"type": "antibiotic_stewardship", "focus": "duration review", "q": "Assess antibiotic duration appropriateness"},
    {"type": "longitudinal_analysis", "focus": "lab trend", "q": "Analyze lab result trends over past 72 hours"},
    {"type": "longitudinal_analysis", "focus": "vital trend", "q": "Analyze vital sign trends for clinical deterioration"},
    {"type": "readmission_risk", "focus": "30-day", "q": "Assess 30-day readmission risk and interventions"},
    {"type": "nutrition_assessment", "focus": "malnutrition", "q": "Screen for malnutrition and recommend nutrition plan"},
    {"type": "pain_management", "focus": "opioid review", "q": "Review opioid use and multimodal pain strategy"},
    {"type": "handoff", "focus": "shift change", "q": "Prepare structured SBAR handoff for shift change"},
    {"type": "handoff", "focus": "service transfer", "q": "Prepare handoff for transfer to another service"},
    {"type": "consult_triage", "focus": "specialty consult", "q": "Evaluate need for specialty consultation"},
]

HADM_IDS = [f"HADM_{i}" for i in range(10001, 10021)]


def gen_ehr(target: int) -> list:
    tasks = []
    idx = 0

    for ehr_task in EHR_TASKS:
        for hadm_id in random.sample(HADM_IDS, min(8, len(HADM_IDS))):
            if idx >= target:
                return tasks
            idx += 1
            patient = random.choice(PATIENTS[:10])
            age, gender, cmb = patient_str(patient)

            ticket = (
                f"{ehr_task['q']} for patient {hadm_id} ({age}-year-old {gender}).{cmb} "
                f"Focus: {ehr_task['focus']}. "
                f"Review all available data and provide your assessment."
            )
            tasks.append(make_open(ticket, f"Task type: {ehr_task['type']}, focus: {ehr_task['focus']}",
                                   "ehr_management", f"ehr_bal_{ehr_task['type']}_{hadm_id}_{idx:03d}",
                                   ehr_task["type"],
                                   [{"name": "get_patient_records", "arguments": {"patient_id": hadm_id}, "compare_args": ["patient_id"]},
                                    {"name": "get_lab_results", "arguments": {"patient_id": hadm_id}, "compare_args": ["patient_id"]}]))

    # MCQA for clinical scoring
    scoring_mcqa = [
        ("A patient has BP 85/50, HR 120, creatinine 3.2, bilirubin 4.1, platelets 45k, GCS 12, PaO2/FiO2 200. What is the SOFA score category?",
         "SOFA ≥ 10 (high mortality risk)", ["SOFA 0-5 (low risk)", "SOFA 6-9 (moderate)", "Cannot calculate", "SOFA ≥ 15"]),
        ("An 80-year-old with pneumonia, confusion, BUN 25, RR 32, BP 85/50. What is the CURB-65 score?",
         "CURB-65 = 5 (highest severity)", ["CURB-65 = 1", "CURB-65 = 2", "CURB-65 = 3", "Cannot determine"]),
        ("A patient with AFib, age 76, HTN, DM, prior stroke, female. What is CHA2DS2-VASc score?",
         "CHA2DS2-VASc ≥ 6 (high risk, anticoagulate)", ["Score 0 (no treatment)", "Score 1-2 (consider)", "Score 3 (moderate)", "Cannot calculate"]),
    ]
    for q, correct, distractors in scoring_mcqa:
        if idx >= target:
            return tasks
        idx += 1
        tasks.append(make_mcqa(q, correct, distractors, "ehr_management", f"ehr_mcqa_{idx:03d}", "clinical_scoring"))

    return tasks


# ─────────────────────────────────────────────────────────────
#  Psychiatry — expanded (target: +339)
# ─────────────────────────────────────────────────────────────

PSYCH_CONDITIONS = [
    {"name": "major_depressive_disorder", "sx": ["depressed mood", "anhedonia", "insomnia", "weight change", "fatigue", "guilt", "poor concentration", "suicidal ideation"],
     "tx": ["sertraline", "fluoxetine", "CBT", "venlafaxine", "bupropion", "mirtazapine"], "dx": ["bipolar", "adjustment disorder", "hypothyroidism", "grief"]},
    {"name": "generalized_anxiety_disorder", "sx": ["excessive worry", "restlessness", "muscle tension", "sleep disturbance", "irritability", "difficulty concentrating"],
     "tx": ["sertraline", "buspirone", "CBT", "venlafaxine", "duloxetine"], "dx": ["panic disorder", "hyperthyroidism", "PTSD", "social anxiety"]},
    {"name": "schizophrenia", "sx": ["auditory hallucinations", "paranoid delusions", "flat affect", "social withdrawal", "disorganized speech", "avolition"],
     "tx": ["risperidone", "olanzapine", "aripiprazole", "clozapine", "paliperidone"], "dx": ["schizoaffective", "brief psychotic disorder", "substance-induced psychosis", "delusional disorder"]},
    {"name": "bipolar I disorder", "sx": ["decreased sleep", "grandiosity", "pressured speech", "racing thoughts", "impulsivity", "distractibility", "increased goal-directed activity"],
     "tx": ["lithium", "valproate", "quetiapine", "lamotrigine", "carbamazepine"], "dx": ["schizoaffective", "ADHD", "hyperthyroidism", "BPD"]},
    {"name": "PTSD", "sx": ["flashbacks", "nightmares", "hypervigilance", "avoidance", "emotional numbing", "startle response", "irritability"],
     "tx": ["sertraline", "CPT", "prolonged exposure", "prazosin", "venlafaxine"], "dx": ["acute stress disorder", "adjustment disorder", "MDD", "TBI"]},
    {"name": "OCD", "sx": ["intrusive thoughts", "compulsive rituals", "contamination fears", "checking behavior", "symmetry obsessions", "mental rituals"],
     "tx": ["fluvoxamine", "high-dose sertraline", "ERP therapy", "clomipramine", "fluoxetine"], "dx": ["GAD", "hoarding", "body dysmorphic", "tic disorder"]},
    {"name": "panic disorder", "sx": ["palpitations", "shortness of breath", "chest tightness", "derealization", "fear of dying", "tingling", "dizziness"],
     "tx": ["sertraline", "CBT with interoceptive exposure", "paroxetine", "alprazolam short-term"], "dx": ["cardiac arrhythmia", "hyperthyroidism", "pheochromocytoma", "GAD"]},
    {"name": "alcohol use disorder", "sx": ["tolerance", "withdrawal", "craving", "continued use despite consequences", "blackouts", "morning tremors"],
     "tx": ["naltrexone", "acamprosate", "disulfiram", "motivational interviewing", "AA/12-step"], "dx": ["other SUDs", "liver disease", "anxiety", "depression"]},
    {"name": "opioid use disorder", "sx": ["tolerance", "withdrawal", "craving", "needle marks", "pinpoint pupils", "constipation", "respiratory depression risk"],
     "tx": ["buprenorphine/naloxone", "methadone", "naltrexone XR", "counseling"], "dx": ["chronic pain syndrome", "other SUDs", "depression"]},
    {"name": "ADHD", "sx": ["inattention", "hyperactivity", "impulsivity", "disorganization", "forgetfulness", "difficulty with tasks requiring sustained focus"],
     "tx": ["methylphenidate", "amphetamine salts", "atomoxetine", "behavioral strategies", "guanfacine"], "dx": ["anxiety", "sleep disorder", "bipolar", "thyroid"]},
    {"name": "borderline personality disorder", "sx": ["unstable relationships", "fear of abandonment", "identity disturbance", "impulsivity", "self-harm", "emotional lability", "chronic emptiness"],
     "tx": ["DBT", "mentalization-based therapy", "mood stabilizers adjunct", "brief CBT"], "dx": ["bipolar II", "PTSD", "MDD", "histrionic PD"]},
    {"name": "anorexia nervosa", "sx": ["restriction", "body image distortion", "fear of weight gain", "amenorrhea", "bradycardia", "hypothermia", "lanugo"],
     "tx": ["nutritional rehabilitation", "FBT", "CBT-E", "olanzapine", "medical stabilization"], "dx": ["bulimia", "ARFID", "malabsorption", "hyperthyroidism"]},
    {"name": "social anxiety disorder", "sx": ["fear of social situations", "avoidance", "performance anxiety", "blushing", "trembling", "fear of judgment"],
     "tx": ["sertraline", "CBT with exposure", "venlafaxine", "propranolol for performance"], "dx": ["GAD", "avoidant PD", "selective mutism", "agoraphobia"]},
    {"name": "insomnia disorder", "sx": ["difficulty initiating sleep", "difficulty maintaining sleep", "early morning awakening", "daytime fatigue", "irritability"],
     "tx": ["CBT-I", "sleep hygiene", "melatonin", "trazodone", "suvorexant"], "dx": ["sleep apnea", "restless legs", "depression", "substance use"]},
    {"name": "delirium", "sx": ["acute confusion", "fluctuating consciousness", "inattention", "disorganized thinking", "perceptual disturbances", "psychomotor changes"],
     "tx": ["treat underlying cause", "reorientation", "haloperidol PRN", "avoid benzodiazepines"], "dx": ["dementia", "psychosis", "depression", "seizure"]},
]

PSYCH_Q_TYPES = [
    "What is the MOST likely diagnosis?",
    "What is the FIRST-LINE pharmacotherapy?",
    "Which psychotherapy approach is MOST evidence-based?",
    "Which screening instrument should be administered?",
    "What is the MOST important safety assessment?",
]


def gen_psychiatry(target: int) -> list:
    tasks = []
    idx = 0
    assessments_map = {
        "major_depressive_disorder": ["phq9", "suicide_risk"],
        "generalized_anxiety_disorder": ["gad7", "mental_status"],
        "schizophrenia": ["mental_status", "mmse"],
        "bipolar I disorder": ["mental_status", "suicide_risk"],
        "PTSD": ["pcl5", "suicide_risk"],
        "OCD": ["mental_status", "phq9"],
        "panic disorder": ["gad7", "mental_status"],
        "alcohol use disorder": ["cage", "mental_status"],
        "opioid use disorder": ["mental_status", "suicide_risk"],
        "ADHD": ["mental_status", "phq9"],
        "borderline personality disorder": ["suicide_risk", "mental_status"],
        "anorexia nervosa": ["eating_disorder", "suicide_risk"],
        "social anxiety disorder": ["gad7", "phq9"],
        "insomnia disorder": ["mental_status", "phq9"],
        "delirium": ["mental_status", "mmse"],
    }

    for condition in PSYCH_CONDITIONS:
        # Generate varied presentations
        for patient in random.sample(PATIENTS, min(5, len(PATIENTS))):
            if idx >= target * 0.55:
                break
            idx += 1
            age, gender, cmb = patient_str(patient)
            num_sx = random.randint(3, min(5, len(condition["sx"])))
            sx = random.sample(condition["sx"], num_sx)
            duration = random.choice(["2 weeks", "1 month", "3 months", "6 months", "1 year"])

            ticket = (
                f"A {age}-year-old {gender} ({patient['ctx']}) presents with {', '.join(sx[:-1])} and {sx[-1]} "
                f"for {duration}.{cmb} "
                f"Perform psychiatric assessment and provide diagnosis with management plan."
            )
            assessments = assessments_map.get(condition["name"], ["mental_status"])
            tasks.append(make_open(ticket, f"Diagnosis: {condition['name']}. First-line: {condition['tx'][0]}.",
                                   "psychiatry", f"psy_bal_{condition['name']}_{idx:03d}", condition["name"],
                                   [{"name": "perform_assessment", "arguments": {"assessment_type": a}, "compare_args": []}
                                    for a in assessments]))

        # MCQA variants
        for q_type in PSYCH_Q_TYPES:
            if idx >= target:
                break
            idx += 1
            patient = random.choice(PATIENTS)
            age, gender, cmb = patient_str(patient)
            sx = random.sample(condition["sx"], min(4, len(condition["sx"])))

            question = f"A {age}-year-old {gender} presents with {', '.join(sx)}.{cmb} {q_type}"

            if "diagnosis" in q_type.lower():
                correct = condition["name"].replace("_", " ").title()
                distractors = [d.replace("_", " ").title() for d in condition["dx"]]
            elif "pharmacotherapy" in q_type.lower():
                correct = condition["tx"][0]
                distractors = condition["tx"][1:4] + ["no medication needed"]
            elif "psychotherapy" in q_type.lower():
                therapies = [t for t in condition["tx"] if any(w in t.lower() for w in ["cbt", "dbt", "exposure", "therapy", "counseling", "motivational", "fbt"])]
                correct = therapies[0] if therapies else condition["tx"][1]
                distractors = ["psychoanalysis", "hypnotherapy", "biofeedback", "art therapy"]
            elif "screening" in q_type.lower():
                assess = assessments_map.get(condition["name"], ["mental_status"])[0]
                correct = assess.upper().replace("_", "-")
                distractors = ["AUDIT-C", "DAST-10", "MoCA", "Y-BOCS", "EPDS"]
            else:
                correct = "Suicide risk assessment"
                distractors = ["Physical exam", "Neuroimaging", "Genetic testing", "Sleep study"]

            tasks.append(make_mcqa(question, correct, distractors, "psychiatry",
                                   f"psy_mcqa_{condition['name']}_{idx:03d}", condition["name"]))

    return tasks


# ─────────────────────────────────────────────────────────────
#  Obstetrics — expanded (target: +370)
# ─────────────────────────────────────────────────────────────

OB_CONDITIONS = [
    {"name": "preeclampsia", "presentations": ["BP 160/100, proteinuria, headache at 32w", "BP 150/95, RUQ pain, elevated LFTs at 36w", "new HTN 145/95, edema at 28w", "severe features: BP 170/110, visual changes, thrombocytopenia at 34w"],
     "mgmt": ["magnesium sulfate", "antihypertensives", "delivery timing", "fetal monitoring"], "dx": ["gestational HTN", "chronic HTN", "HELLP", "TTP-HUS"]},
    {"name": "gestational diabetes", "presentations": ["failed GCT at 28w, BMI 32", "fasting glucose 102, macrosomia on US", "3-hour GTT abnormal", "polyhydramnios, large for dates"],
     "mgmt": ["dietary modification", "glucose monitoring", "insulin if needed", "fetal surveillance"], "dx": ["pre-existing T2DM", "IGT", "steroid-induced"]},
    {"name": "placenta previa", "presentations": ["painless bleeding at 32w, complete previa", "bright red bleeding at 28w, low-lying placenta", "recurrent bleeding at 34w"],
     "mgmt": ["pelvic rest", "serial US", "planned cesarean", "blood type and screen"], "dx": ["abruption", "vasa previa", "cervical pathology", "bloody show"]},
    {"name": "placental abruption", "presentations": ["painful vaginal bleeding, rigid uterus at 35w", "abdominal pain after trauma, fetal distress at 30w", "dark bleeding, uterine tenderness, back pain at 33w"],
     "mgmt": ["emergent delivery if severe", "fetal monitoring", "IV access and fluids", "coagulation studies"], "dx": ["placenta previa", "preterm labor", "uterine rupture"]},
    {"name": "preterm labor", "presentations": ["regular contractions q5min, cervix 3cm at 30w", "pelvic pressure, cervical shortening at 28w", "PPROM confirmed at 32w"],
     "mgmt": ["tocolytics", "betamethasone", "Mg for neuroprotection", "antibiotics if PPROM"], "dx": ["Braxton-Hicks", "UTI", "cervical insufficiency"]},
    {"name": "ectopic pregnancy", "presentations": ["unilateral pain, empty uterus, hCG 1500 at 6w", "acute RLQ pain, adnexal mass at 7w", "shoulder tip pain, hemodynamic instability at 8w"],
     "mgmt": ["methotrexate if stable", "surgical management", "serial beta-hCG", "Rh status"], "dx": ["threatened abortion", "ovarian cyst", "appendicitis", "PID"]},
    {"name": "postpartum hemorrhage", "presentations": ["EBL 1200mL, boggy uterus 2h postpartum", "ongoing bleeding post-cesarean, atony", "steady bleeding from laceration"],
     "mgmt": ["uterine massage", "uterotonics", "transfusion", "surgical intervention"], "dx": ["atony", "retained placenta", "laceration", "coagulopathy"]},
    {"name": "hyperemesis gravidarum", "presentations": ["vomiting 8x/day, 5% weight loss at 10w", "intractable vomiting, ketonuria, hypokalemia at 8w", "dehydration requiring IV fluids at 12w"],
     "mgmt": ["IV fluids", "thiamine", "ondansetron", "electrolyte correction"], "dx": ["molar pregnancy", "gastroparesis", "biliary disease", "thyrotoxicosis"]},
    {"name": "shoulder dystocia", "presentations": ["turtle sign after head delivery, EFW 4200g", "prolonged second stage, no restitution", "macrosomia, difficulty delivering shoulders"],
     "mgmt": ["McRoberts", "suprapubic pressure", "episiotomy", "Woods/Rubin maneuver"], "dx": ["normal labor", "cephalopelvic disproportion"]},
    {"name": "cord prolapse", "presentations": ["visible cord after ROM, FHR deceleration", "palpable cord on exam after SROM", "cord presenting with AROM"],
     "mgmt": ["elevate presenting part", "emergency cesarean", "knee-chest position", "fill bladder"], "dx": ["cord compression", "fetal distress other cause"]},
    {"name": "chorioamnionitis", "presentations": ["maternal fever 38.5, tachycardia, foul-smelling fluid after prolonged ROM", "tender uterus, fetal tachycardia, WBC 18k", "maternal fever, prolonged rupture >18h"],
     "mgmt": ["IV antibiotics", "delivery", "antipyretics", "neonatal evaluation"], "dx": ["UTI", "other infection source", "epidural fever"]},
    {"name": "postpartum depression", "presentations": ["crying, hopelessness, difficulty bonding 3 weeks postpartum", "insomnia, guilt, inability to care for infant 6 weeks PP", "suicidal thoughts, anxiety about infant safety 2 weeks PP"],
     "mgmt": ["SSRIs", "psychotherapy", "support groups", "brexanolone if severe"], "dx": ["baby blues", "postpartum psychosis", "thyroid dysfunction", "adjustment disorder"]},
]

OB_PATIENTS = [
    {"age_range": (18, 22), "gravida": "G1P0", "ctx": "primigravida"},
    {"age_range": (22, 28), "gravida": "G1P0", "ctx": "primigravida"},
    {"age_range": (25, 30), "gravida": "G2P1", "ctx": "multigravida"},
    {"age_range": (28, 33), "gravida": "G2P1", "ctx": "multigravida with prior cesarean"},
    {"age_range": (33, 38), "gravida": "G3P2", "ctx": "advanced maternal age"},
    {"age_range": (38, 42), "gravida": "G4P3", "ctx": "grand multipara, AMA"},
    {"age_range": (16, 19), "gravida": "G1P0", "ctx": "adolescent primigravida"},
    {"age_range": (30, 36), "gravida": "G1P0", "ctx": "IVF pregnancy"},
]

OB_Q_TYPES = [
    "What is the MOST appropriate next step in management?",
    "What is the MOST likely diagnosis?",
    "Which complication should be assessed for FIRST?",
    "What is the MOST important fetal monitoring step?",
]


def gen_obstetrics(target: int) -> list:
    tasks = []
    idx = 0

    for condition in OB_CONDITIONS:
        for presentation in condition["presentations"]:
            for patient in random.sample(OB_PATIENTS, min(3, len(OB_PATIENTS))):
                if idx >= target * 0.55:
                    break
                idx += 1
                age = random.randint(*patient["age_range"])

                ticket = (
                    f"A {age}-year-old {patient['gravida']} ({patient['ctx']}) presents: {presentation}. "
                    f"Provide comprehensive obstetric assessment and management plan."
                )
                tasks.append(make_open(ticket, f"Diagnosis: {condition['name']}. Management: {', '.join(condition['mgmt'][:3])}.",
                                       "obstetrics", f"ob_bal_{condition['name']}_{idx:03d}", condition["name"],
                                       [{"name": "assess_obstetric_status", "arguments": {}, "compare_args": []}]))
            if idx >= target * 0.55:
                break

        # MCQA per condition
        for q_type in OB_Q_TYPES:
            if idx >= target:
                break
            idx += 1
            patient = random.choice(OB_PATIENTS)
            age = random.randint(*patient["age_range"])
            pres = random.choice(condition["presentations"])

            question = f"A {age}-year-old {patient['gravida']} presents: {pres}. {q_type}"

            if "management" in q_type.lower():
                correct, distractors = condition["mgmt"][0], condition["mgmt"][1:] + ["expectant management"]
            elif "diagnosis" in q_type.lower():
                correct = condition["name"].replace("_", " ").title()
                distractors = [d.replace("_", " ").title() for d in condition["dx"]]
            elif "complication" in q_type.lower():
                correct = condition["dx"][0] if condition["dx"] else "DIC"
                distractors = (condition["dx"][1:] if len(condition["dx"]) > 1 else []) + ["no complications expected", "routine follow-up"]
            else:
                correct = "Continuous fetal heart rate monitoring"
                distractors = ["Intermittent auscultation", "No monitoring needed", "Weekly NST", "Biophysical profile only"]

            tasks.append(make_mcqa(question, correct, distractors, "obstetrics",
                                   f"ob_mcqa_{condition['name']}_{idx:03d}", condition["name"]))

    return tasks


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    # Current counts in train parquet
    current = {
        "drug_interaction": 316,
        "triage_emergency": 222,
        "ehr_management": 204,
        "psychiatry": 161,
        "obstetrics": 130,
    }
    target_per_domain = 500

    generators = {
        "drug_interaction": gen_drug_interaction,
        "triage_emergency": gen_triage,
        "ehr_management": gen_ehr,
        "psychiatry": gen_psychiatry,
        "obstetrics": gen_obstetrics,
    }

    total_new = 0
    for domain, gen_fn in generators.items():
        need = target_per_domain - current[domain]
        if need <= 0:
            print(f"  {domain}: already at {current[domain]}, skip")
            continue

        # Generate more than needed (70/30 split will reduce train count)
        raw_target = int(need / 0.7) + 10
        tasks = gen_fn(raw_target)
        random.shuffle(tasks)

        split_idx = int(len(tasks) * 0.7)
        train = tasks[:split_idx]
        test = tasks[split_idx:]

        out_dir = PROJECT_ROOT / "data" / "domains" / domain
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "tasks_balanced.json", "w") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        with open(out_dir / "split_tasks_balanced.json", "w") as f:
            json.dump({"train": [t["id"] for t in train], "test": [t["id"] for t in test]}, f, indent=2)

        mcqa = sum(1 for t in tasks if t.get("options"))
        total_new += len(tasks)
        print(f"  {domain}: {len(tasks)} tasks ({mcqa} MCQA + {len(tasks)-mcqa} open) "
              f"[{len(train)} train / {len(test)} test]")

    print(f"\nTotal new tasks: {total_new}")


if __name__ == "__main__":
    main()
