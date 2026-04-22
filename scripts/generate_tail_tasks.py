#!/usr/bin/env python3
"""Generate high-quality tasks for tail domains: psychiatry and obstetrics.

Creates clinically realistic scenario-based tasks with MCQA options
and open-ended variants for RL training.
"""
import json
import random
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────────
#  Psychiatry Task Templates
# ─────────────────────────────────────────────────────────────

PSYCH_CONDITIONS = [
    {
        "name": "major_depressive_disorder",
        "presentations": [
            "depressed mood for 3 weeks, anhedonia, insomnia, poor concentration, weight loss",
            "fatigue, hopelessness, guilt, decreased appetite for 1 month, withdrew from social activities",
            "crying spells, insomnia for 6 weeks, difficulty at work, loss of interest in hobbies",
        ],
        "key_assessments": ["phq9", "suicide_risk", "mental_status"],
        "treatments": ["SSRIs", "CBT", "SNRIs", "mirtazapine"],
        "differentials": ["bipolar disorder", "adjustment disorder", "hypothyroidism", "substance use"],
    },
    {
        "name": "generalized_anxiety_disorder",
        "presentations": [
            "excessive worry about multiple domains for 8 months, muscle tension, restlessness, difficulty sleeping",
            "persistent anxiety, irritability, fatigue, difficulty concentrating for over a year",
            "chronic worry, GI complaints, palpitations, tension headaches for 6 months",
        ],
        "key_assessments": ["gad7", "mental_status", "substance_use"],
        "treatments": ["SSRIs", "buspirone", "CBT", "SNRIs"],
        "differentials": ["panic disorder", "hyperthyroidism", "caffeine excess", "PTSD"],
    },
    {
        "name": "schizophrenia",
        "presentations": [
            "auditory hallucinations, paranoid delusions, social withdrawal, flat affect for 8 months",
            "disorganized speech, bizarre behavior, hearing voices for 1 year, declining function",
            "persecutory delusions, thought broadcasting, deteriorating self-care for 6 months",
        ],
        "key_assessments": ["mental_status", "mmse", "capacity"],
        "treatments": ["risperidone", "olanzapine", "aripiprazole", "clozapine"],
        "differentials": ["schizoaffective disorder", "brief psychotic disorder", "substance-induced psychosis", "delusional disorder"],
    },
    {
        "name": "bipolar_disorder_type_1",
        "presentations": [
            "decreased need for sleep, grandiosity, pressured speech, impulsive spending for 1 week",
            "elevated mood, increased energy, racing thoughts, risky sexual behavior for 5 days",
            "irritability, psychomotor agitation, flight of ideas, decreased sleep for 10 days",
        ],
        "key_assessments": ["mental_status", "suicide_risk", "substance_use"],
        "treatments": ["lithium", "valproate", "quetiapine", "lamotrigine"],
        "differentials": ["schizoaffective disorder", "ADHD", "hyperthyroidism", "substance-induced mania"],
    },
    {
        "name": "PTSD",
        "presentations": [
            "nightmares, hypervigilance, avoidance of trauma reminders, emotional numbness after combat 6 months ago",
            "flashbacks, startle response, insomnia, irritability following sexual assault 1 year ago",
            "intrusive memories, emotional detachment, difficulty trusting others after motor vehicle accident 4 months ago",
        ],
        "key_assessments": ["pcl5", "suicide_risk", "substance_use"],
        "treatments": ["CPT", "PE therapy", "sertraline", "prazosin for nightmares"],
        "differentials": ["acute stress disorder", "adjustment disorder", "major depression", "TBI"],
    },
    {
        "name": "alcohol_use_disorder",
        "presentations": [
            "drinking escalated over 2 years, morning tremors, missed work, family complaints",
            "blackouts, withdrawal seizure history, drinking to avoid shaking, liver enzyme elevation",
            "daily drinking 12 beers, neglecting responsibilities, failed attempts to cut down",
        ],
        "key_assessments": ["cage", "mental_status", "suicide_risk"],
        "treatments": ["naltrexone", "acamprosate", "disulfiram", "motivational interviewing"],
        "differentials": ["other substance use", "liver disease", "anxiety disorder", "depression"],
    },
    {
        "name": "panic_disorder",
        "presentations": [
            "recurrent episodes of palpitations, chest tightness, derealization, fear of dying",
            "sudden onset shortness of breath, tingling, dizziness, feeling of impending doom",
            "unexpected panic attacks 3x/week, agoraphobia developing, avoids crowded places",
        ],
        "key_assessments": ["gad7", "mental_status", "phq9"],
        "treatments": ["SSRIs", "CBT with exposure", "benzodiazepines short-term"],
        "differentials": ["cardiac arrhythmia", "hyperthyroidism", "pheochromocytoma", "GAD"],
    },
    {
        "name": "anorexia_nervosa",
        "presentations": [
            "18yo female, BMI 16.5, amenorrhea, excessive exercise, body image distortion",
            "22yo female, restricting intake to 500cal/day, lanugo hair, bradycardia, electrolyte abnormalities",
            "17yo male, significant weight loss, food rituals, purging behavior, dental erosion",
        ],
        "key_assessments": ["eating_disorder", "mental_status", "suicide_risk"],
        "treatments": ["nutritional rehabilitation", "FBT", "CBT-E", "olanzapine for severe cases"],
        "differentials": ["bulimia nervosa", "malabsorption", "hyperthyroidism", "malignancy"],
    },
    {
        "name": "OCD",
        "presentations": [
            "repetitive handwashing 4+ hours/day, contamination fears, raw skin on hands",
            "intrusive violent thoughts, checking locks/stove 30+ times, significantly impaired function",
            "symmetry obsessions, counting rituals, unable to leave house on time for work",
        ],
        "key_assessments": ["mental_status", "phq9", "gad7"],
        "treatments": ["fluvoxamine", "high-dose SSRIs", "ERP therapy", "clomipramine"],
        "differentials": ["GAD", "hoarding disorder", "body dysmorphic disorder", "tic disorder"],
    },
    {
        "name": "ADHD_adult",
        "presentations": [
            "lifelong difficulty concentrating, loses items frequently, poor time management, impulsivity",
            "chronic procrastination, job changes, relationship difficulties, childhood history of hyperactivity",
            "cannot focus in meetings, forgetful, starts many projects but finishes none, restlessness",
        ],
        "key_assessments": ["mental_status", "substance_use", "phq9"],
        "treatments": ["methylphenidate", "amphetamine salts", "atomoxetine", "behavioral strategies"],
        "differentials": ["anxiety disorder", "sleep disorder", "bipolar disorder", "thyroid dysfunction"],
    },
]

PSYCH_PATIENTS = [
    {"age_range": (18, 25), "gender": "F", "context": "college student"},
    {"age_range": (18, 25), "gender": "M", "context": "college student"},
    {"age_range": (30, 45), "gender": "M", "context": "professional"},
    {"age_range": (30, 45), "gender": "F", "context": "working mother"},
    {"age_range": (55, 70), "gender": "M", "context": "retired veteran"},
    {"age_range": (55, 70), "gender": "F", "context": "recently widowed"},
    {"age_range": (14, 17), "gender": "M", "context": "high school student"},
    {"age_range": (14, 17), "gender": "F", "context": "high school student"},
]

PSYCH_MCQA_TEMPLATES = [
    {
        "q": "What is the MOST appropriate initial pharmacotherapy for this patient?",
        "make_options": lambda c: c["treatments"] + ["observation only"] if len(c["treatments"]) < 5 else c["treatments"][:5],
        "answer_idx": 0,
    },
    {
        "q": "Which screening instrument is MOST appropriate for this patient's presentation?",
        "make_options": lambda c: [c["key_assessments"][0].upper().replace("_", "-")] + ["AUDIT-C", "DAST-10", "CAGE", "MDQ"],
        "answer_idx": 0,
    },
    {
        "q": "Which of the following is the MOST likely diagnosis?",
        "make_options": lambda c: [c["name"].replace("_", " ").title()] + c["differentials"][:4],
        "answer_idx": 0,
    },
    {
        "q": "Which condition should be ruled out FIRST in the differential diagnosis?",
        "make_options": lambda c: c["differentials"][:4] + ["none of the above"],
        "answer_idx": 0,
    },
]


def generate_psychiatry_tasks(target: int = 100) -> list[dict]:
    tasks = []
    task_idx = 0

    # Open-ended clinical scenario tasks
    for condition in PSYCH_CONDITIONS:
        for presentation in condition["presentations"]:
            for patient in random.sample(PSYCH_PATIENTS, min(2, len(PSYCH_PATIENTS))):
                if task_idx >= target * 0.6:  # 60% open-ended
                    break
                task_idx += 1
                age = random.randint(*patient["age_range"])
                gender = "male" if patient["gender"] == "M" else "female"

                ticket = (
                    f"A {age}-year-old {gender} ({patient['context']}) presents with {presentation}. "
                    f"No prior psychiatric history. "
                    f"Perform a comprehensive psychiatric evaluation and provide your assessment and management plan."
                )

                tasks.append({
                    "id": f"psych_{condition['name']}_{patient['context'].replace(' ', '_')}_{task_idx:03d}",
                    "ticket": ticket,
                    "correct_answer": f"Diagnosis: {condition['name'].replace('_', ' ')}. "
                                      f"Key assessments: {', '.join(condition['key_assessments'])}. "
                                      f"First-line treatment: {condition['treatments'][0]}.",
                    "_source_domain": "psychiatry",
                    "description": {
                        "condition": condition["name"],
                        "patient_type": patient["context"],
                        "task_type": "clinical_scenario",
                    },
                    "evaluation_criteria": {
                        "actions": [
                            {"name": "perform_assessment", "arguments": {"assessment_type": a}, "compare_args": []}
                            for a in condition["key_assessments"]
                        ],
                        "reward_basis": ["ACTION", "NL_ASSERTION"],
                        "assertions": [
                            f"Agent should consider {condition['name'].replace('_', ' ')} as primary diagnosis",
                            f"Agent should assess for {', '.join(condition['key_assessments'])}",
                        ],
                    },
                })
            if task_idx >= target * 0.6:
                break

    # MCQA tasks
    for condition in PSYCH_CONDITIONS:
        for mcqa in PSYCH_MCQA_TEMPLATES:
            if task_idx >= target:
                break
            task_idx += 1
            patient = random.choice(PSYCH_PATIENTS)
            age = random.randint(*patient["age_range"])
            gender = "male" if patient["gender"] == "M" else "female"
            presentation = random.choice(condition["presentations"])

            raw_options = mcqa["make_options"](condition)
            # Shuffle options but track correct answer
            correct_text = raw_options[mcqa["answer_idx"]]
            random.shuffle(raw_options)
            correct_letter = chr(65 + raw_options.index(correct_text))

            options = {chr(65 + i): opt for i, opt in enumerate(raw_options[:5])}

            question = (
                f"A {age}-year-old {gender} presents with {presentation}. "
                f"{mcqa['q']}"
            )

            tasks.append({
                "id": f"psych_mcqa_{condition['name']}_{task_idx:03d}",
                "ticket": question,
                "options": options,
                "correct_answer": correct_letter,
                "_source_domain": "psychiatry",
                "description": {
                    "condition": condition["name"],
                    "task_type": "mcqa",
                },
                "evaluation_criteria": {
                    "reward_basis": ["MCQA"],
                },
            })

        if task_idx >= target:
            break

    return tasks


# ─────────────────────────────────────────────────────────────
#  Obstetrics Task Templates
# ─────────────────────────────────────────────────────────────

OB_CONDITIONS = [
    {
        "name": "preeclampsia",
        "presentations": [
            "32 weeks gestation, BP 160/100, proteinuria 2+, headache, visual changes",
            "36 weeks, BP 150/95, 3+ protein, RUQ pain, elevated LFTs, platelets 90k",
            "28 weeks, new onset hypertension 145/95, trace proteinuria, edema, weight gain",
        ],
        "key_assessments": ["fetal", "labor", "risk"],
        "management": ["magnesium sulfate", "antihypertensives", "delivery timing"],
        "differentials": ["gestational hypertension", "chronic hypertension", "HELLP syndrome", "TTP"],
    },
    {
        "name": "gestational_diabetes",
        "presentations": [
            "28 weeks, failed 1-hour glucose challenge 185 mg/dL, BMI 32, polyhydramnios",
            "24 weeks, family hx diabetes, fasting glucose 102, macrosomia on ultrasound",
            "30 weeks, 3-hour GTT abnormal (2/4 values elevated), no prior diabetes history",
        ],
        "key_assessments": ["fetal", "risk", "biophysical"],
        "management": ["dietary modification", "glucose monitoring", "insulin if needed", "fetal surveillance"],
        "differentials": ["pre-existing type 2 diabetes", "impaired glucose tolerance", "steroid-induced hyperglycemia"],
    },
    {
        "name": "placenta_previa",
        "presentations": [
            "32 weeks, painless vaginal bleeding, ultrasound shows complete previa",
            "28 weeks, episode of bright red bleeding, no contractions, low-lying placenta",
            "34 weeks, second episode vaginal bleeding, hemodynamically stable, previa on prior US",
        ],
        "key_assessments": ["fetal", "labor", "risk"],
        "management": ["pelvic rest", "serial ultrasounds", "planned cesarean", "blood type and screen"],
        "differentials": ["placental abruption", "vasa previa", "cervical pathology", "bloody show"],
    },
    {
        "name": "preterm_labor",
        "presentations": [
            "30 weeks, regular contractions q5min, cervix 3cm/80%, positive fFN",
            "28 weeks, pelvic pressure, cervical shortening to 15mm on TVUS, mild contractions",
            "32 weeks, ruptured membranes confirmed by nitrazine/ferning, no contractions yet",
        ],
        "key_assessments": ["labor", "fetal", "ctg"],
        "management": ["tocolytics", "betamethasone", "magnesium for neuroprotection", "antibiotics if PPROM"],
        "differentials": ["Braxton-Hicks", "UTI", "cervical insufficiency", "preterm PROM"],
    },
    {
        "name": "ectopic_pregnancy",
        "presentations": [
            "6 weeks amenorrhea, unilateral pelvic pain, vaginal spotting, beta-hCG 1500 with empty uterus on TVUS",
            "7 weeks LMP, acute RLQ pain, positive pregnancy test, adnexal mass on ultrasound",
            "8 weeks, worsening abdominal pain, shoulder tip pain, hemodynamic instability, positive pregnancy test",
        ],
        "key_assessments": ["risk", "fetal"],
        "management": ["methotrexate if stable", "surgical management", "serial beta-hCG", "Rh status"],
        "differentials": ["threatened abortion", "ovarian cyst rupture", "appendicitis", "PID"],
    },
    {
        "name": "postpartum_hemorrhage",
        "presentations": [
            "2 hours post vaginal delivery, estimated blood loss 1200mL, boggy uterus, tachycardia",
            "30 minutes post cesarean, ongoing bleeding, uterine atony despite oxytocin, BP dropping",
            "1 hour postpartum, steady bleeding from perineal laceration, well-contracted uterus",
        ],
        "key_assessments": ["labor", "risk"],
        "management": ["uterine massage", "uterotonics", "transfusion", "surgical intervention if refractory"],
        "differentials": ["uterine atony", "retained placenta", "genital tract laceration", "coagulopathy"],
    },
    {
        "name": "hyperemesis_gravidarum",
        "presentations": [
            "10 weeks, persistent vomiting 8x/day, 5% weight loss, ketonuria, unable to tolerate oral intake",
            "8 weeks, severe nausea/vomiting, dehydration, electrolyte abnormalities, elevated liver enzymes",
            "12 weeks, intractable vomiting despite antiemetics, requiring IV fluids, hypokalemia",
        ],
        "key_assessments": ["fetal", "risk"],
        "management": ["IV fluids", "thiamine", "antiemetics (ondansetron)", "electrolyte correction"],
        "differentials": ["molar pregnancy", "gastroparesis", "biliary disease", "thyrotoxicosis"],
    },
    {
        "name": "shoulder_dystocia",
        "presentations": [
            "active labor, fetal head delivered but anterior shoulder impacted, turtle sign observed",
            "G2P1, GDM, EFW 4200g, prolonged second stage, head delivered but no restitution",
            "precipitous delivery, macrosomic infant, difficulty delivering shoulders after head",
        ],
        "key_assessments": ["labor", "fetal", "risk"],
        "management": ["McRoberts maneuver", "suprapubic pressure", "episiotomy", "Rubin/Woods maneuver"],
        "differentials": ["normal labor progression", "cephalopelvic disproportion"],
    },
]

OB_PATIENTS = [
    {"age_range": (20, 28), "gravida": "G1P0", "context": "primigravida"},
    {"age_range": (28, 35), "gravida": "G2P1", "context": "multigravida"},
    {"age_range": (35, 42), "gravida": "G3P2", "context": "advanced maternal age"},
    {"age_range": (18, 22), "gravida": "G1P0", "context": "young primigravida"},
    {"age_range": (30, 38), "gravida": "G4P3", "context": "grand multipara"},
]

OB_MCQA_TEMPLATES = [
    {
        "q": "What is the MOST appropriate next step in management?",
        "make_options": lambda c: c["management"][:4] + ["expectant management"],
        "answer_idx": 0,
    },
    {
        "q": "Which of the following is the MOST likely diagnosis?",
        "make_options": lambda c: [c["name"].replace("_", " ").title()] + c["differentials"][:4],
        "answer_idx": 0,
    },
    {
        "q": "Which condition should be considered FIRST in the differential?",
        "make_options": lambda c: c["differentials"][:4] + ["none of the above"],
        "answer_idx": 0,
    },
]


def generate_obstetrics_tasks(target: int = 100) -> list[dict]:
    tasks = []
    task_idx = 0

    # Open-ended clinical scenario tasks
    for condition in OB_CONDITIONS:
        for presentation in condition["presentations"]:
            for patient in random.sample(OB_PATIENTS, min(2, len(OB_PATIENTS))):
                if task_idx >= target * 0.6:
                    break
                task_idx += 1
                age = random.randint(*patient["age_range"])

                ticket = (
                    f"A {age}-year-old {patient['gravida']} ({patient['context']}) presents: {presentation}. "
                    f"Provide a comprehensive obstetric assessment and management plan."
                )

                tasks.append({
                    "id": f"ob_{condition['name']}_{patient['context'].replace(' ', '_')}_{task_idx:03d}",
                    "ticket": ticket,
                    "correct_answer": f"Diagnosis: {condition['name'].replace('_', ' ')}. "
                                      f"Key management: {', '.join(condition['management'][:3])}.",
                    "_source_domain": "obstetrics",
                    "description": {
                        "condition": condition["name"],
                        "patient_type": patient["context"],
                        "task_type": "clinical_scenario",
                    },
                    "evaluation_criteria": {
                        "actions": [
                            {"name": "assess_obstetric_status", "arguments": {"assessment_type": a}, "compare_args": []}
                            for a in condition["key_assessments"]
                        ],
                        "reward_basis": ["ACTION", "NL_ASSERTION"],
                        "assertions": [
                            f"Agent should identify {condition['name'].replace('_', ' ')}",
                            f"Agent should recommend {condition['management'][0]}",
                        ],
                    },
                })
            if task_idx >= target * 0.6:
                break

    # MCQA tasks
    for condition in OB_CONDITIONS:
        for mcqa in OB_MCQA_TEMPLATES:
            if task_idx >= target:
                break
            task_idx += 1
            patient = random.choice(OB_PATIENTS)
            age = random.randint(*patient["age_range"])
            presentation = random.choice(condition["presentations"])

            raw_options = mcqa["make_options"](condition)
            correct_text = raw_options[mcqa["answer_idx"]]
            random.shuffle(raw_options)
            correct_letter = chr(65 + raw_options.index(correct_text))

            options = {chr(65 + i): opt for i, opt in enumerate(raw_options[:5])}

            question = (
                f"A {age}-year-old {patient['gravida']} presents: {presentation}. "
                f"{mcqa['q']}"
            )

            tasks.append({
                "id": f"ob_mcqa_{condition['name']}_{task_idx:03d}",
                "ticket": question,
                "options": options,
                "correct_answer": correct_letter,
                "_source_domain": "obstetrics",
                "description": {
                    "condition": condition["name"],
                    "task_type": "mcqa",
                },
                "evaluation_criteria": {
                    "reward_basis": ["MCQA"],
                },
            })

        if task_idx >= target:
            break

    return tasks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate tail domain tasks")
    parser.add_argument("--target", type=int, default=100, help="Target tasks per domain")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    for domain, gen_fn in [("psychiatry", generate_psychiatry_tasks), ("obstetrics", generate_obstetrics_tasks)]:
        tasks = gen_fn(target=args.target)
        random.shuffle(tasks)

        split_idx = int(len(tasks) * 0.7)
        train_tasks = tasks[:split_idx]
        test_tasks = tasks[split_idx:]

        out_dir = PROJECT_ROOT / "data" / "domains" / domain
        out_dir.mkdir(parents=True, exist_ok=True)

        tasks_path = out_dir / "tasks_scaled.json"
        with open(tasks_path, "w") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        split_path = out_dir / "split_tasks_scaled.json"
        with open(split_path, "w") as f:
            json.dump(
                {"train": [t["id"] for t in train_tasks], "test": [t["id"] for t in test_tasks]},
                f, indent=2,
            )

        mcqa_count = sum(1 for t in tasks if t.get("options"))
        open_count = len(tasks) - mcqa_count
        print(f"{domain}: {len(tasks)} tasks ({mcqa_count} MCQA + {open_count} open-ended) "
              f"[{len(train_tasks)} train / {len(test_tasks)} test] → {tasks_path}")

    print("Done!")


if __name__ == "__main__":
    main()
