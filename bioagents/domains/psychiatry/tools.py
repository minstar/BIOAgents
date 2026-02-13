"""Tools for the Psychiatry / Mental Health domain.

Provides 14 tools for psychiatric assessment and management:
1.  get_patient_presentation  - Chief complaint, referral, history summary
2.  get_psychiatric_history   - Detailed psychiatric history and diagnoses
3.  perform_mental_status_exam - Mental status examination
4.  administer_phq9           - PHQ-9 depression screening
5.  administer_gad7           - GAD-7 anxiety screening
6.  assess_suicide_risk       - Columbia-SSRS suicide risk assessment
7.  screen_substance_use      - AUDIT/DAST substance use screening
8.  administer_mmse           - Mini-Mental State Examination
9.  get_current_medications   - Current psychiatric medications
10. check_drug_interactions   - Psychiatric drug interaction check
11. get_social_history        - Social support, housing, employment
12. review_treatment_guidelines - Evidence-based treatment guidelines
13. think                     - Internal reasoning
14. submit_answer             - Submit diagnosis and treatment plan
"""

import json
from typing import Any

from bioagents.domains.psychiatry.data_model import PsychiatryDB
from bioagents.environment.toolkit import ToolKitBase, is_tool


class PsychiatryTools(ToolKitBase):
    """Tool kit for Psychiatry / Mental Health domain."""

    def __init__(self, db: PsychiatryDB):
        super().__init__()
        self.db = db

    @is_tool
    def get_patient_presentation(self, patient_id: str) -> str:
        """Get initial psychiatric presentation including chief complaint, referral source, demographics, and vital signs.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with patient presentation.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        return json.dumps({
            "patient_id": p.patient_id, "name": p.name, "age": p.age, "sex": p.sex,
            "chief_complaint": p.chief_complaint,
            "referral_source": p.referral_source,
            "vitals": p.vitals,
            "current_diagnoses": p.current_diagnoses,
            "allergies": p.allergies,
        }, indent=2)

    @is_tool
    def get_psychiatric_history(self, patient_id: str) -> str:
        """Get detailed psychiatric history including past diagnoses, hospitalizations, trauma history, and family psychiatric history.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with psychiatric history.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        return json.dumps({
            "patient_id": patient_id,
            "psychiatric_history": p.psychiatric_history,
            "current_diagnoses": p.current_diagnoses,
            "medical_comorbidities": p.medical_comorbidities,
            "trauma_history": p.trauma_history,
            "family_psychiatric_history": p.family_psychiatric_history,
        }, indent=2)

    @is_tool
    def perform_mental_status_exam(self, patient_id: str) -> str:
        """Perform a Mental Status Examination (MSE) including appearance, behavior, speech, mood, affect, thought process/content, perceptions, cognition, insight, and judgment.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with MSE findings.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        mse = p.mental_status_exam
        if not mse:
            return json.dumps({"patient_id": patient_id, "note": "MSE not yet documented"})
        return json.dumps({
            "patient_id": patient_id,
            "mental_status_exam": mse.model_dump(),
        }, indent=2)

    @is_tool
    def administer_phq9(self, patient_id: str) -> str:
        """Administer the PHQ-9 (Patient Health Questionnaire-9) depression screening. Returns total score, severity level, and individual item responses.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with PHQ-9 results.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        for s in p.screening_scores:
            if s.instrument == "PHQ-9":
                return json.dumps({
                    "patient_id": patient_id,
                    "instrument": "PHQ-9",
                    "total_score": s.total_score,
                    "severity": s.severity,
                    "item_responses": s.item_responses,
                    "risk_flags": s.risk_flags,
                    "interpretation": s.interpretation,
                }, indent=2)
        return json.dumps({"patient_id": patient_id, "note": "PHQ-9 not yet administered. Screening unavailable in records."})

    @is_tool
    def administer_gad7(self, patient_id: str) -> str:
        """Administer the GAD-7 (Generalized Anxiety Disorder-7) screening. Returns total score, severity level, and item responses.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with GAD-7 results.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        for s in p.screening_scores:
            if s.instrument == "GAD-7":
                return json.dumps({
                    "patient_id": patient_id,
                    "instrument": "GAD-7",
                    "total_score": s.total_score,
                    "severity": s.severity,
                    "item_responses": s.item_responses,
                    "interpretation": s.interpretation,
                }, indent=2)
        return json.dumps({"patient_id": patient_id, "note": "GAD-7 not yet administered."})

    @is_tool
    def assess_suicide_risk(self, patient_id: str) -> str:
        """Perform Columbia Suicide Severity Rating Scale (C-SSRS) assessment. Returns risk level, ideation severity, behavior history, and recommended actions.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with suicide risk assessment.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        for s in p.screening_scores:
            if s.instrument in ("Columbia-SSRS", "C-SSRS"):
                return json.dumps({
                    "patient_id": patient_id,
                    "instrument": "Columbia-SSRS",
                    "risk_level": p.suicide_risk_level,
                    "total_score": s.total_score,
                    "severity": s.severity,
                    "risk_flags": s.risk_flags,
                    "item_responses": s.item_responses,
                    "interpretation": s.interpretation,
                    "recommended_actions": _suicide_risk_actions(p.suicide_risk_level),
                }, indent=2)
        # Generate assessment from available data
        risk = p.suicide_risk_level
        return json.dumps({
            "patient_id": patient_id,
            "instrument": "Columbia-SSRS",
            "risk_level": risk,
            "recommended_actions": _suicide_risk_actions(risk),
            "note": "Full C-SSRS data not available; risk level from clinical assessment.",
        }, indent=2)

    @is_tool
    def screen_substance_use(self, patient_id: str) -> str:
        """Screen for substance use disorders using AUDIT (alcohol) and DAST (drugs). Returns scores and substance use history.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with substance use screening results.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        results = {"patient_id": patient_id, "substance_use": p.substance_use}
        for s in p.screening_scores:
            if s.instrument in ("AUDIT", "DAST-10"):
                results[s.instrument] = {
                    "total_score": s.total_score, "severity": s.severity,
                    "interpretation": s.interpretation,
                }
        return json.dumps(results, indent=2)

    @is_tool
    def administer_mmse(self, patient_id: str) -> str:
        """Administer Mini-Mental State Examination (MMSE) for cognitive screening. Returns total score and domain-specific results.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with MMSE results.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        for s in p.screening_scores:
            if s.instrument == "MMSE":
                return json.dumps({
                    "patient_id": patient_id, "instrument": "MMSE",
                    "total_score": s.total_score, "severity": s.severity,
                    "item_responses": s.item_responses,
                    "interpretation": s.interpretation,
                }, indent=2)
        return json.dumps({"patient_id": patient_id, "note": "MMSE not administered."})

    @is_tool
    def get_current_medications(self, patient_id: str) -> str:
        """Get current psychiatric and non-psychiatric medications including dosages, adherence, and reported side effects.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with medication list.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        meds = [m.model_dump() for m in p.medications]
        return json.dumps({"patient_id": patient_id, "medications": meds}, indent=2)

    @is_tool
    def check_drug_interactions(self, drug_a: str, drug_b: str) -> str:
        """Check for interactions between two psychiatric medications. Includes serotonin syndrome risk, QTc prolongation, metabolic interactions.

        Args:
            drug_a: First drug name.
            drug_b: Second drug name.

        Returns:
            JSON string with interaction details.
        """
        interactions = _check_psych_interactions(drug_a.lower(), drug_b.lower())
        return json.dumps({
            "drug_a": drug_a, "drug_b": drug_b,
            "interactions": interactions,
        }, indent=2)

    @is_tool
    def get_social_history(self, patient_id: str) -> str:
        """Get social history including support system, housing, employment, legal issues, and protective/risk factors.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with social history.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        return json.dumps({
            "patient_id": patient_id,
            "social_support": p.social_support,
            "housing_status": p.housing_status,
            "employment": p.employment,
            "legal_issues": p.legal_issues,
            "substance_use": p.substance_use,
        }, indent=2)

    @is_tool
    def review_treatment_guidelines(self, condition: str) -> str:
        """Look up evidence-based treatment guidelines for a psychiatric condition. Returns first-line, second-line treatments, and special considerations.

        Args:
            condition: Psychiatric condition (e.g., 'MDD', 'GAD', 'PTSD', 'bipolar', 'schizophrenia').

        Returns:
            JSON string with treatment guideline summary.
        """
        key = condition.lower().replace(" ", "_")
        gl = self.db.treatment_guidelines.get(key)
        if gl:
            return json.dumps({"condition": condition, "guidelines": gl}, indent=2)
        return json.dumps(_default_guidelines(condition), indent=2)

    @is_tool
    def think(self, thought: str) -> str:
        """Record internal clinical reasoning without taking an action.

        Args:
            thought: Your clinical reasoning process.

        Returns:
            Acknowledgment.
        """
        return json.dumps({"thought_recorded": True, "content": thought[:500]})

    @is_tool
    def submit_answer(self, diagnosis: str, risk_level: str, treatment_plan: str, disposition: str, reasoning: str) -> str:
        """Submit psychiatric assessment and treatment plan.

        Args:
            diagnosis: Primary psychiatric diagnosis (DSM-5).
            risk_level: Suicide/self-harm risk level (low/moderate/high/imminent).
            treatment_plan: Recommended treatment (medications, therapy, safety plan).
            disposition: Recommended disposition (outpatient, IOP, PHP, inpatient, involuntary hold).
            reasoning: Clinical reasoning for decisions.

        Returns:
            Confirmation.
        """
        return json.dumps({
            "submitted": True,
            "diagnosis": diagnosis, "risk_level": risk_level,
            "treatment_plan": treatment_plan[:500],
            "disposition": disposition, "reasoning": reasoning[:500],
        })


def _suicide_risk_actions(risk_level: str) -> list:
    actions = {
        "low": ["Safety planning", "Follow-up appointment within 1-2 weeks", "Crisis hotline information"],
        "moderate": ["Safety planning with means restriction", "Increase monitoring", "Consider medication adjustment", "Follow-up within 1 week"],
        "high": ["1:1 observation", "Means restriction assessment", "Consider voluntary admission", "Contact crisis team", "Safety plan with family involvement"],
        "imminent": ["Immediate psychiatric evaluation", "1:1 continuous observation", "Involuntary hold if refusing treatment", "Remove all means", "Emergency department evaluation"],
    }
    return actions.get(risk_level, actions["low"])


def _check_psych_interactions(a: str, b: str) -> list:
    serotonergic = {"sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram",
                    "venlafaxine", "duloxetine", "tramadol", "trazodone", "buspirone",
                    "lithium", "fentanyl", "mirtazapine"}
    qtc_prolonging = {"citalopram", "escitalopram", "haloperidol", "ziprasidone",
                      "quetiapine", "chlorpromazine", "methadone", "amitriptyline"}
    cyp2d6_inhibitors = {"fluoxetine", "paroxetine", "bupropion"}
    cyp2d6_substrates = {"aripiprazole", "risperidone", "haloperidol", "codeine", "tramadol"}

    interactions = []
    pair = {a, b}
    if pair.issubset(serotonergic):
        interactions.append({"type": "serotonin_syndrome_risk", "severity": "high",
                             "description": f"Both {a} and {b} have serotonergic activity. Risk of serotonin syndrome."})
    if pair.issubset(qtc_prolonging):
        interactions.append({"type": "qtc_prolongation", "severity": "moderate",
                             "description": f"Both {a} and {b} can prolong QTc interval. Monitor ECG."})
    if pair & cyp2d6_inhibitors and pair & cyp2d6_substrates:
        inh = pair & cyp2d6_inhibitors
        sub = pair & cyp2d6_substrates
        interactions.append({"type": "metabolic_interaction", "severity": "moderate",
                             "description": f"CYP2D6 inhibitor ({', '.join(inh)}) may increase levels of {', '.join(sub)}."})
    if not interactions:
        interactions.append({"type": "none_found", "severity": "low",
                             "description": "No major interactions identified. Always verify with pharmacy database."})
    return interactions


def _default_guidelines(condition: str) -> dict:
    guidelines_map = {
        "mdd": {"first_line": ["SSRI (sertraline, escitalopram)", "CBT", "Combined SSRI + CBT"],
                 "second_line": ["SNRI (venlafaxine, duloxetine)", "Mirtazapine", "Bupropion"],
                 "augmentation": ["Aripiprazole", "Lithium", "Thyroid hormone"],
                 "special_considerations": ["Monitor for suicidal ideation in first 4 weeks", "Black box warning for <25yo"]},
        "gad": {"first_line": ["SSRI", "SNRI", "CBT"],
                "second_line": ["Buspirone", "Pregabalin", "Hydroxyzine"],
                "avoid": ["Long-term benzodiazepines", "Barbiturates"]},
        "ptsd": {"first_line": ["Sertraline", "Paroxetine", "CPT", "PE therapy"],
                 "second_line": ["Venlafaxine", "EMDR"], "avoid": ["Benzodiazepines"]},
        "bipolar": {"first_line": ["Lithium", "Valproate", "Quetiapine", "Lamotrigine (depression)"],
                    "avoid": ["SSRI monotherapy (risk of mania)"]},
        "schizophrenia": {"first_line": ["Second-generation antipsychotic (risperidone, aripiprazole, olanzapine)"],
                          "treatment_resistant": ["Clozapine"], "monitoring": ["Metabolic panel, weight, A1c, lipids"]},
    }
    key = condition.lower().replace(" ", "_")
    for k, v in guidelines_map.items():
        if k in key or key in k:
            return {"condition": condition, "guidelines": v}
    return {"condition": condition, "note": "Specific guidelines not found. Refer to APA Practice Guidelines."}
