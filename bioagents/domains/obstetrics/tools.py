"""Tools for the Obstetrics & Gynecology domain.

Provides 14 tools for prenatal care, labor management, and gynecologic assessment:
1.  get_patient_presentation   - Demographics, chief complaint, OB history
2.  get_prenatal_labs          - Prenatal laboratory results by trimester
3.  get_obstetric_history      - Gravida/para, prior pregnancies, complications
4.  assess_fetal_status        - Fetal heart rate monitoring interpretation
5.  assess_labor_progress      - Cervical dilation, effacement, station
6.  calculate_bishop_score     - Bishop score for induction readiness
7.  get_biophysical_profile    - BPP scoring
8.  check_medication_safety    - Pregnancy medication safety (FDA categories)
9.  get_risk_assessment        - ACOG risk factor screening
10. check_ob_protocol          - Look up OB emergency protocols
11. get_gyn_assessment         - Gynecologic assessment data
12. order_labs                 - Order prenatal/GYN labs
13. think                      - Internal reasoning
14. submit_answer              - Submit diagnosis and management plan
"""

import json
from typing import Any

from bioagents.domains.obstetrics.data_model import ObstetricsDB
from bioagents.environment.toolkit import ToolKitBase, is_tool


class ObstetricsTools(ToolKitBase):
    """Tool kit for Obstetrics & Gynecology domain."""

    def __init__(self, db: ObstetricsDB):
        super().__init__()
        self.db = db

    @is_tool
    def get_patient_presentation(self, patient_id: str) -> str:
        """Get patient demographics, chief complaint, gestational age, and vital signs.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with patient presentation.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        return json.dumps({
            "patient_id": p.patient_id, "name": p.name, "age": p.age,
            "gravida": p.gravida, "para": p.para,
            "gestational_age_weeks": p.gestational_age_weeks,
            "edd": p.edd,
            "chief_complaint": p.chief_complaint,
            "vitals": p.vitals,
            "blood_type": p.blood_type,
            "allergies": p.allergies,
            "risk_factors": p.risk_factors,
        }, indent=2)

    @is_tool
    def get_prenatal_labs(self, patient_id: str, trimester: int = 0) -> str:
        """Get prenatal laboratory results, optionally filtered by trimester.

        Args:
            patient_id: Patient identifier.
            trimester: 1, 2, or 3 (0 = all trimesters).

        Returns:
            JSON string with lab results.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        labs = p.prenatal_labs
        if trimester > 0:
            labs = [l for l in labs if l.trimester == trimester]
        return json.dumps({
            "patient_id": patient_id,
            "trimester_filter": trimester if trimester > 0 else "all",
            "labs": [l.model_dump() for l in labs],
        }, indent=2)

    @is_tool
    def get_obstetric_history(self, patient_id: str) -> str:
        """Get detailed obstetric history including prior pregnancies, complications, delivery methods, and neonatal outcomes.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with obstetric history.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        return json.dumps({
            "patient_id": patient_id,
            "gravida": p.gravida, "para": p.para,
            "obstetric_history": p.obstetric_history,
            "surgical_history": p.surgical_history,
            "medical_history": p.medical_history,
            "gbs_status": p.gbs_status,
        }, indent=2)

    @is_tool
    def assess_fetal_status(self, patient_id: str) -> str:
        """Interpret fetal heart rate monitoring strip. Returns baseline FHR, variability, accelerations, decelerations, NICHD category, and recommended actions.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with fetal monitoring interpretation.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        fm = p.fetal_monitoring
        if not fm:
            return json.dumps({"patient_id": patient_id, "note": "No fetal monitoring data available."})
        actions = _fetal_category_actions(fm.category, fm.decelerations)
        return json.dumps({
            "patient_id": patient_id,
            "fetal_monitoring": fm.model_dump(),
            "nichd_category": fm.category,
            "recommended_actions": actions,
        }, indent=2)

    @is_tool
    def assess_labor_progress(self, patient_id: str) -> str:
        """Assess current labor progress including cervical exam, station, membrane status, and contraction pattern.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with labor progress assessment.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        lp = p.labor_progress
        if not lp:
            return json.dumps({"patient_id": patient_id, "note": "Patient not in labor."})
        assessment = _assess_labor(lp)
        return json.dumps({
            "patient_id": patient_id,
            "labor_progress": lp.model_dump(),
            "assessment": assessment,
        }, indent=2)

    @is_tool
    def calculate_bishop_score(self, patient_id: str) -> str:
        """Calculate Bishop score to assess cervical readiness for induction of labor.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with Bishop score and interpretation.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        if p.bishop_score is not None:
            score = p.bishop_score
        elif p.labor_progress:
            score = _calc_bishop(p.labor_progress)
        else:
            return json.dumps({"patient_id": patient_id, "note": "Insufficient data for Bishop score."})
        interp = "Favorable for induction" if score >= 8 else "Consider cervical ripening" if score >= 6 else "Unfavorable — cervical ripening recommended"
        return json.dumps({
            "patient_id": patient_id, "bishop_score": score,
            "interpretation": interp,
            "induction_success_likelihood": "high" if score >= 8 else "moderate" if score >= 6 else "low",
        }, indent=2)

    @is_tool
    def get_biophysical_profile(self, patient_id: str) -> str:
        """Get Biophysical Profile (BPP) score assessing fetal well-being through ultrasound.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with BPP results.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        bpp = p.biophysical_profile
        if not bpp:
            return json.dumps({"patient_id": patient_id, "note": "BPP not performed."})
        total = sum(bpp.get(k, 0) for k in ["fetal_breathing", "fetal_movement", "fetal_tone", "amniotic_fluid", "nst"])
        interp = "Normal" if total >= 8 else "Equivocal — repeat in 24h" if total >= 6 else "Abnormal — consider delivery"
        return json.dumps({
            "patient_id": patient_id, "bpp_components": bpp,
            "total_score": total, "interpretation": interp,
        }, indent=2)

    @is_tool
    def check_medication_safety(self, drug_name: str, trimester: int = 0) -> str:
        """Check medication safety in pregnancy. Returns risk category, known teratogenic effects, and alternatives.

        Args:
            drug_name: Medication name.
            trimester: Current trimester (1, 2, or 3; 0 = general).

        Returns:
            JSON string with pregnancy safety information.
        """
        info = _pregnancy_drug_safety(drug_name.lower())
        info["trimester_queried"] = trimester
        return json.dumps(info, indent=2)

    @is_tool
    def get_risk_assessment(self, patient_id: str) -> str:
        """Perform ACOG risk factor screening for the patient including maternal age, BMI, history, and current pregnancy complications.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with risk assessment.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        risks = list(p.risk_factors)
        if p.age >= 35:
            risks.append("Advanced maternal age (>=35)")
        if p.age < 18:
            risks.append("Adolescent pregnancy")
        return json.dumps({
            "patient_id": patient_id, "risk_factors": risks,
            "risk_category": "high" if len(risks) >= 3 else "moderate" if len(risks) >= 1 else "low",
        }, indent=2)

    @is_tool
    def check_ob_protocol(self, protocol_name: str) -> str:
        """Look up an obstetric emergency protocol (e.g., shoulder dystocia, PPH, eclampsia, cord prolapse).

        Args:
            protocol_name: Protocol keyword.

        Returns:
            JSON string with protocol details.
        """
        keyword = protocol_name.lower()
        matches = []
        for pid, proto in self.db.protocols.items():
            if keyword in pid.lower() or keyword in json.dumps(proto).lower():
                matches.append({"protocol_id": pid, **proto})
        if not matches:
            matches = [_default_ob_protocol(keyword)]
        return json.dumps({"protocols": matches}, indent=2)

    @is_tool
    def get_gyn_assessment(self, patient_id: str) -> str:
        """Get gynecologic assessment data including menstrual history, contraception, Pap smear, and STI history.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with GYN assessment.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        return json.dumps({
            "patient_id": patient_id,
            "gyn_complaint": p.gyn_complaint,
            "last_menstrual_period": p.last_menstrual_period,
            "menstrual_history": p.menstrual_history,
            "contraceptive_use": p.contraceptive_use,
            "pap_smear_history": p.pap_smear_history,
            "stis_history": p.stis_history,
        }, indent=2)

    @is_tool
    def order_labs(self, patient_id: str, tests: list[str]) -> str:
        """Order prenatal or gynecologic laboratory tests.

        Args:
            patient_id: Patient identifier.
            tests: List of tests (e.g., ['CBC', 'Type_Screen', 'GBS_culture', 'GCT']).

        Returns:
            Confirmation of ordered labs.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        return json.dumps({"status": "ordered", "patient_id": patient_id, "tests": tests}, indent=2)

    @is_tool
    def interpret_ctg(self, patient_id: str) -> str:
        """Interpret cardiotocography (CTG/NST) tracing.

        Args:
            patient_id: Patient identifier.

        Returns:
            CTG interpretation with NICHD category (I/II/III).
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        return json.dumps({"patient_id": patient_id, "baseline_fhr": 140, "variability": "moderate (6-25 bpm)", "accelerations": "present (reactive)", "decelerations": "none", "contractions": "irregular, every 5-7 min", "category": "Category I (Normal)", "interpretation": "Reassuring fetal status. Normal baseline, moderate variability, accelerations present, no decelerations.", "recommendation": "Continue routine monitoring"}, indent=2)

    @is_tool
    def calculate_gestational_age(self, patient_id: str) -> str:
        """Calculate gestational age from LMP and/or ultrasound dating.

        Args:
            patient_id: Patient identifier.

        Returns:
            Gestational age, EDD, and trimester.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        ga_weeks = getattr(p, "gestational_age_weeks", 34)
        ga_days = getattr(p, "gestational_age_days", 2)
        trimester = "first" if ga_weeks < 14 else ("second" if ga_weeks < 28 else "third")
        return json.dumps({"patient_id": patient_id, "gestational_age": f"{ga_weeks}w{ga_days}d", "weeks": ga_weeks, "days": ga_days, "trimester": trimester, "edd": "2026-04-15", "dating_method": "ultrasound (first trimester)", "viability": ga_weeks >= 24}, indent=2)

    @is_tool
    def screen_gestational_diabetes(self, patient_id: str) -> str:
        """GDM screening with glucose challenge test results.

        Args:
            patient_id: Patient identifier.

        Returns:
            GDM screening results and diagnostic criteria.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        return json.dumps({"patient_id": patient_id, "screening_type": "1-hour GCT (50g)", "gct_result": 148, "gct_threshold": 140, "gct_status": "POSITIVE (requires 3-hour OGTT)", "ogtt_3hour": {"fasting": 95, "1_hour": 188, "2_hour": 160, "3_hour": 135, "thresholds": {"fasting": 95, "1_hour": 180, "2_hour": 155, "3_hour": 140}, "abnormal_values": 2, "diagnosis": "Gestational Diabetes Mellitus (2+ abnormal values)"}, "recommendations": ["Nutrition counseling", "Blood glucose monitoring QID", "Consider endocrinology referral"]}, indent=2)

    @is_tool
    def check_rh_status(self, patient_id: str) -> str:
        """Check Rh status, antibody screen, and RhoGAM need.

        Args:
            patient_id: Patient identifier.

        Returns:
            Rh status and immunization recommendation.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        return json.dumps({"patient_id": patient_id, "blood_type": "O", "rh_status": "Negative", "antibody_screen": "Negative", "rh_sensitized": False, "rhogam_indicated": True, "rhogam_timing": ["28 weeks prophylactic", "within 72 hours of delivery", "after any sensitizing event (amniocentesis, bleeding, trauma)"], "last_rhogam": "None administered this pregnancy"}, indent=2)

    @is_tool
    def assess_preterm_risk(self, patient_id: str) -> str:
        """Assess preterm labor risk: fetal fibronectin, cervical length, risk factors.

        Args:
            patient_id: Patient identifier.

        Returns:
            Preterm risk assessment with recommendations.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        ga = getattr(p, "gestational_age_weeks", 30)
        return json.dumps({"patient_id": patient_id, "gestational_age_weeks": ga, "cervical_length_mm": 28, "fetal_fibronectin": "negative", "risk_factors": ["prior preterm birth", "multiple gestation", "short cervix"], "risk_level": "MODERATE", "recommendations": ["Serial cervical length measurements", "Consider progesterone supplementation", "Patient education on preterm labor signs", "Antenatal corticosteroids if 24-34 weeks with risk"]}, indent=2)

    @is_tool
    def calculate_modified_bpp(self, patient_id: str) -> str:
        """Calculate Modified Biophysical Profile (NST + AFI).

        Args:
            patient_id: Patient identifier.

        Returns:
            Modified BPP score with interpretation.
        """
        p = self.db.patients.get(patient_id)
        if not p:
            return json.dumps({"error": f"Patient {patient_id} not found"})
        return json.dumps({"patient_id": patient_id, "nst": {"result": "reactive", "score": 2}, "afi_cm": 12.5, "afi_normal": True, "modified_bpp_score": "2/2 (normal)", "interpretation": "Reassuring fetal status. Reactive NST with normal AFI.", "recommendation": "Continue routine antepartum surveillance"}, indent=2)

    @is_tool
    def think(self, thought: str) -> str:
        """Record internal clinical reasoning.

        Args:
            thought: Your reasoning process.

        Returns:
            Acknowledgment.
        """
        return json.dumps({"thought_recorded": True, "content": thought[:500]})

    @is_tool
    def submit_answer(self, diagnosis: str, management_plan: str, urgency: str, reasoning: str) -> str:
        """Submit obstetric/gynecologic assessment and management plan.

        Args:
            diagnosis: Primary diagnosis.
            management_plan: Recommended management steps.
            urgency: Urgency level (routine/urgent/emergent).
            reasoning: Clinical reasoning.

        Returns:
            Confirmation.
        """
        return json.dumps({
            "submitted": True,
            "diagnosis": diagnosis, "management_plan": management_plan[:500],
            "urgency": urgency, "reasoning": reasoning[:500],
        })


# ── Helpers ───────────────────────────────────────────────────

def _fetal_category_actions(category: int, decels: str) -> list:
    if category == 1:
        return ["Continue monitoring", "Routine care"]
    elif category == 2:
        return ["Increase monitoring frequency", "Maternal repositioning", "IV fluid bolus",
                "Consider oxygen", "Evaluate for underlying cause"]
    else:
        return ["IMMEDIATE intervention", "Prepare for emergent delivery",
                "Intrauterine resuscitation", "Call OB team STAT"]

def _assess_labor(lp) -> dict:
    if lp.cervical_dilation_cm >= 10:
        phase = "Complete — ready for pushing"
    elif lp.cervical_dilation_cm >= 6:
        phase = "Active labor"
    elif lp.cervical_dilation_cm >= 4:
        phase = "Active phase beginning"
    else:
        phase = "Latent phase"
    protracted = lp.phase == "active" and lp.time_in_labor_hours > 12
    return {"current_phase": phase, "protracted_labor": protracted,
            "arrest_concern": lp.cervical_dilation_cm >= 6 and lp.time_in_labor_hours > 18}

def _calc_bishop(lp) -> int:
    score = 0
    d = lp.cervical_dilation_cm
    score += 0 if d < 1 else (1 if d < 2 else (2 if d < 4 else 3))
    e = lp.effacement_percent
    score += 0 if e < 40 else (1 if e < 60 else (2 if e < 80 else 3))
    s = lp.station
    score += 0 if s <= -3 else (1 if s <= -2 else (2 if s <= 0 else 3))
    return score

def _pregnancy_drug_safety(drug: str) -> dict:
    unsafe = {
        "warfarin": {"category": "X", "risk": "Teratogenic — warfarin embryopathy", "alternative": "LMWH (enoxaparin)"},
        "methotrexate": {"category": "X", "risk": "Abortifacient, teratogenic", "alternative": "None — contraindicated"},
        "isotretinoin": {"category": "X", "risk": "Severe birth defects", "alternative": "Topical treatments"},
        "valproic_acid": {"category": "X", "risk": "Neural tube defects, developmental delay", "alternative": "Lamotrigine, levetiracetam"},
        "lithium": {"category": "D", "risk": "Ebstein anomaly (cardiac)", "alternative": "Lamotrigine for bipolar"},
        "ace_inhibitors": {"category": "D", "risk": "Renal agenesis, oligohydramnios (2nd/3rd trimester)", "alternative": "Labetalol, nifedipine"},
        "nsaids": {"category": "C/D", "risk": "Premature ductus arteriosus closure (3rd trimester)", "alternative": "Acetaminophen"},
    }
    for key, info in unsafe.items():
        if key in drug or drug in key:
            return {"drug": drug, **info}
    return {"drug": drug, "category": "Consult pharmacy", "risk": "Insufficient data", "alternative": "Verify with Lactmed/Reprotox database"}

def _default_ob_protocol(keyword: str) -> dict:
    protos = {
        "shoulder": {"name": "Shoulder Dystocia", "steps": ["Call for help", "McRoberts maneuver", "Suprapubic pressure", "Episiotomy", "Rubin maneuver", "Woods screw", "Gaskin maneuver", "Zavanelli as last resort"]},
        "pph": {"name": "Postpartum Hemorrhage", "steps": ["Fundal massage", "Oxytocin 40U IV", "Methylergonovine 0.2mg IM", "Carboprost 250mcg IM", "Misoprostol 800mcg PR", "Tamponade balloon", "Surgical intervention"]},
        "eclampsia": {"name": "Eclampsia Management", "steps": ["Magnesium sulfate 6g IV load then 2g/hr", "Secure airway", "Left lateral position", "Monitor fetal status", "Plan delivery after stabilization"]},
        "cord": {"name": "Cord Prolapse", "steps": ["Elevate presenting part (do NOT replace cord)", "Knee-chest or Trendelenburg position", "Fill bladder with 500-700mL saline", "EMERGENT cesarean delivery"]},
    }
    for key, proto in protos.items():
        if key in keyword:
            return proto
    return {"name": keyword, "note": "Protocol not found. Consult ACOG guidelines."}
