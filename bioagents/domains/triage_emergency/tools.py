"""Tools for the Triage & Emergency domain.

Provides 20 tools for emergency department triage and initial management:
1.  get_patient_presentation    — Chief complaint, vitals, symptoms
2.  get_vital_signs             — Detailed vital signs
3.  assess_airway_breathing     — ABC assessment
4.  get_medical_history         — PMH, medications, allergies
5.  calculate_gcs               — Glasgow Coma Scale
6.  calculate_esi_level         — ESI algorithm support
7.  get_ed_status               — ED occupancy and resources
8.  check_protocol              — Look up emergency protocol
9.  order_stat_labs             — Order STAT laboratory tests
10. order_imaging               — Order STAT imaging
11. calculate_sirs_criteria     — SIRS criteria evaluation
12. calculate_qsofa             — Quick SOFA score
13. screen_sepsis               — Combined sepsis screening
14. calculate_trauma_score      — Revised Trauma Score (RTS)
15. check_toxicology            — Toxicology panel and toxidromes
16. calculate_heart_score       — HEART score for chest pain
17. activate_emergency_protocol — Activate specific emergency protocol
18. get_bed_assignment          — Bed/area assignment by acuity
19. think                       — Internal reasoning
20. submit_answer               — Submit triage decision
"""

import json
from typing import Any

from bioagents.domains.triage_emergency.data_model import TriageEmergencyDB
from bioagents.environment.toolkit import ToolKitBase, is_tool


class TriageEmergencyTools(ToolKitBase):
    """Tool kit for Triage & Emergency domain."""

    def __init__(self, db: TriageEmergencyDB):
        super().__init__()
        self.db = db

    @is_tool
    def get_patient_presentation(self, patient_id: str) -> str:
        """Get the initial presentation of an ED patient including chief complaint, arrival mode, onset, vitals, symptoms, and pain score.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with patient presentation data.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        return json.dumps({
            "patient_id": patient.patient_id,
            "age": patient.age,
            "sex": patient.sex,
            "chief_complaint": patient.chief_complaint,
            "arrival_mode": patient.arrival_mode,
            "onset_time": patient.onset_time,
            "presenting_symptoms": patient.presenting_symptoms,
            "pain_score": patient.pain_score,
            "triage_vitals": patient.triage_vitals,
        }, indent=2)

    @is_tool
    def get_vital_signs(self, patient_id: str) -> str:
        """Get detailed vital signs for the patient.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with vital signs and interpretation.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        vitals = patient.triage_vitals
        interpretation = _interpret_vitals(vitals, patient.age)

        return json.dumps({
            "patient_id": patient_id,
            "vitals": vitals,
            "interpretation": interpretation,
        }, indent=2)

    @is_tool
    def assess_airway_breathing(self, patient_id: str) -> str:
        """Perform ABC (Airway, Breathing, Circulation) assessment.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with ABC assessment.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        vitals = patient.triage_vitals
        gcs = patient.gcs or 15

        airway = "patent" if gcs > 8 else "compromised - needs intervention"
        breathing_rate = vitals.get("respiratory_rate", 16)
        spo2 = vitals.get("spo2", 98)

        if breathing_rate < 8 or breathing_rate > 30 or spo2 < 90:
            breathing = "abnormal - immediate intervention needed"
        elif breathing_rate > 22 or spo2 < 94:
            breathing = "concerning - monitor closely"
        else:
            breathing = "adequate"

        sbp = vitals.get("systolic_bp", 120)
        hr = vitals.get("heart_rate", 80)
        circulation = "stable"
        if sbp < 90 or hr > 120:
            circulation = "unstable - hemodynamic compromise"
        elif sbp < 100 or hr > 100:
            circulation = "borderline - close monitoring"

        return json.dumps({
            "patient_id": patient_id,
            "airway": airway,
            "breathing": breathing,
            "circulation": circulation,
            "gcs": gcs,
            "overall_stability": "unstable" if "unstable" in circulation or "compromised" in airway else "stable",
        }, indent=2)

    @is_tool
    def get_medical_history(self, patient_id: str) -> str:
        """Get patient's medical history, medications, and allergies.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with medical history.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        return json.dumps({
            "patient_id": patient_id,
            "medical_history": patient.medical_history,
            "current_medications": patient.medications,
            "allergies": patient.allergies,
        }, indent=2)

    @is_tool
    def calculate_gcs(self, patient_id: str) -> str:
        """Calculate or retrieve the Glasgow Coma Scale score.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with GCS components and total.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        gcs = patient.gcs or 15
        # Approximate component breakdown
        if gcs >= 14:
            eye, verbal, motor = 4, 5, 6
            interpretation = "Normal / fully alert"
        elif gcs >= 9:
            eye, verbal, motor = 3, 4, min(gcs - 7, 6)
            interpretation = "Moderate impairment"
        else:
            eye, verbal, motor = max(1, gcs - 5), max(1, min(gcs - 3, 3)), max(1, min(gcs - 2, 4))
            interpretation = "Severe impairment — protect airway"

        return json.dumps({
            "patient_id": patient_id,
            "gcs_total": gcs,
            "components": {"eye": eye, "verbal": verbal, "motor": motor},
            "interpretation": interpretation,
        }, indent=2)

    @is_tool
    def calculate_esi_level(self, patient_id: str) -> str:
        """Apply ESI (Emergency Severity Index) triage algorithm.

        Provides structured ESI decision support:
        - ESI 1: Immediate life-saving intervention needed
        - ESI 2: High-risk situation, confused/lethargic, severe pain
        - ESI 3: Multiple resources needed, stable
        - ESI 4: One resource needed
        - ESI 5: No resources needed

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with ESI assessment factors.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        vitals = patient.triage_vitals
        gcs = patient.gcs or 15

        # ESI decision factors
        factors = {
            "requires_immediate_intervention": gcs <= 8 or vitals.get("systolic_bp", 120) < 70,
            "high_risk_situation": _is_high_risk(patient),
            "vital_sign_danger_zone": _in_danger_zone(vitals, patient.age),
            "estimated_resources_needed": _estimate_resources(patient),
            "pain_score": patient.pain_score,
            "gcs": gcs,
        }

        # Suggested ESI
        if factors["requires_immediate_intervention"]:
            suggested_esi = 1
            reasoning = "Immediate life-saving intervention required"
        elif factors["high_risk_situation"] or factors["vital_sign_danger_zone"]:
            suggested_esi = 2
            reasoning = "High-risk or vital signs in danger zone"
        elif factors["estimated_resources_needed"] >= 2:
            suggested_esi = 3
            reasoning = f"Stable, but needs {factors['estimated_resources_needed']} resources"
        elif factors["estimated_resources_needed"] == 1:
            suggested_esi = 4
            reasoning = "Needs one resource"
        else:
            suggested_esi = 5
            reasoning = "No resources needed, minor complaint"

        return json.dumps({
            "patient_id": patient_id,
            "esi_factors": factors,
            "suggested_esi_level": suggested_esi,
            "reasoning": reasoning,
            "note": "Final ESI determination should incorporate clinical judgment",
        }, indent=2)

    @is_tool
    def get_ed_status(self) -> str:
        """Get current Emergency Department operational status including bed availability, wait times, and resource status.

        Returns:
            JSON string with ED status.
        """
        status = self.db.ed_status
        resources = {
            rid: {
                "type": r.resource_type,
                "available": r.current_available,
                "total": r.total_capacity,
                "wait_minutes": r.wait_time_minutes,
            }
            for rid, r in self.db.resources.items()
        }

        return json.dumps({
            "ed_status": {
                "total_patients": status.total_patients,
                "waiting_patients": status.waiting_patients,
                "beds_available": status.beds_available,
                "beds_total": status.beds_total,
                "average_wait_minutes": status.average_wait_minutes,
                "diversion_status": status.diversion_status,
                "pending_admissions": status.pending_admissions,
            },
            "resources": resources,
        }, indent=2)

    @is_tool
    def check_protocol(self, protocol_name: str) -> str:
        """Look up an emergency protocol or order set by name.

        Args:
            protocol_name: Name or keyword (e.g., 'STEMI', 'stroke', 'sepsis').

        Returns:
            JSON string with matching protocol details.
        """
        keyword = protocol_name.lower()
        matches = []

        for pid, proto in self.db.protocols.items():
            if keyword in proto.name.lower() or any(keyword in tc.lower() for tc in proto.trigger_conditions):
                matches.append({
                    "protocol_id": pid,
                    "name": proto.name,
                    "trigger_conditions": proto.trigger_conditions,
                    "immediate_actions": proto.immediate_actions,
                    "medications": proto.medications,
                    "diagnostics": proto.diagnostics,
                    "time_critical": proto.time_critical,
                    "time_target_minutes": proto.time_target_minutes,
                })

        if not matches:
            return json.dumps({"message": f"No protocol found for '{protocol_name}'"})

        return json.dumps({"protocols": matches}, indent=2)

    @is_tool
    def order_stat_labs(self, patient_id: str, tests: list[str]) -> str:
        """Order STAT laboratory tests for a patient.

        Args:
            patient_id: Patient identifier.
            tests: List of test names (e.g., ['CBC', 'BMP', 'troponin', 'lactate']).

        Returns:
            JSON string confirming ordered tests.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        return json.dumps({
            "status": "ordered",
            "patient_id": patient_id,
            "tests_ordered": tests,
            "priority": "STAT",
            "estimated_turnaround": "15-30 minutes for most tests",
        }, indent=2)

    @is_tool
    def order_imaging(self, patient_id: str, modality: str, body_part: str, indication: str = "") -> str:
        """Order STAT imaging for a patient.

        Args:
            patient_id: Patient identifier.
            modality: Imaging type (e.g., 'CT', 'X-ray', 'ultrasound', 'MRI').
            body_part: Body part (e.g., 'head', 'chest', 'abdomen').
            indication: Clinical indication.

        Returns:
            JSON string confirming imaging order.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        return json.dumps({
            "status": "ordered",
            "patient_id": patient_id,
            "modality": modality,
            "body_part": body_part,
            "indication": indication,
            "priority": "STAT",
            "estimated_wait": "10-20 minutes" if modality != "MRI" else "60-90 minutes",
        }, indent=2)

    @is_tool
    def calculate_sirs_criteria(self, patient_id: str) -> str:
        """Check Systemic Inflammatory Response Syndrome (SIRS) criteria.

        Evaluates four SIRS criteria:
        - Temperature >38°C or <36°C
        - Heart rate >90 bpm
        - Respiratory rate >20 breaths/min
        - WBC >12,000/mm³ or <4,000/mm³

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with criteria met and total SIRS score (0-4).
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        vitals = patient.triage_vitals
        temp = vitals.get("temperature", 37.0)
        hr = vitals.get("heart_rate", 80)
        rr = vitals.get("respiratory_rate", 16)
        wbc = vitals.get("wbc", 8.0)

        criteria = {
            "temperature": {
                "value": temp,
                "unit": "°C",
                "met": temp > 38.0 or temp < 36.0,
                "detail": ">38°C or <36°C",
            },
            "heart_rate": {
                "value": hr,
                "unit": "bpm",
                "met": hr > 90,
                "detail": ">90 bpm",
            },
            "respiratory_rate": {
                "value": rr,
                "unit": "breaths/min",
                "met": rr > 20,
                "detail": ">20 breaths/min",
            },
            "wbc": {
                "value": wbc,
                "unit": "x10³/mm³",
                "met": wbc > 12.0 or wbc < 4.0,
                "detail": ">12K or <4K /mm³",
            },
        }

        score = sum(1 for c in criteria.values() if c["met"])
        sirs_positive = score >= 2

        return json.dumps({
            "patient_id": patient_id,
            "sirs_criteria": criteria,
            "sirs_score": score,
            "sirs_positive": sirs_positive,
            "interpretation": f"SIRS {'POSITIVE' if sirs_positive else 'NEGATIVE'} — {score}/4 criteria met",
        }, indent=2)

    @is_tool
    def calculate_qsofa(self, patient_id: str) -> str:
        """Calculate quick SOFA (qSOFA) score for sepsis screening.

        Evaluates three criteria (1 point each):
        - Altered mentation (GCS <15)
        - Systolic blood pressure ≤100 mmHg
        - Respiratory rate ≥22 breaths/min

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with qSOFA score (0-3) and sepsis screening result.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        vitals = patient.triage_vitals
        gcs = patient.gcs or 15
        sbp = vitals.get("systolic_bp", 120)
        rr = vitals.get("respiratory_rate", 16)

        criteria = {
            "altered_mentation": {
                "value": gcs,
                "met": gcs < 15,
                "detail": "GCS <15",
            },
            "systolic_bp": {
                "value": sbp,
                "unit": "mmHg",
                "met": sbp <= 100,
                "detail": "SBP ≤100 mmHg",
            },
            "respiratory_rate": {
                "value": rr,
                "unit": "breaths/min",
                "met": rr >= 22,
                "detail": "RR ≥22 breaths/min",
            },
        }

        score = sum(1 for c in criteria.values() if c["met"])
        high_risk = score >= 2

        return json.dumps({
            "patient_id": patient_id,
            "qsofa_criteria": criteria,
            "qsofa_score": score,
            "high_risk": high_risk,
            "interpretation": (
                "HIGH RISK — qSOFA ≥2, assess for organ dysfunction"
                if high_risk
                else f"qSOFA {score}/3 — continue monitoring"
            ),
        }, indent=2)

    @is_tool
    def screen_sepsis(self, patient_id: str) -> str:
        """Perform combined sepsis screening using SIRS, qSOFA, and suspected infection indicators.

        Integrates SIRS criteria, qSOFA score, and clinical suspicion for infection
        to provide an overall sepsis risk level.

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with combined sepsis screening result and risk level.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        vitals = patient.triage_vitals
        gcs = patient.gcs or 15
        temp = vitals.get("temperature", 37.0)
        hr = vitals.get("heart_rate", 80)
        rr = vitals.get("respiratory_rate", 16)
        sbp = vitals.get("systolic_bp", 120)
        wbc = vitals.get("wbc", 8.0)
        lactate = vitals.get("lactate", 1.0)

        sirs_count = sum([
            temp > 38.0 or temp < 36.0,
            hr > 90,
            rr > 20,
            wbc > 12.0 or wbc < 4.0,
        ])

        qsofa_count = sum([
            gcs < 15,
            sbp <= 100,
            rr >= 22,
        ])

        cc = patient.chief_complaint.lower()
        infection_keywords = ["fever", "infection", "cough", "dysuria", "cellulitis",
                              "abscess", "pneumonia", "uti", "sepsis", "wound"]
        suspected_infection = any(kw in cc for kw in infection_keywords) or temp > 38.0 or temp < 36.0

        if qsofa_count >= 2 and suspected_infection and (lactate > 2.0 or sbp < 90):
            risk_level = "CRITICAL — possible septic shock"
            recommendation = "Activate sepsis bundle immediately; obtain lactate, blood cultures; start IV fluids and broad-spectrum antibiotics within 1 hour"
        elif qsofa_count >= 2 and suspected_infection:
            risk_level = "HIGH — probable sepsis"
            recommendation = "Obtain blood cultures, lactate, CBC; initiate IV fluids and empiric antibiotics within 1 hour"
        elif sirs_count >= 2 and suspected_infection:
            risk_level = "MODERATE — possible sepsis"
            recommendation = "Obtain blood cultures, lactate, CBC; monitor closely for deterioration"
        elif suspected_infection:
            risk_level = "LOW — infection suspected, no sepsis criteria met"
            recommendation = "Standard workup; monitor vitals and reassess if condition changes"
        else:
            risk_level = "UNLIKELY — no infection indicators"
            recommendation = "Sepsis unlikely based on current presentation; consider alternative diagnoses"

        return json.dumps({
            "patient_id": patient_id,
            "sirs_score": sirs_count,
            "qsofa_score": qsofa_count,
            "suspected_infection": suspected_infection,
            "lactate": lactate,
            "risk_level": risk_level,
            "recommendation": recommendation,
        }, indent=2)

    @is_tool
    def calculate_trauma_score(self, patient_id: str, mechanism: str) -> str:
        """Calculate the Revised Trauma Score (RTS) based on GCS, systolic BP, and respiratory rate.

        The RTS is used to quantify injury severity and guide trauma center triage.

        Args:
            patient_id: Patient identifier.
            mechanism: Mechanism of injury (e.g., 'MVC', 'fall', 'penetrating', 'blunt').

        Returns:
            JSON string with RTS value and trauma center criteria.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        vitals = patient.triage_vitals
        gcs = patient.gcs or 15
        sbp = vitals.get("systolic_bp", 120)
        rr = vitals.get("respiratory_rate", 16)

        if gcs >= 13:
            gcs_coded = 4
        elif gcs >= 9:
            gcs_coded = 3
        elif gcs >= 6:
            gcs_coded = 2
        elif gcs >= 4:
            gcs_coded = 1
        else:
            gcs_coded = 0

        if sbp > 89:
            sbp_coded = 4
        elif sbp >= 76:
            sbp_coded = 3
        elif sbp >= 50:
            sbp_coded = 2
        elif sbp >= 1:
            sbp_coded = 1
        else:
            sbp_coded = 0

        if 10 <= rr <= 29:
            rr_coded = 4
        elif rr > 29:
            rr_coded = 3
        elif 6 <= rr <= 9:
            rr_coded = 2
        elif 1 <= rr <= 5:
            rr_coded = 1
        else:
            rr_coded = 0

        rts = round(0.9368 * gcs_coded + 0.7326 * sbp_coded + 0.2908 * rr_coded, 4)

        high_risk_mechanisms = ["mvc high speed", "penetrating", "ejection", "pedestrian struck",
                                "motorcycle", "fall >20 feet", "fall >6 meters"]
        mechanism_high_risk = any(m in mechanism.lower() for m in high_risk_mechanisms)

        if rts < 4.0:
            trauma_center = "Level I Trauma Center — critical injury"
        elif rts < 7.84 or mechanism_high_risk:
            trauma_center = "Level I or II Trauma Center — significant injury"
        else:
            trauma_center = "Local facility may be appropriate — reassess if condition changes"

        return json.dumps({
            "patient_id": patient_id,
            "mechanism_of_injury": mechanism,
            "gcs": gcs,
            "systolic_bp": sbp,
            "respiratory_rate": rr,
            "coded_values": {"gcs": gcs_coded, "sbp": sbp_coded, "rr": rr_coded},
            "rts": rts,
            "mechanism_high_risk": mechanism_high_risk,
            "trauma_center_recommendation": trauma_center,
        }, indent=2)

    @is_tool
    def check_toxicology(self, patient_id: str) -> str:
        """Get toxicology panel results and evaluate for common toxidromes.

        Returns available tox screen results and maps findings to recognized
        toxidrome patterns (sympathomimetic, anticholinergic, opioid, sedative-hypnotic,
        cholinergic, serotonin syndrome).

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with toxicology results and suspected toxidromes.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        vitals = patient.triage_vitals
        gcs = patient.gcs or 15
        hr = vitals.get("heart_rate", 80)
        rr = vitals.get("respiratory_rate", 16)
        temp = vitals.get("temperature", 37.0)
        sbp = vitals.get("systolic_bp", 120)

        tox_screen = vitals.get("tox_screen", {
            "acetaminophen": "negative",
            "salicylate": "negative",
            "ethanol": "negative",
            "urine_drug_screen": "pending",
        })

        suspected_toxidromes = []

        if hr > 100 and temp > 38.0 and sbp > 140:
            suspected_toxidromes.append({
                "toxidrome": "Sympathomimetic",
                "features": "Tachycardia, hypertension, hyperthermia",
                "common_agents": ["cocaine", "amphetamines", "methamphetamine", "MDMA"],
            })
        if hr > 100 and temp > 38.0 and gcs < 15:
            suspected_toxidromes.append({
                "toxidrome": "Anticholinergic",
                "features": "Tachycardia, hyperthermia, altered mentation, dry skin",
                "common_agents": ["diphenhydramine", "tricyclic antidepressants", "atropine", "jimson weed"],
            })
        if rr < 12 and gcs < 12:
            suspected_toxidromes.append({
                "toxidrome": "Opioid",
                "features": "Respiratory depression, decreased consciousness, miosis",
                "common_agents": ["heroin", "fentanyl", "oxycodone", "morphine"],
                "antidote": "Naloxone 0.4-2mg IV/IM/IN",
            })
        if rr < 12 and gcs < 15 and hr < 70:
            suspected_toxidromes.append({
                "toxidrome": "Sedative-Hypnotic",
                "features": "CNS depression, respiratory depression, hypotension",
                "common_agents": ["benzodiazepines", "barbiturates", "ethanol", "GHB"],
                "antidote": "Flumazenil (for benzodiazepines, use with caution)",
            })
        if hr < 60 and rr > 20 and sbp < 100:
            suspected_toxidromes.append({
                "toxidrome": "Cholinergic",
                "features": "Bradycardia, salivation, lacrimation, urination, diaphoresis (SLUDGE)",
                "common_agents": ["organophosphates", "carbamates", "nerve agents"],
                "antidote": "Atropine + Pralidoxime",
            })
        if hr > 100 and temp > 38.5 and gcs < 15:
            suspected_toxidromes.append({
                "toxidrome": "Serotonin Syndrome",
                "features": "Tachycardia, hyperthermia, agitation, clonus, hyperreflexia",
                "common_agents": ["SSRIs", "SNRIs", "MAOIs", "tramadol", "linezolid"],
                "antidote": "Cyproheptadine",
            })

        return json.dumps({
            "patient_id": patient_id,
            "tox_screen": tox_screen,
            "suspected_toxidromes": suspected_toxidromes if suspected_toxidromes else "No classic toxidrome pattern identified",
            "recommendation": "Correlate with history, physical exam, and additional labs (metabolic panel, osmolality, ECG)" if suspected_toxidromes else "Tox screen pending; monitor clinically",
        }, indent=2)

    @is_tool
    def calculate_heart_score(self, patient_id: str) -> str:
        """Calculate the HEART score for chest pain evaluation.

        Evaluates five components (0-2 points each):
        - History (slightly suspicious, moderately suspicious, highly suspicious)
        - ECG (normal, non-specific repolarization, significant ST deviation)
        - Age (<45, 45-64, ≥65)
        - Risk factors (0-1, 2, ≥3 of HTN, DM, hyperlipidemia, obesity, smoking, family hx)
        - Troponin (≤normal, 1-3x normal, >3x normal)

        Args:
            patient_id: Patient identifier.

        Returns:
            JSON string with HEART score (0-10) and risk category.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        vitals = patient.triage_vitals
        age = patient.age
        cc = patient.chief_complaint.lower()
        history = patient.medical_history or []
        history_lower = [h.lower() for h in history]

        if "crushing" in cc or "radiating" in cc or "exertional" in cc:
            history_score = 2
            history_detail = "Highly suspicious"
        elif "chest pain" in cc or "pressure" in cc:
            history_score = 1
            history_detail = "Moderately suspicious"
        else:
            history_score = 0
            history_detail = "Slightly suspicious"

        ecg_result = vitals.get("ecg", "normal")
        if "st_elevation" in ecg_result or "st_depression" in ecg_result:
            ecg_score = 2
            ecg_detail = "Significant ST deviation"
        elif ecg_result != "normal":
            ecg_score = 1
            ecg_detail = "Non-specific repolarization disturbance"
        else:
            ecg_score = 0
            ecg_detail = "Normal"

        if age >= 65:
            age_score = 2
        elif age >= 45:
            age_score = 1
        else:
            age_score = 0

        risk_factors_list = ["hypertension", "diabetes", "hyperlipidemia",
                             "obesity", "smoking", "family history"]
        risk_count = sum(1 for rf in risk_factors_list if any(rf in h for h in history_lower))
        if risk_count >= 3:
            risk_score = 2
        elif risk_count >= 1:
            risk_score = 1
        else:
            risk_score = 0

        troponin = vitals.get("troponin", 0.01)
        troponin_upper_normal = 0.04
        if troponin > 3 * troponin_upper_normal:
            troponin_score = 2
            troponin_detail = f">3x upper limit of normal ({troponin} ng/mL)"
        elif troponin > troponin_upper_normal:
            troponin_score = 1
            troponin_detail = f"1-3x upper limit of normal ({troponin} ng/mL)"
        else:
            troponin_score = 0
            troponin_detail = f"≤ normal ({troponin} ng/mL)"

        total = history_score + ecg_score + age_score + risk_score + troponin_score

        if total <= 3:
            risk_category = "LOW risk (1.7% MACE) — consider early discharge"
        elif total <= 6:
            risk_category = "MODERATE risk (16.6% MACE) — observation and further workup"
        else:
            risk_category = "HIGH risk (50.1% MACE) — early invasive strategy recommended"

        return json.dumps({
            "patient_id": patient_id,
            "heart_score_components": {
                "history": {"score": history_score, "detail": history_detail},
                "ecg": {"score": ecg_score, "detail": ecg_detail},
                "age": {"score": age_score, "value": age},
                "risk_factors": {"score": risk_score, "count": risk_count},
                "troponin": {"score": troponin_score, "detail": troponin_detail},
            },
            "heart_score_total": total,
            "risk_category": risk_category,
        }, indent=2)

    @is_tool
    def activate_emergency_protocol(self, patient_id: str, protocol_type: str) -> str:
        """Activate a specific emergency protocol with time-critical steps.

        Supported protocol types:
        - stroke_code: Acute stroke pathway
        - stemi_code: ST-elevation MI pathway
        - trauma_code: Trauma team activation
        - sepsis_bundle: Sepsis management bundle (SEP-1)
        - massive_transfusion: Massive transfusion protocol

        Args:
            patient_id: Patient identifier.
            protocol_type: One of 'stroke_code', 'stemi_code', 'trauma_code', 'sepsis_bundle', 'massive_transfusion'.

        Returns:
            JSON string with activation steps and time targets.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        protocols = {
            "stroke_code": {
                "name": "Acute Stroke Code",
                "time_targets": {
                    "CT_head": "25 minutes from arrival",
                    "CT_interpretation": "45 minutes from arrival",
                    "tPA_administration": "60 minutes from arrival (if eligible)",
                    "interventional_radiology_notification": "Immediate if LVO suspected",
                },
                "immediate_actions": [
                    "Notify stroke team and neurology",
                    "Establish IV access — 2 large bore IVs",
                    "STAT non-contrast CT head",
                    "Obtain blood glucose, CBC, BMP, coagulation studies",
                    "Determine last known well time",
                    "Calculate NIHSS score",
                    "Hold anticoagulants and antiplatelets until CT reviewed",
                ],
                "key_decision_points": [
                    "tPA eligibility (within 4.5 hours of onset)",
                    "Large vessel occlusion — consider thrombectomy (within 24 hours)",
                ],
            },
            "stemi_code": {
                "name": "STEMI Code — Cardiac Catheterization Activation",
                "time_targets": {
                    "ECG_acquisition": "10 minutes from arrival",
                    "cath_lab_activation": "Immediate upon STEMI recognition",
                    "door_to_balloon": "90 minutes or less",
                    "first_medical_contact_to_device": "120 minutes if transfer needed",
                },
                "immediate_actions": [
                    "12-lead ECG — STAT",
                    "Activate cardiac cath lab",
                    "Aspirin 325mg PO (if not already given)",
                    "Heparin per protocol",
                    "Clopidogrel/Ticagrelor loading dose per cardiology",
                    "Serial troponins",
                    "Establish 2 large bore IVs",
                    "Continuous telemetry monitoring",
                ],
                "key_decision_points": [
                    "Primary PCI vs fibrinolysis (if PCI not available within 120 min)",
                    "Cardiogenic shock assessment — consider mechanical support",
                ],
            },
            "trauma_code": {
                "name": "Trauma Team Activation",
                "time_targets": {
                    "primary_survey": "Within 5 minutes of arrival",
                    "FAST_exam": "Within 10 minutes",
                    "CT_pan_scan": "Within 30 minutes if hemodynamically stable",
                    "OR_availability": "Notify OR within 15 minutes if surgical emergency",
                },
                "immediate_actions": [
                    "Trauma team to bedside — full team activation",
                    "Primary survey (ABCDE)",
                    "C-spine immobilization until cleared",
                    "2 large bore IVs — initiate crystalloid",
                    "Type and crossmatch — 4 units pRBC",
                    "FAST exam",
                    "Chest and pelvis X-ray",
                    "Foley catheter and NG/OG tube as indicated",
                ],
                "key_decision_points": [
                    "Hemodynamic instability — OR vs interventional radiology",
                    "Massive transfusion protocol if indicated",
                    "Neurosurgical consultation if head injury suspected",
                ],
            },
            "sepsis_bundle": {
                "name": "SEP-1 Sepsis Management Bundle",
                "time_targets": {
                    "lactate_measurement": "Within 30 minutes",
                    "blood_cultures": "Before antibiotics, within 30 minutes",
                    "broad_spectrum_antibiotics": "Within 1 hour of recognition",
                    "iv_fluid_bolus": "30 mL/kg crystalloid within 3 hours if hypotensive or lactate ≥4",
                    "repeat_lactate": "Within 6 hours if initial lactate elevated",
                    "vasopressors": "If hypotensive after fluid resuscitation",
                },
                "immediate_actions": [
                    "Obtain blood cultures (2 sets from different sites)",
                    "STAT lactate level",
                    "Start broad-spectrum IV antibiotics",
                    "IV crystalloid 30 mL/kg bolus if SBP <90 or lactate ≥4",
                    "Continuous vital sign monitoring (every 15 minutes)",
                    "Foley catheter — monitor urine output",
                    "Obtain CBC, BMP, hepatic panel, coagulation studies",
                ],
                "key_decision_points": [
                    "Source control — imaging and/or surgical consultation",
                    "ICU admission if vasopressors needed or persistent hypotension",
                    "Refractory shock — consider stress-dose steroids",
                ],
            },
            "massive_transfusion": {
                "name": "Massive Transfusion Protocol (MTP)",
                "time_targets": {
                    "blood_bank_notification": "Immediate",
                    "first_pack_available": "Within 10 minutes",
                    "ratio_target": "1:1:1 pRBC:FFP:platelets",
                    "reassessment": "After every 4-6 units pRBC",
                },
                "immediate_actions": [
                    "Notify blood bank — activate MTP",
                    "Type and crossmatch — issue uncrossmatched O-negative if critical",
                    "Establish 2 large bore IVs or central line",
                    "Initiate 1:1:1 transfusion ratio (pRBC:FFP:Platelets)",
                    "Administer tranexamic acid (TXA) 1g IV if within 3 hours of injury",
                    "Warm all blood products",
                    "Obtain baseline CBC, coagulation studies, fibrinogen, ionized calcium",
                    "Repeat labs after every 4-6 units",
                ],
                "key_decision_points": [
                    "Surgical control of hemorrhage — OR consultation",
                    "Monitor for transfusion complications (hypothermia, hypocalcemia, hyperkalemia)",
                    "Consider cryoprecipitate if fibrinogen <150 mg/dL",
                ],
            },
        }

        protocol = protocols.get(protocol_type)
        if not protocol:
            return json.dumps({
                "error": f"Unknown protocol type '{protocol_type}'",
                "available_protocols": list(protocols.keys()),
            })

        return json.dumps({
            "patient_id": patient_id,
            "protocol_activated": True,
            "protocol_type": protocol_type,
            "protocol_name": protocol["name"],
            "time_targets": protocol["time_targets"],
            "immediate_actions": protocol["immediate_actions"],
            "key_decision_points": protocol["key_decision_points"],
        }, indent=2)

    @is_tool
    def get_bed_assignment(self, patient_id: str, acuity: int) -> str:
        """Get recommended bed or area assignment based on ESI acuity level.

        Assigns patients to appropriate ED zones:
        - ESI 1: Resuscitation bay
        - ESI 2: Acute care / monitored bed
        - ESI 3: Acute care
        - ESI 4: Fast-track / urgent care area
        - ESI 5: Fast-track / waiting area

        Args:
            patient_id: Patient identifier.
            acuity: ESI acuity level (1-5).

        Returns:
            JSON string with bed/area assignment and monitoring requirements.
        """
        patient = self.db.patients.get(patient_id)
        if not patient:
            return json.dumps({"error": f"Patient {patient_id} not found"})

        if acuity not in range(1, 6):
            return json.dumps({"error": f"Invalid acuity level {acuity}. Must be 1-5."})

        status = self.db.ed_status
        beds_available = status.beds_available if status else 5

        assignments = {
            1: {
                "zone": "Resuscitation Bay",
                "bed_type": "Resuscitation / trauma bay",
                "monitoring": "Continuous cardiac monitoring, pulse oximetry, ETCO2",
                "nurse_ratio": "1:1",
                "reassessment_interval": "Continuous",
                "requirements": ["Crash cart at bedside", "Airway equipment ready", "Defibrillator on standby"],
            },
            2: {
                "zone": "Acute Care — Monitored",
                "bed_type": "Monitored acute care bed",
                "monitoring": "Continuous cardiac monitoring, pulse oximetry",
                "nurse_ratio": "1:2",
                "reassessment_interval": "Every 15-30 minutes",
                "requirements": ["Telemetry monitoring", "IV access established"],
            },
            3: {
                "zone": "Acute Care",
                "bed_type": "Standard acute care bed",
                "monitoring": "Intermittent vital signs",
                "nurse_ratio": "1:3-4",
                "reassessment_interval": "Every 60 minutes",
                "requirements": ["Standard monitoring equipment"],
            },
            4: {
                "zone": "Fast-Track",
                "bed_type": "Fast-track exam room or chair",
                "monitoring": "Vital signs on arrival and discharge",
                "nurse_ratio": "1:4-5",
                "reassessment_interval": "Every 120 minutes or PRN",
                "requirements": ["Basic exam equipment"],
            },
            5: {
                "zone": "Fast-Track / Results Waiting",
                "bed_type": "Chair or waiting area",
                "monitoring": "Vital signs on arrival",
                "nurse_ratio": "1:5-6",
                "reassessment_interval": "PRN / before discharge",
                "requirements": ["Minimal — discharge instructions preparation"],
            },
        }

        assignment = assignments[acuity]

        if beds_available <= 0 and acuity >= 3:
            assignment["overflow_note"] = "ED at capacity — patient may be placed in hallway bed; escalate to charge nurse"

        return json.dumps({
            "patient_id": patient_id,
            "esi_level": acuity,
            "assignment": assignment,
            "ed_beds_available": beds_available,
        }, indent=2)

    @is_tool
    def think(self, thought: str) -> str:
        """Record internal reasoning without taking an action.

        Args:
            thought: Your clinical reasoning process.

        Returns:
            Acknowledgment of thought.
        """
        return json.dumps({"thought_recorded": True, "content": thought[:500]})

    @is_tool
    def submit_answer(self, esi_level: int, disposition: str, reasoning: str, initial_orders: str = "") -> str:
        """Submit your triage decision.

        Args:
            esi_level: ESI level (1-5).
            disposition: Expected disposition (e.g., 'admit_icu', 'admit_floor', 'observe', 'discharge').
            reasoning: Clinical reasoning for your decision.
            initial_orders: Initial orders/interventions.

        Returns:
            Confirmation of submitted decision.
        """
        return json.dumps({
            "submitted": True,
            "esi_level": esi_level,
            "disposition": disposition,
            "reasoning": reasoning[:500],
            "initial_orders": initial_orders[:500],
        })


# ── Helper Functions ──────────────────────────────────────────

def _interpret_vitals(vitals: dict, age: int) -> dict:
    """Interpret vital signs and flag abnormalities."""
    flags = []
    hr = vitals.get("heart_rate", 80)
    sbp = vitals.get("systolic_bp", 120)
    rr = vitals.get("respiratory_rate", 16)
    spo2 = vitals.get("spo2", 98)
    temp = vitals.get("temperature", 37.0)

    if hr > 100:
        flags.append("tachycardia")
    elif hr < 60:
        flags.append("bradycardia")
    if sbp < 90:
        flags.append("hypotension")
    elif sbp > 180:
        flags.append("hypertensive urgency")
    if rr > 22:
        flags.append("tachypnea")
    elif rr < 10:
        flags.append("bradypnea")
    if spo2 < 90:
        flags.append("severe hypoxemia")
    elif spo2 < 94:
        flags.append("hypoxemia")
    if temp > 38.3:
        flags.append("fever")
    elif temp < 36.0:
        flags.append("hypothermia")

    return {"abnormalities": flags, "critical": any(
        f in flags for f in ["hypotension", "severe hypoxemia", "bradypnea"]
    )}


def _is_high_risk(patient) -> bool:
    """Determine if patient is in a high-risk situation."""
    high_risk_complaints = [
        "chest pain", "stroke", "seizure", "severe bleeding",
        "altered mental status", "difficulty breathing",
        "anaphylaxis", "overdose", "suicidal",
    ]
    cc = patient.chief_complaint.lower()
    return any(hrc in cc for hrc in high_risk_complaints) or (patient.pain_score or 0) >= 8


def _in_danger_zone(vitals: dict, age: int) -> bool:
    """Check if vital signs are in the ESI danger zone."""
    hr = vitals.get("heart_rate", 80)
    rr = vitals.get("respiratory_rate", 16)
    spo2 = vitals.get("spo2", 98)
    return hr > 120 or hr < 50 or rr > 28 or rr < 8 or spo2 < 90


def _estimate_resources(patient) -> int:
    """Estimate number of ED resources needed."""
    cc = patient.chief_complaint.lower()
    resources = 0

    # Labs likely needed
    if any(w in cc for w in ["pain", "fever", "infection", "bleeding", "weakness"]):
        resources += 1
    # Imaging likely needed
    if any(w in cc for w in ["chest pain", "trauma", "fall", "headache", "abdominal"]):
        resources += 1
    # IV/medications likely
    if any(w in cc for w in ["pain", "fever", "nausea", "dehydration"]):
        resources += 1

    return min(resources, 4)
