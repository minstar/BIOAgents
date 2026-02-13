"""Tools for the Triage & Emergency domain.

Provides 12 tools for emergency department triage and initial management:
1.  get_patient_presentation  — Chief complaint, vitals, symptoms
2.  get_vital_signs           — Detailed vital signs
3.  assess_airway_breathing   — ABC assessment
4.  get_medical_history       — PMH, medications, allergies
5.  calculate_gcs             — Glasgow Coma Scale
6.  calculate_esi_level       — ESI algorithm support
7.  get_ed_status             — ED occupancy and resources
8.  check_protocol            — Look up emergency protocol
9.  order_stat_labs           — Order STAT laboratory tests
10. order_imaging             — Order STAT imaging
11. think                     — Internal reasoning
12. submit_answer             — Submit triage decision
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
