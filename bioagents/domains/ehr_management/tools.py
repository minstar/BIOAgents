"""Medical tools for the EHR Management domain.

Provides tools for:
- Patient record lookup & admission history
- Lab result trend analysis (time-series)
- Vital sign monitoring & alert detection
- Medication history review
- Clinical score calculation (SOFA, NEWS, APACHE-II)
- Discharge planning & readmission risk assessment
- ICD code lookup & procedure review
"""

from typing import List, Optional

from bioagents.environment.toolkit import ToolKitBase, ToolType, is_tool
from bioagents.domains.ehr_management.data_model import EHRDB, EHRRecord


class EHRTools(ToolKitBase):
    """Tools available to the EHR management agent."""

    db: EHRDB

    def __init__(self, db: EHRDB) -> None:
        super().__init__(db)

    # ==========================================
    # Helper: resolve record
    # ==========================================

    def _get_record(self, hadm_id: str) -> EHRRecord:
        """Resolve a record from hadm_id, raising on miss."""
        if hadm_id not in self.db.records:
            raise ValueError(
                f"Admission '{hadm_id}' not found. "
                f"Available: {list(self.db.records.keys())[:10]}"
            )
        return self.db.records[hadm_id]

    def _find_hadm_for_patient(self, patient_id: str) -> str:
        """Get the most recent admission for a patient."""
        if patient_id in self.db.patient_index:
            hadm_ids = self.db.patient_index[patient_id]
            if hadm_ids:
                return hadm_ids[-1]
        # Fallback: scan records
        for hadm_id, rec in self.db.records.items():
            if rec.demographics.patient_id == patient_id:
                return hadm_id
        raise ValueError(f"No admissions found for patient '{patient_id}'.")

    # ==========================================
    # Category 1: Patient Overview
    # ==========================================

    @is_tool(ToolType.READ)
    def get_patient_summary(self, hadm_id: str) -> dict:
        """Get a comprehensive summary of a patient admission including demographics, diagnoses, length of stay, and current status.

        Args:
            hadm_id: Hospital admission ID

        Returns:
            Summary dictionary with demographics, admission info, diagnoses, and key indicators
        """
        rec = self._get_record(hadm_id)
        self.db.query_log.append({"action": "get_patient_summary", "hadm_id": hadm_id})

        active_meds = [m.drug for m in rec.medication_orders if m.status == "active"]
        latest_scores = {}
        for cs in rec.clinical_scores:
            latest_scores[cs.score_name] = {
                "value": cs.score_value,
                "interpretation": cs.interpretation,
            }

        return {
            "demographics": rec.demographics.model_dump(),
            "admission": {
                "hadm_id": rec.admission.hadm_id,
                "admit_time": rec.admission.admit_time,
                "discharge_time": rec.admission.discharge_time,
                "admit_type": rec.admission.admit_type,
                "diagnosis_at_admission": rec.admission.diagnosis_at_admission,
                "icd_codes": rec.admission.icd_codes,
                "los_days": rec.admission.los_days,
                "is_readmission": rec.admission.is_readmission,
            },
            "icu_stays": len(rec.icu_stays),
            "active_medications": active_meds,
            "latest_clinical_scores": latest_scores,
            "quality_indicators": rec.quality_indicators.model_dump() if rec.quality_indicators else None,
            "prior_admissions": rec.prior_admissions,
        }

    @is_tool(ToolType.READ)
    def get_admission_history(self, patient_id: str) -> list:
        """Get the complete admission history for a patient (all past hospital visits).

        Args:
            patient_id: Patient MRN

        Returns:
            List of admission summaries ordered by date
        """
        self.db.query_log.append({"action": "get_admission_history", "patient_id": patient_id})

        hadm_ids = self.db.patient_index.get(patient_id, [])
        if not hadm_ids:
            return [{"message": f"No admissions found for patient '{patient_id}'."}]

        history = []
        for hadm_id in hadm_ids:
            if hadm_id in self.db.records:
                rec = self.db.records[hadm_id]
                history.append({
                    "hadm_id": hadm_id,
                    "admit_time": rec.admission.admit_time,
                    "discharge_time": rec.admission.discharge_time,
                    "diagnosis": rec.admission.diagnosis_at_admission,
                    "los_days": rec.admission.los_days,
                    "admit_type": rec.admission.admit_type,
                    "is_readmission": rec.admission.is_readmission,
                })

        return sorted(history, key=lambda x: x["admit_time"])

    # ==========================================
    # Category 2: Lab Trends
    # ==========================================

    @is_tool(ToolType.READ)
    def get_lab_results(self, hadm_id: str, lab_name: str = "", last_n: int = 10) -> list:
        """Get lab results for an admission, optionally filtered by lab test name. Returns time-series data for trend analysis.

        Args:
            hadm_id: Hospital admission ID
            lab_name: Optional lab test name filter (e.g., 'Creatinine', 'WBC', 'Hemoglobin')
            last_n: Maximum number of recent results to return

        Returns:
            List of lab results with values, units, flags, and timestamps
        """
        rec = self._get_record(hadm_id)
        self.db.query_log.append({
            "action": "get_lab_results", "hadm_id": hadm_id, "lab_name": lab_name,
        })

        labs = rec.lab_events
        if lab_name:
            labs = [l for l in labs if lab_name.lower() in l.label.lower()]

        # Sort by time descending
        labs = sorted(labs, key=lambda x: x.charttime, reverse=True)[:last_n]

        results = []
        for lab in labs:
            entry = {
                "label": lab.label,
                "value": lab.value,
                "unit": lab.valueuom,
                "charttime": lab.charttime,
                "flag": lab.flag,
            }
            # Add reference range
            ref = self.db.lab_reference_ranges.get(lab.label, {})
            if ref:
                entry["ref_lower"] = ref.get("lower")
                entry["ref_upper"] = ref.get("upper")
            results.append(entry)

        if not results:
            return [{"message": f"No lab results found for '{lab_name}' in admission {hadm_id}."}]

        return results

    @is_tool(ToolType.READ)
    def get_lab_trend(self, hadm_id: str, lab_name: str) -> dict:
        """Analyze the trend of a specific lab test over time (e.g., rising, falling, stable).

        Args:
            hadm_id: Hospital admission ID
            lab_name: Lab test name to analyze trend for

        Returns:
            Trend analysis with values, direction, min/max, and clinical interpretation
        """
        rec = self._get_record(hadm_id)
        self.db.query_log.append({
            "action": "get_lab_trend", "hadm_id": hadm_id, "lab_name": lab_name,
        })

        labs = [l for l in rec.lab_events if lab_name.lower() in l.label.lower()]
        if not labs:
            return {"error": f"No '{lab_name}' results found for admission {hadm_id}."}

        labs = sorted(labs, key=lambda x: x.charttime)
        values = [l.value for l in labs]
        times = [l.charttime for l in labs]

        # Determine trend
        if len(values) < 2:
            trend = "single_value"
        else:
            first_half = values[: len(values) // 2]
            second_half = values[len(values) // 2 :]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            pct_change = (avg_second - avg_first) / max(abs(avg_first), 0.001) * 100

            if pct_change > 15:
                trend = "rising"
            elif pct_change < -15:
                trend = "falling"
            else:
                trend = "stable"

        # Check abnormal flags
        abnormal_count = sum(1 for l in labs if l.flag == "abnormal")
        ref = self.db.lab_reference_ranges.get(lab_name, {})

        return {
            "lab_name": lab_name,
            "num_measurements": len(values),
            "first_time": times[0],
            "last_time": times[-1],
            "values": values,
            "timestamps": times,
            "min_value": min(values),
            "max_value": max(values),
            "latest_value": values[-1],
            "trend": trend,
            "abnormal_count": abnormal_count,
            "reference_range": ref if ref else "Not available",
        }

    # ==========================================
    # Category 3: Vital Sign Monitoring
    # ==========================================

    @is_tool(ToolType.READ)
    def get_vital_signs(self, hadm_id: str, last_n: int = 12) -> list:
        """Get recent vital sign measurements for an admission.

        Args:
            hadm_id: Hospital admission ID
            last_n: Maximum number of recent readings to return

        Returns:
            List of vital sign readings ordered by time
        """
        rec = self._get_record(hadm_id)
        self.db.query_log.append({"action": "get_vital_signs", "hadm_id": hadm_id})

        vitals = sorted(rec.vital_events, key=lambda x: x.charttime, reverse=True)[:last_n]
        return [v.model_dump() for v in vitals]

    @is_tool(ToolType.READ)
    def detect_vital_alerts(self, hadm_id: str) -> list:
        """Detect abnormal vital sign patterns that may require clinical attention. Checks for: tachycardia, bradycardia, hypotension, hypertension, hypoxia, fever, etc.

        Args:
            hadm_id: Hospital admission ID

        Returns:
            List of detected alerts with severity and recommended actions
        """
        rec = self._get_record(hadm_id)
        self.db.query_log.append({"action": "detect_vital_alerts", "hadm_id": hadm_id})

        if not rec.vital_events:
            return [{"message": "No vital signs recorded."}]

        latest = max(rec.vital_events, key=lambda x: x.charttime)
        alerts = []

        # Heart rate
        if latest.heart_rate is not None:
            if latest.heart_rate > 120:
                alerts.append({
                    "type": "tachycardia", "severity": "high",
                    "value": latest.heart_rate, "unit": "bpm",
                    "message": f"Heart rate {latest.heart_rate} bpm (>120). Consider ECG and underlying cause workup.",
                })
            elif latest.heart_rate < 50:
                alerts.append({
                    "type": "bradycardia", "severity": "moderate",
                    "value": latest.heart_rate, "unit": "bpm",
                    "message": f"Heart rate {latest.heart_rate} bpm (<50). Check medications and cardiac rhythm.",
                })

        # Blood pressure
        if latest.sbp is not None:
            if latest.sbp < 90:
                alerts.append({
                    "type": "hypotension", "severity": "high",
                    "value": latest.sbp, "unit": "mmHg",
                    "message": f"SBP {latest.sbp} mmHg (<90). Assess for sepsis, hemorrhage, or cardiogenic shock.",
                })
            elif latest.sbp > 180:
                alerts.append({
                    "type": "hypertensive_urgency", "severity": "high",
                    "value": latest.sbp, "unit": "mmHg",
                    "message": f"SBP {latest.sbp} mmHg (>180). Evaluate for end-organ damage.",
                })

        # SpO2
        if latest.spo2 is not None and latest.spo2 < 92:
            alerts.append({
                "type": "hypoxia", "severity": "high",
                "value": latest.spo2, "unit": "%",
                "message": f"SpO2 {latest.spo2}% (<92). Increase supplemental O2, consider ABG.",
            })

        # Temperature
        if latest.temperature is not None:
            if latest.temperature > 38.3:
                alerts.append({
                    "type": "fever", "severity": "moderate",
                    "value": latest.temperature, "unit": "°C",
                    "message": f"Temperature {latest.temperature}°C (>38.3). Blood cultures, infection workup.",
                })
            elif latest.temperature < 36.0:
                alerts.append({
                    "type": "hypothermia", "severity": "moderate",
                    "value": latest.temperature, "unit": "°C",
                    "message": f"Temperature {latest.temperature}°C (<36.0). Warm blankets, recheck.",
                })

        # GCS
        if latest.gcs_total is not None and latest.gcs_total < 13:
            alerts.append({
                "type": "altered_consciousness", "severity": "high",
                "value": latest.gcs_total, "unit": "GCS",
                "message": f"GCS {latest.gcs_total} (<13). Neurological assessment, consider CT head.",
            })

        if not alerts:
            return [{"message": "No abnormal vital signs detected at latest reading.", "charttime": latest.charttime}]

        return alerts

    # ==========================================
    # Category 4: Medication Review
    # ==========================================

    @is_tool(ToolType.READ)
    def get_medication_orders(self, hadm_id: str, active_only: bool = False) -> list:
        """Get medication orders for an admission.

        Args:
            hadm_id: Hospital admission ID
            active_only: If true, return only currently active orders

        Returns:
            List of medication orders with drug name, dose, route, and status
        """
        rec = self._get_record(hadm_id)
        self.db.query_log.append({
            "action": "get_medication_orders", "hadm_id": hadm_id, "active_only": active_only,
        })

        meds = rec.medication_orders
        if active_only:
            meds = [m for m in meds if m.status == "active"]

        return [m.model_dump() for m in meds]

    # ==========================================
    # Category 5: Clinical Scores & Outcomes
    # ==========================================

    @is_tool(ToolType.READ)
    def get_clinical_scores(self, hadm_id: str) -> list:
        """Get all calculated clinical severity scores for an admission (SOFA, NEWS, APACHE-II, SAPS-II, etc.).

        Args:
            hadm_id: Hospital admission ID

        Returns:
            List of clinical scores with values, interpretation, and component breakdown
        """
        rec = self._get_record(hadm_id)
        self.db.query_log.append({"action": "get_clinical_scores", "hadm_id": hadm_id})

        if not rec.clinical_scores:
            return [{"message": "No clinical scores calculated for this admission."}]

        return [cs.model_dump() for cs in rec.clinical_scores]

    @is_tool(ToolType.READ)
    def get_quality_indicators(self, hadm_id: str) -> dict:
        """Get quality and outcome indicators for the admission including readmission risk, mortality risk, expected LOS, sepsis flag, and AKI staging.

        Args:
            hadm_id: Hospital admission ID

        Returns:
            Quality indicators with risk scores and flags
        """
        rec = self._get_record(hadm_id)
        self.db.query_log.append({"action": "get_quality_indicators", "hadm_id": hadm_id})

        if rec.quality_indicators is None:
            return {"message": "Quality indicators not available for this admission."}

        return rec.quality_indicators.model_dump()

    # ==========================================
    # Category 6: Procedures & Discharge
    # ==========================================

    @is_tool(ToolType.READ)
    def get_procedures(self, hadm_id: str) -> list:
        """Get all procedures performed during the admission.

        Args:
            hadm_id: Hospital admission ID

        Returns:
            List of procedures with name, time, and outcome
        """
        rec = self._get_record(hadm_id)
        self.db.query_log.append({"action": "get_procedures", "hadm_id": hadm_id})

        if not rec.procedures:
            return [{"message": "No procedures recorded for this admission."}]

        return [p.model_dump() for p in rec.procedures]

    @is_tool(ToolType.READ)
    def get_discharge_summary(self, hadm_id: str) -> dict:
        """Get the discharge summary for a completed admission. Includes diagnoses, discharge medications, and follow-up plan.

        Args:
            hadm_id: Hospital admission ID

        Returns:
            Discharge summary text with structured data
        """
        rec = self._get_record(hadm_id)
        self.db.query_log.append({"action": "get_discharge_summary", "hadm_id": hadm_id})

        if rec.discharge_summary is None:
            return {"message": "Discharge summary not yet available (patient may still be admitted)."}

        return rec.discharge_summary.model_dump()

    # ==========================================
    # Category 7: ICD Lookup
    # ==========================================

    @is_tool(ToolType.READ)
    def lookup_icd_code(self, code: str) -> dict:
        """Look up the description for an ICD-10 diagnosis or procedure code.

        Args:
            code: ICD-10 code (e.g., 'J18.9', 'I50.9')

        Returns:
            Code description and related information
        """
        self.db.query_log.append({"action": "lookup_icd_code", "code": code})

        desc = self.db.icd_descriptions.get(code, "")
        if desc:
            return {"code": code, "description": desc}

        # Partial match
        matches = {k: v for k, v in self.db.icd_descriptions.items() if k.startswith(code[:3])}
        if matches:
            return {
                "code": code,
                "description": "Exact code not found.",
                "related_codes": matches,
            }

        return {"code": code, "description": "Code not found in the database."}

    # ==========================================
    # Category 8: Reasoning & Answer
    # ==========================================

    @is_tool(ToolType.GENERIC)
    def think(self, thought: str) -> str:
        """Internal reasoning tool. Use to reason through EHR analysis and clinical decisions.

        Args:
            thought: Your clinical reasoning process

        Returns:
            Empty string (thinking is logged)
        """
        return ""

    @is_tool(ToolType.GENERIC)
    def submit_answer(self, answer: str, reasoning: str = "") -> str:
        """Submit your final clinical assessment or recommendation based on EHR review.

        Args:
            answer: Your clinical assessment, recommendation, or answer to the task question
            reasoning: Your clinical reasoning supporting the answer

        Returns:
            Confirmation of the submitted answer
        """
        return f"Assessment '{answer}' submitted. Reasoning: {reasoning}"
