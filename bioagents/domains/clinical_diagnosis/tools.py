"""Medical tools for the Clinical Diagnosis domain.

Provides tools for:
- Patient record lookup
- Vital signs retrieval
- Lab test ordering and results
- Drug interaction checking
- Clinical guideline search
- Differential diagnosis generation
- Medical literature search (simulated)
- Specialist referral
"""

from typing import List, Optional

from bioagents.environment.toolkit import ToolKitBase, ToolType, is_tool
from bioagents.domains.clinical_diagnosis.data_model import (
    ClinicalDB,
    Patient,
    VitalSigns,
    LabResult,
    LabOrder,
    Medication,
    ClinicalNote,
    DrugInteraction,
    ClinicalGuideline,
)


class ClinicalTools(ToolKitBase):
    """Tools available to the clinical diagnosis agent."""

    db: ClinicalDB

    def __init__(self, db: ClinicalDB) -> None:
        super().__init__(db)

    # ==========================================
    # Category 1: Patient Information
    # ==========================================

    @is_tool(ToolType.READ)
    def get_patient_info(self, patient_id: str) -> dict:
        """Get basic patient demographics, allergies, conditions, and current medications.

        Args:
            patient_id: The unique patient identifier (e.g., 'P001')

        Returns:
            Patient demographics and medical summary

        Raises:
            ValueError: If the patient is not found
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found in the system.")
        
        patient = self.db.patients[patient_id]
        return {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "age": patient.age,
            "sex": patient.sex,
            "date_of_birth": patient.date_of_birth,
            "blood_type": patient.blood_type,
            "chief_complaint": patient.chief_complaint,
            "allergies": [a.model_dump() for a in patient.allergies],
            "active_conditions": [
                c.model_dump() for c in patient.conditions if c.status in ("active", "chronic")
            ],
            "current_medications": [
                m.model_dump() for m in patient.medications if m.end_date is None
            ],
            "family_history": patient.family_history,
            "social_history": patient.social_history,
        }

    @is_tool(ToolType.READ)
    def get_patient_history(self, patient_id: str) -> dict:
        """Get the complete medical history of a patient including all conditions, past medications, and family/social history.

        Args:
            patient_id: The unique patient identifier

        Returns:
            Complete medical history

        Raises:
            ValueError: If the patient is not found
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")
        
        patient = self.db.patients[patient_id]
        return {
            "patient_id": patient.patient_id,
            "all_conditions": [c.model_dump() for c in patient.conditions],
            "all_medications": [m.model_dump() for m in patient.medications],
            "family_history": patient.family_history,
            "social_history": patient.social_history,
        }

    # ==========================================
    # Category 2: Vital Signs
    # ==========================================

    @is_tool(ToolType.READ)
    def get_vital_signs(self, patient_id: str) -> dict:
        """Get the most recent vital signs for a patient.

        Args:
            patient_id: The unique patient identifier

        Returns:
            Most recent vital signs (BP, HR, temperature, RR, SpO2)

        Raises:
            ValueError: If the patient is not found or no vitals recorded
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")
        
        patient = self.db.patients[patient_id]
        if not patient.vital_signs:
            raise ValueError(f"No vital signs recorded for patient '{patient_id}'.")
        
        latest = patient.vital_signs[-1]
        return latest.model_dump()

    @is_tool(ToolType.READ)
    def get_vital_signs_trend(self, patient_id: str, num_readings: int = 5) -> list:
        """Get the trend of vital signs over recent readings for a patient.

        Args:
            patient_id: The unique patient identifier
            num_readings: Number of recent readings to retrieve (default 5)

        Returns:
            List of recent vital sign readings, ordered oldest to newest
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")
        
        patient = self.db.patients[patient_id]
        readings = patient.vital_signs[-int(num_readings):]
        return [v.model_dump() for v in readings]

    # ==========================================
    # Category 3: Lab Tests
    # ==========================================

    @is_tool(ToolType.READ)
    def get_lab_results(self, patient_id: str, category: str = "") -> list:
        """Get lab test results for a patient, optionally filtered by category.

        Args:
            patient_id: The unique patient identifier
            category: Optional category filter (e.g., 'hematology', 'chemistry', 'microbiology'). Empty string returns all.

        Returns:
            List of lab test results

        Raises:
            ValueError: If the patient is not found
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")
        
        patient = self.db.patients[patient_id]
        results = patient.lab_results
        
        if category:
            results = [r for r in results if r.category.lower() == category.lower()]
        
        return [r.model_dump() for r in results]

    @is_tool(ToolType.WRITE)
    def order_lab_test(self, patient_id: str, test_name: str, priority: str = "routine") -> dict:
        """Order a new lab test for a patient. The test will be simulated and results returned.

        Args:
            patient_id: The unique patient identifier
            test_name: Name of the test to order (e.g., 'Complete Blood Count', 'Basic Metabolic Panel', 'Urinalysis')
            priority: Priority level ('routine', 'urgent', 'stat')

        Returns:
            Order confirmation with order_id and status

        Raises:
            ValueError: If the patient is not found
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")
        
        if priority not in ("routine", "urgent", "stat"):
            raise ValueError(f"Invalid priority '{priority}'. Must be 'routine', 'urgent', or 'stat'.")
        
        order_id = f"ORD-{len(self.db.lab_orders) + 1:04d}"
        
        from datetime import datetime
        order = LabOrder(
            order_id=order_id,
            patient_id=patient_id,
            test_name=test_name,
            order_date=datetime.now().strftime("%Y-%m-%d"),
            status="ordered",
            priority=priority,
        )
        self.db.lab_orders[order_id] = order
        
        return {
            "order_id": order_id,
            "test_name": test_name,
            "patient_id": patient_id,
            "status": "ordered",
            "priority": priority,
            "message": f"Lab test '{test_name}' ordered for patient {patient_id}. Results will be available soon.",
        }

    # ==========================================
    # Category 4: Medications & Drug Interactions
    # ==========================================

    @is_tool(ToolType.READ)
    def get_medications(self, patient_id: str) -> list:
        """Get the current medication list for a patient.

        Args:
            patient_id: The unique patient identifier

        Returns:
            List of current medications with dosage, frequency, and route
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")
        
        patient = self.db.patients[patient_id]
        current_meds = [m for m in patient.medications if m.end_date is None]
        return [m.model_dump() for m in current_meds]

    @is_tool(ToolType.READ)
    def check_drug_interaction(self, drug_a: str, drug_b: str) -> dict:
        """Check for known interactions between two drugs.

        Args:
            drug_a: Name of the first drug
            drug_b: Name of the second drug

        Returns:
            Interaction information including severity and recommendation
        """
        drug_a_lower = drug_a.lower()
        drug_b_lower = drug_b.lower()
        
        for interaction in self.db.drug_interactions:
            ia = interaction.drug_a.lower()
            ib = interaction.drug_b.lower()
            if (ia == drug_a_lower and ib == drug_b_lower) or \
               (ia == drug_b_lower and ib == drug_a_lower):
                return interaction.model_dump()
        
        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": "none",
            "description": "No known interaction found between these medications.",
            "recommendation": "No special precautions needed based on available data.",
        }

    @is_tool(ToolType.WRITE)
    def prescribe_medication(
        self, patient_id: str, drug_name: str, dosage: str, frequency: str, route: str = "oral"
    ) -> dict:
        """Prescribe a new medication for a patient. Checks for allergies and drug interactions.

        Args:
            patient_id: The unique patient identifier
            drug_name: Name of the medication
            dosage: Dosage (e.g., '500mg')
            frequency: Frequency (e.g., 'twice daily', 'every 8 hours')
            route: Route of administration (default 'oral')

        Returns:
            Prescription confirmation or warnings

        Raises:
            ValueError: If the patient is not found or allergy detected
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")
        
        patient = self.db.patients[patient_id]
        
        # Check allergies
        for allergy in patient.allergies:
            if allergy.allergen.lower() in drug_name.lower():
                raise ValueError(
                    f"ALLERGY ALERT: Patient is allergic to '{allergy.allergen}' "
                    f"(reaction: {allergy.reaction}, severity: {allergy.severity}). "
                    f"Cannot prescribe '{drug_name}'."
                )
        
        # Check drug interactions with current medications
        warnings = []
        current_meds = [m for m in patient.medications if m.end_date is None]
        for med in current_meds:
            interaction = self.check_drug_interaction(drug_name, med.drug_name)
            if interaction.get("severity") not in ("none", None):
                warnings.append(interaction)
        
        from datetime import datetime
        new_med = Medication(
            drug_name=drug_name,
            dosage=dosage,
            frequency=frequency,
            route=route,
            start_date=datetime.now().strftime("%Y-%m-%d"),
            prescriber="agent",
        )
        patient.medications.append(new_med)
        
        result = {
            "status": "prescribed",
            "drug_name": drug_name,
            "dosage": dosage,
            "frequency": frequency,
            "route": route,
            "patient_id": patient_id,
        }
        if warnings:
            result["interaction_warnings"] = warnings
        
        return result

    # ==========================================
    # Category 5: Clinical Notes
    # ==========================================

    @is_tool(ToolType.READ)
    def get_clinical_notes(self, patient_id: str, note_type: str = "") -> list:
        """Get clinical notes for a patient, optionally filtered by type.

        Args:
            patient_id: The unique patient identifier
            note_type: Optional filter ('admission', 'progress', 'discharge', 'consultation', 'procedure'). Empty returns all.

        Returns:
            List of clinical notes
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")
        
        patient = self.db.patients[patient_id]
        notes = patient.clinical_notes
        
        if note_type:
            notes = [n for n in notes if n.note_type == note_type]
        
        return [n.model_dump() for n in notes]

    @is_tool(ToolType.WRITE)
    def add_clinical_note(
        self, patient_id: str, note_type: str, content: str, diagnosis_codes: str = ""
    ) -> dict:
        """Add a new clinical note to the patient record.

        Args:
            patient_id: The unique patient identifier
            note_type: Type of note ('admission', 'progress', 'discharge', 'consultation', 'procedure')
            content: The note content
            diagnosis_codes: Comma-separated ICD-10 codes (optional)

        Returns:
            Confirmation of the note being added
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")
        
        from datetime import datetime
        patient = self.db.patients[patient_id]
        note_id = f"NOTE-{patient_id}-{len(patient.clinical_notes) + 1:03d}"
        
        codes = [c.strip() for c in diagnosis_codes.split(",") if c.strip()] if diagnosis_codes else None
        
        note = ClinicalNote(
            note_id=note_id,
            note_type=note_type,
            author="agent",
            date=datetime.now().strftime("%Y-%m-%d"),
            content=content,
            diagnosis_codes=codes,
        )
        patient.clinical_notes.append(note)
        
        return {"note_id": note_id, "status": "added", "note_type": note_type}

    # ==========================================
    # Category 6: Diagnosis & Guidelines
    # ==========================================

    @is_tool(ToolType.READ)
    def get_differential_diagnosis(self, symptoms: str) -> list:
        """Generate a differential diagnosis based on provided symptoms. Uses the internal knowledge base.

        Args:
            symptoms: Comma-separated list of symptoms (e.g., 'fever, cough, dyspnea, chest pain')

        Returns:
            List of possible diagnoses ranked by likelihood
        """
        # This will be enhanced with actual medical knowledge later
        # For now, returns a structured placeholder that tool simulation will fill
        symptom_list = [s.strip().lower() for s in symptoms.split(",")]
        
        # Simple rule-based differential (will be replaced by LLM simulation)
        differentials = []
        
        symptom_condition_map = {
            frozenset(["fever", "cough", "dyspnea"]): [
                {"condition": "Community-acquired pneumonia", "probability": "high", "icd10": "J18.9"},
                {"condition": "Acute bronchitis", "probability": "moderate", "icd10": "J20.9"},
                {"condition": "COVID-19", "probability": "moderate", "icd10": "U07.1"},
            ],
            frozenset(["chest pain", "dyspnea"]): [
                {"condition": "Acute coronary syndrome", "probability": "high", "icd10": "I21.9"},
                {"condition": "Pulmonary embolism", "probability": "moderate", "icd10": "I26.99"},
                {"condition": "Pneumothorax", "probability": "low", "icd10": "J93.9"},
            ],
            frozenset(["headache", "fever", "neck stiffness"]): [
                {"condition": "Bacterial meningitis", "probability": "high", "icd10": "G00.9"},
                {"condition": "Viral meningitis", "probability": "moderate", "icd10": "A87.9"},
                {"condition": "Subarachnoid hemorrhage", "probability": "low", "icd10": "I60.9"},
            ],
            frozenset(["abdominal pain", "nausea", "vomiting"]): [
                {"condition": "Acute appendicitis", "probability": "moderate", "icd10": "K35.80"},
                {"condition": "Acute cholecystitis", "probability": "moderate", "icd10": "K81.0"},
                {"condition": "Gastroenteritis", "probability": "moderate", "icd10": "K52.9"},
            ],
        }
        
        symptom_set = frozenset(symptom_list)
        best_match = None
        best_overlap = 0
        
        for key_symptoms, conditions in symptom_condition_map.items():
            overlap = len(symptom_set & key_symptoms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = conditions
        
        if best_match:
            differentials = best_match
        else:
            differentials = [
                {
                    "condition": "Further evaluation needed",
                    "probability": "undetermined",
                    "icd10": "R69",
                    "note": f"Symptoms presented: {', '.join(symptom_list)}. Consider additional workup.",
                }
            ]
        
        return differentials

    @is_tool(ToolType.READ)
    def search_clinical_guidelines(self, condition: str) -> list:
        """Search for clinical guidelines related to a specific condition.

        Args:
            condition: The medical condition to search guidelines for

        Returns:
            List of relevant clinical guidelines
        """
        condition_lower = condition.lower()
        results = []
        
        for gid, guideline in self.db.clinical_guidelines.items():
            if condition_lower in guideline.condition.lower() or \
               condition_lower in guideline.title.lower():
                results.append(guideline.model_dump())
        
        if not results:
            return [{
                "message": f"No specific guidelines found for '{condition}'. Consider searching with broader terms.",
                "suggestion": "Try searching for the general disease category or related conditions.",
            }]
        
        return results

    @is_tool(ToolType.WRITE)
    def record_diagnosis(
        self, patient_id: str, diagnosis: str, icd10_code: str = "", confidence: str = "moderate", reasoning: str = ""
    ) -> dict:
        """Record a diagnosis for a patient.

        Args:
            patient_id: The unique patient identifier
            diagnosis: The diagnosis name
            icd10_code: ICD-10 code for the diagnosis
            confidence: Confidence level ('low', 'moderate', 'high')
            reasoning: Clinical reasoning supporting the diagnosis

        Returns:
            Confirmation of the diagnosis being recorded
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")
        
        from datetime import datetime
        
        patient = self.db.patients[patient_id]
        patient.conditions.append(
            __import__("bioagents.domains.clinical_diagnosis.data_model", fromlist=["Condition"]).Condition(
                condition_name=diagnosis,
                icd10_code=icd10_code or None,
                status="active",
                onset_date=datetime.now().strftime("%Y-%m-%d"),
                notes=reasoning,
            )
        )
        
        log_entry = {
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "icd10_code": icd10_code,
            "confidence": confidence,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
        }
        self.db.diagnosis_log.append(log_entry)
        
        return {
            "status": "recorded",
            "diagnosis": diagnosis,
            "icd10_code": icd10_code,
            "confidence": confidence,
            "patient_id": patient_id,
        }

    # ==========================================
    # Category 7: Medical Literature Search (simulated)
    # ==========================================

    @is_tool(ToolType.READ)
    def search_medical_literature(self, query: str) -> list:
        """Search medical literature (PubMed-style) for relevant articles and evidence.

        Args:
            query: Search query (e.g., 'pneumonia treatment guidelines 2025')

        Returns:
            List of relevant articles with title, abstract snippet, and PMID
        """
        # This will be connected to actual search/simulation later
        # For now provides structured format for tool simulation
        return [
            {
                "title": f"Search results for: {query}",
                "message": "Literature search simulation - will be connected to PubMed/medical knowledge base.",
                "note": "Use search_clinical_guidelines for guideline-specific queries.",
            }
        ]

    # ==========================================
    # Category 8: Communication & Workflow
    # ==========================================

    @is_tool(ToolType.GENERIC)
    def transfer_to_specialist(self, summary: str, specialty: str) -> str:
        """Transfer the case to a specialist with a clinical summary.

        Args:
            summary: Summary of the patient's condition and reason for referral
            specialty: Medical specialty to refer to (e.g., 'cardiology', 'neurology', 'surgery')

        Returns:
            Confirmation of the referral
        """
        return f"Referral to {specialty} submitted successfully. Summary: {summary}"

    # ==========================================
    # Category 9: Clinical Scoring Tools
    # ==========================================

    @is_tool(ToolType.READ)
    def calculate_curb65(self, patient_id: str) -> dict:
        """Calculate the CURB-65 pneumonia severity score for a patient.

        Criteria (1 point each):
        - Confusion (new-onset)
        - Urea (BUN) > 7 mmol/L
        - Respiratory rate >= 30/min
        - Blood pressure: systolic < 90 mmHg or diastolic <= 60 mmHg
        - Age >= 65

        Args:
            patient_id: The unique patient identifier

        Returns:
            CURB-65 score (0-5), breakdown, and management recommendation

        Raises:
            ValueError: If the patient is not found
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")

        patient = self.db.patients[patient_id]

        # --- Gather values from DB or simulate ---
        # Age
        age = patient.age
        age_point = 1 if age >= 65 else 0

        # Vital signs (latest)
        if patient.vital_signs:
            latest_vitals = patient.vital_signs[-1]
            rr = latest_vitals.respiratory_rate
            sbp = latest_vitals.blood_pressure_systolic
            dbp = latest_vitals.blood_pressure_diastolic
        else:
            rr, sbp, dbp = 22, 125, 78  # simulated normal defaults

        rr_point = 1 if rr >= 30 else 0
        bp_point = 1 if (sbp < 90 or dbp <= 60) else 0

        # Urea / BUN from lab results
        urea_value = None
        for lab in reversed(patient.lab_results):
            if lab.test_name.lower() in ("urea", "bun", "blood urea nitrogen"):
                try:
                    urea_value = float(lab.value)
                except (ValueError, TypeError):
                    pass
                break
        if urea_value is None:
            urea_value = 5.2  # simulated normal
        urea_point = 1 if urea_value > 7 else 0

        # Confusion — check conditions or clinical notes
        confusion = False
        for cond in patient.conditions:
            if "confusion" in cond.condition_name.lower() or "delirium" in cond.condition_name.lower():
                confusion = True
                break
        if not confusion:
            for note in patient.clinical_notes:
                if "confusion" in note.content.lower() or "altered mental status" in note.content.lower():
                    confusion = True
                    break
        confusion_point = 1 if confusion else 0

        total = confusion_point + urea_point + rr_point + bp_point + age_point

        # Management recommendation
        if total <= 1:
            recommendation = "Low risk. Consider outpatient treatment with oral antibiotics."
            risk_level = "low"
        elif total == 2:
            recommendation = "Moderate risk. Consider short inpatient stay or hospital-supervised outpatient care."
            risk_level = "moderate"
        elif total == 3:
            recommendation = "High risk. Hospital admission recommended. Consider IV antibiotics."
            risk_level = "high"
        else:
            recommendation = "Very high risk. Urgent admission; consider ICU care and IV antibiotics."
            risk_level = "very_high"

        return {
            "patient_id": patient_id,
            "score": total,
            "max_score": 5,
            "risk_level": risk_level,
            "breakdown": {
                "confusion": {"present": confusion, "points": confusion_point},
                "urea_gt_7": {"value_mmol_L": urea_value, "points": urea_point},
                "respiratory_rate_gte_30": {"value": rr, "points": rr_point},
                "low_blood_pressure": {"systolic": sbp, "diastolic": dbp, "points": bp_point},
                "age_gte_65": {"age": age, "points": age_point},
            },
            "recommendation": recommendation,
        }

    @is_tool(ToolType.READ)
    def calculate_wells_score(self, patient_id: str, indication: str) -> dict:
        """Calculate the Wells score for Pulmonary Embolism (PE) or Deep Vein Thrombosis (DVT).

        Args:
            patient_id: The unique patient identifier
            indication: Either 'pe' for pulmonary embolism or 'dvt' for deep vein thrombosis

        Returns:
            Wells score with probability category and recommended next steps

        Raises:
            ValueError: If the patient is not found or indication is invalid
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")
        indication = indication.lower().strip()
        if indication not in ("pe", "dvt"):
            raise ValueError(f"Invalid indication '{indication}'. Must be 'pe' or 'dvt'.")

        patient = self.db.patients[patient_id]

        # Helper: check conditions and notes for keywords
        def _has_keyword(*keywords: str) -> bool:
            for cond in patient.conditions:
                for kw in keywords:
                    if kw in cond.condition_name.lower():
                        return True
            for note in patient.clinical_notes:
                for kw in keywords:
                    if kw in note.content.lower():
                        return True
            return False

        hr = patient.vital_signs[-1].heart_rate if patient.vital_signs else 78

        if indication == "pe":
            criteria = {
                "clinical_signs_dvt": {"present": _has_keyword("dvt", "deep vein", "leg swelling"), "points": 3.0},
                "pe_most_likely_diagnosis": {"present": _has_keyword("pulmonary embolism", "pe suspect"), "points": 3.0},
                "heart_rate_gt_100": {"present": hr > 100, "points": 1.5},
                "immobilization_or_surgery": {"present": _has_keyword("immobil", "surgery", "post-operative"), "points": 1.5},
                "previous_dvt_pe": {"present": _has_keyword("history of dvt", "history of pe", "prior dvt", "prior pe"), "points": 1.5},
                "hemoptysis": {"present": _has_keyword("hemoptysis"), "points": 1.0},
                "malignancy": {"present": _has_keyword("cancer", "malignancy", "neoplasm", "carcinoma"), "points": 1.0},
            }
            total = sum(c["points"] for c in criteria.values() if c["present"])

            if total < 2:
                category, probability = "low", "~1.3% prevalence"
                recommendation = "Consider D-dimer testing. If negative, PE effectively excluded."
            elif total <= 6:
                category, probability = "moderate", "~16.2% prevalence"
                recommendation = "D-dimer testing recommended. If positive, proceed to CT pulmonary angiography (CTPA)."
            else:
                category, probability = "high", "~37.5% prevalence"
                recommendation = "Proceed directly to CT pulmonary angiography (CTPA). Consider empiric anticoagulation."

            return {
                "patient_id": patient_id,
                "indication": "Pulmonary Embolism (PE)",
                "score": total,
                "probability_category": category,
                "estimated_prevalence": probability,
                "criteria": criteria,
                "recommendation": recommendation,
            }

        # DVT
        criteria = {
            "active_cancer": {"present": _has_keyword("cancer", "malignancy", "neoplasm", "carcinoma"), "points": 1},
            "paralysis_or_immobilization": {"present": _has_keyword("paralysis", "paresis", "immobil", "cast"), "points": 1},
            "bedridden_gt_3_days_or_surgery": {"present": _has_keyword("bedridden", "surgery", "post-operative"), "points": 1},
            "localized_tenderness": {"present": _has_keyword("calf tenderness", "leg tenderness", "localized tenderness"), "points": 1},
            "entire_leg_swollen": {"present": _has_keyword("entire leg swollen", "whole leg edema"), "points": 1},
            "calf_swelling_gt_3cm": {"present": _has_keyword("calf swelling", "calf circumference"), "points": 1},
            "pitting_edema": {"present": _has_keyword("pitting edema"), "points": 1},
            "collateral_superficial_veins": {"present": _has_keyword("collateral vein", "superficial vein"), "points": 1},
            "previous_dvt": {"present": _has_keyword("history of dvt", "prior dvt", "previous dvt"), "points": 1},
            "alternative_diagnosis_likely": {"present": _has_keyword("cellulitis", "baker's cyst", "superficial thrombophlebitis"), "points": -2},
        }
        total = max(sum(c["points"] for c in criteria.values() if c["present"]), 0)

        if total <= 0:
            category = "low"
            recommendation = "DVT unlikely. Consider D-dimer to exclude. If negative, no further imaging."
        elif total <= 2:
            category = "moderate"
            recommendation = "Moderate probability. Perform D-dimer; if positive, obtain compression ultrasound."
        else:
            category = "high"
            recommendation = "DVT likely. Proceed to compression ultrasound. Consider empiric anticoagulation."

        return {
            "patient_id": patient_id,
            "indication": "Deep Vein Thrombosis (DVT)",
            "score": total,
            "probability_category": category,
            "criteria": criteria,
            "recommendation": recommendation,
        }

    @is_tool(ToolType.READ)
    def calculate_chads2_vasc(self, patient_id: str) -> dict:
        """Calculate the CHA2DS2-VASc score for stroke risk in atrial fibrillation.

        Criteria:
        - C: Congestive heart failure (1 pt)
        - H: Hypertension (1 pt)
        - A2: Age >= 75 (2 pts)
        - D: Diabetes mellitus (1 pt)
        - S2: Prior Stroke/TIA/thromboembolism (2 pts)
        - V: Vascular disease (MI, PAD, aortic plaque) (1 pt)
        - A: Age 65-74 (1 pt)
        - Sc: Sex category — female (1 pt)

        Args:
            patient_id: The unique patient identifier

        Returns:
            CHA2DS2-VASc score (0-9) with annual stroke risk and anticoagulation recommendation

        Raises:
            ValueError: If the patient is not found
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")

        patient = self.db.patients[patient_id]

        def _has_condition(*keywords: str) -> bool:
            for cond in patient.conditions:
                for kw in keywords:
                    if kw in cond.condition_name.lower():
                        return True
            return False

        age = patient.age

        chf = _has_condition("heart failure", "chf", "congestive heart")
        hypertension = _has_condition("hypertension", "high blood pressure", "htn")
        diabetes = _has_condition("diabetes", "dm type", "dm2", "dm1")
        stroke_tia = _has_condition("stroke", "tia", "transient ischemic", "cerebrovascular accident", "cva")
        vascular = _has_condition("myocardial infarction", "peripheral artery", "pad", "aortic plaque", "mi")
        female = patient.sex == "female"

        criteria = {
            "congestive_heart_failure": {"present": chf, "points": 1},
            "hypertension": {"present": hypertension, "points": 1},
            "age_gte_75": {"present": age >= 75, "points": 2 if age >= 75 else 0},
            "diabetes": {"present": diabetes, "points": 1},
            "stroke_tia_thromboembolism": {"present": stroke_tia, "points": 2 if stroke_tia else 0},
            "vascular_disease": {"present": vascular, "points": 1},
            "age_65_to_74": {"present": 65 <= age < 75, "points": 1 if 65 <= age < 75 else 0},
            "sex_category_female": {"present": female, "points": 1},
        }

        total = sum(c["points"] for c in criteria.values() if c["present"])

        # Annual stroke risk approximation
        stroke_risk_table = {
            0: 0.0, 1: 1.3, 2: 2.2, 3: 3.2, 4: 4.0,
            5: 6.7, 6: 9.8, 7: 9.6, 8: 12.5, 9: 15.2,
        }
        annual_stroke_risk = stroke_risk_table.get(total, 15.2)

        if total == 0:
            recommendation = "Low risk. No anticoagulation or aspirin therapy recommended."
        elif total == 1:
            recommendation = "Low-moderate risk. Consider oral anticoagulation (OAC) or aspirin. OAC preferred."
        else:
            recommendation = f"Anticoagulation recommended (e.g., DOAC or warfarin). Annual stroke risk ~{annual_stroke_risk}%."

        return {
            "patient_id": patient_id,
            "score": total,
            "max_score": 9,
            "annual_stroke_risk_percent": annual_stroke_risk,
            "criteria": criteria,
            "recommendation": recommendation,
        }

    @is_tool(ToolType.READ)
    def calculate_meld_score(self, patient_id: str) -> dict:
        """Calculate the MELD (Model for End-Stage Liver Disease) score.

        Uses bilirubin, INR, and serum creatinine to estimate 90-day mortality risk.
        MELD = 3.78 * ln(bilirubin mg/dL) + 11.2 * ln(INR) + 9.57 * ln(creatinine mg/dL) + 6.43

        Args:
            patient_id: The unique patient identifier

        Returns:
            MELD score with component values and 90-day mortality estimate

        Raises:
            ValueError: If the patient is not found
        """
        import math

        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")

        patient = self.db.patients[patient_id]

        # Try to extract lab values; use simulated realistic defaults if missing
        bilirubin = None
        inr = None
        creatinine = None

        for lab in reversed(patient.lab_results):
            name = lab.test_name.lower()
            try:
                val = float(lab.value)
            except (ValueError, TypeError):
                continue
            if bilirubin is None and "bilirubin" in name and "direct" not in name:
                bilirubin = val
            elif inr is None and name in ("inr", "international normalized ratio", "pt/inr"):
                inr = val
            elif creatinine is None and "creatinine" in name and "clearance" not in name:
                creatinine = val

        # Simulated defaults for missing values (mildly abnormal for liver-disease context)
        if bilirubin is None:
            bilirubin = 2.1
        if inr is None:
            inr = 1.4
        if creatinine is None:
            creatinine = 1.1

        # MELD formula — floor each value at 1.0 per convention
        bili_calc = max(bilirubin, 1.0)
        inr_calc = max(inr, 1.0)
        cr_calc = max(creatinine, 1.0)
        cr_calc = min(cr_calc, 4.0)  # cap creatinine at 4.0 per MELD rules

        raw_score = (
            3.78 * math.log(bili_calc)
            + 11.2 * math.log(inr_calc)
            + 9.57 * math.log(cr_calc)
            + 6.43
        )
        meld = int(round(max(min(raw_score, 40), 6)))

        # 90-day mortality approximation
        if meld <= 9:
            mortality = "~1.9%"
        elif meld <= 19:
            mortality = "~6.0%"
        elif meld <= 29:
            mortality = "~19.6%"
        elif meld <= 39:
            mortality = "~52.6%"
        else:
            mortality = "~71.3%"

        return {
            "patient_id": patient_id,
            "meld_score": meld,
            "components": {
                "bilirubin_mg_dL": bilirubin,
                "inr": inr,
                "creatinine_mg_dL": creatinine,
            },
            "estimated_90_day_mortality": mortality,
            "interpretation": (
                "Higher MELD scores indicate more severe liver disease and higher priority "
                "for liver transplantation. Score range: 6-40."
            ),
        }

    @is_tool(ToolType.READ)
    def calculate_apache2(self, patient_id: str) -> dict:
        """Calculate the APACHE II (Acute Physiology and Chronic Health Evaluation II) score for ICU patients.

        Combines acute physiology score, age points, and chronic health evaluation.
        Score range: 0-71. Higher scores indicate greater severity of illness.

        Args:
            patient_id: The unique patient identifier

        Returns:
            APACHE II score with component breakdown and predicted mortality

        Raises:
            ValueError: If the patient is not found
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")

        patient = self.db.patients[patient_id]

        # --- Vital signs ---
        if patient.vital_signs:
            v = patient.vital_signs[-1]
            temp = v.temperature
            map_val = (v.blood_pressure_systolic + 2 * v.blood_pressure_diastolic) / 3.0
            hr = v.heart_rate
            rr = v.respiratory_rate
        else:
            temp, map_val, hr, rr = 37.0, 85.0, 80, 18

        # --- Lab values (extract or simulate) ---
        def _find_lab(*keywords: str, default: float = 0.0) -> float:
            for lab in reversed(patient.lab_results):
                for kw in keywords:
                    if kw in lab.test_name.lower():
                        try:
                            return float(lab.value)
                        except (ValueError, TypeError):
                            pass
            return default

        sodium = _find_lab("sodium", "na", default=140.0)
        potassium = _find_lab("potassium", "k+", default=4.0)
        creatinine = _find_lab("creatinine", default=1.0)
        hematocrit = _find_lab("hematocrit", "hct", default=42.0)
        wbc = _find_lab("wbc", "white blood cell", "leukocyte", default=9.0)
        ph = _find_lab("arterial ph", "abg ph", "blood ph", default=7.40)
        pao2 = _find_lab("pao2", "arterial oxygen", default=85.0)

        # --- Acute Physiology Score (simplified point assignment) ---
        aps = 0

        # Temperature
        if temp >= 41 or temp <= 29.9:
            aps += 4
        elif temp >= 39 or temp <= 31.9:
            aps += 3
        elif temp >= 38.5 or temp <= 33.9:
            aps += 1
        # MAP
        if map_val >= 160 or map_val <= 49:
            aps += 4
        elif map_val >= 130 or map_val <= 59:
            aps += 3
        elif map_val >= 110 or map_val <= 69:
            aps += 2
        # Heart rate
        if hr >= 180 or hr <= 39:
            aps += 4
        elif hr >= 140 or hr <= 54:
            aps += 3
        elif hr >= 110 or hr <= 69:
            aps += 2
        # Respiratory rate
        if rr >= 50 or rr <= 5:
            aps += 4
        elif rr >= 35 or rr <= 9:
            aps += 3
        elif rr >= 25:
            aps += 1
        # Sodium
        if sodium >= 180 or sodium <= 110:
            aps += 4
        elif sodium >= 160 or sodium <= 119:
            aps += 3
        elif sodium >= 155 or sodium <= 129:
            aps += 2
        elif sodium >= 150:
            aps += 1
        # Potassium
        if potassium >= 7 or potassium <= 2.5:
            aps += 4
        elif potassium >= 6:
            aps += 3
        elif potassium >= 5.5 or potassium <= 2.9:
            aps += 1
        # Creatinine
        if creatinine >= 3.5:
            aps += 4
        elif creatinine >= 2.0:
            aps += 3
        elif creatinine >= 1.5:
            aps += 2
        # Hematocrit
        if hematocrit >= 60 or hematocrit < 20:
            aps += 4
        elif hematocrit >= 50 or hematocrit < 30:
            aps += 2
        elif hematocrit >= 46:
            aps += 1
        # WBC
        if wbc >= 40 or wbc < 1:
            aps += 4
        elif wbc >= 20 or wbc < 3:
            aps += 2
        elif wbc >= 15:
            aps += 1
        # pH
        if ph >= 7.7 or ph < 7.15:
            aps += 4
        elif ph >= 7.6 or ph < 7.25:
            aps += 3
        elif ph >= 7.5 or ph < 7.33:
            aps += 2
        # PaO2
        if pao2 < 55:
            aps += 4
        elif pao2 < 60:
            aps += 3
        elif pao2 < 70:
            aps += 1

        # --- Age points ---
        age = patient.age
        if age >= 75:
            age_points = 6
        elif age >= 65:
            age_points = 5
        elif age >= 55:
            age_points = 3
        elif age >= 45:
            age_points = 2
        else:
            age_points = 0

        # --- Chronic health points (check for severe organ insufficiency) ---
        def _has_chronic(*keywords: str) -> bool:
            for cond in patient.conditions:
                for kw in keywords:
                    if kw in cond.condition_name.lower():
                        return True
            return False

        chronic_points = 0
        chronic_conditions_found = []
        if _has_chronic("cirrhosis", "portal hypertension"):
            chronic_points = 5
            chronic_conditions_found.append("liver")
        if _has_chronic("nyha class iv", "heart failure class iv"):
            chronic_points = max(chronic_points, 5)
            chronic_conditions_found.append("cardiovascular")
        if _has_chronic("dialysis", "chronic renal failure", "esrd"):
            chronic_points = max(chronic_points, 5)
            chronic_conditions_found.append("renal")
        if _has_chronic("immunocompromised", "immunosuppress"):
            chronic_points = max(chronic_points, 5)
            chronic_conditions_found.append("immunocompromised")

        total = aps + age_points + chronic_points

        # Mortality estimate
        if total <= 4:
            mortality = "~4%"
        elif total <= 9:
            mortality = "~8%"
        elif total <= 14:
            mortality = "~15%"
        elif total <= 19:
            mortality = "~25%"
        elif total <= 24:
            mortality = "~40%"
        elif total <= 29:
            mortality = "~55%"
        elif total <= 34:
            mortality = "~73%"
        else:
            mortality = "~85%"

        return {
            "patient_id": patient_id,
            "apache2_score": total,
            "max_score": 71,
            "breakdown": {
                "acute_physiology_score": aps,
                "age_points": age_points,
                "chronic_health_points": chronic_points,
                "chronic_conditions_detected": chronic_conditions_found or ["none"],
            },
            "predicted_mortality": mortality,
            "interpretation": (
                "APACHE II is validated for ICU populations. Higher scores correlate with "
                "increased mortality risk. Non-operative admissions generally carry higher "
                "mortality at the same score."
            ),
        }

    # ==========================================
    # Category 10: Patient Data — Allergies, BMI, Immunizations
    # ==========================================

    @is_tool(ToolType.READ)
    def get_allergy_list(self, patient_id: str) -> dict:
        """Get a detailed allergy list for a patient including reaction type, severity, and cross-reactivity alerts.

        Args:
            patient_id: The unique patient identifier

        Returns:
            Allergy information with cross-reactivity warnings

        Raises:
            ValueError: If the patient is not found
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")

        patient = self.db.patients[patient_id]

        # Known cross-reactivity groups
        cross_reactivity_map = {
            "penicillin": ["amoxicillin", "ampicillin", "piperacillin", "nafcillin",
                           "cephalosporins (1-2% cross-reactivity)"],
            "amoxicillin": ["penicillin", "ampicillin"],
            "sulfa": ["sulfamethoxazole", "sulfasalazine", "thiazide diuretics (low risk)"],
            "nsaid": ["ibuprofen", "naproxen", "aspirin", "ketorolac", "celecoxib (lower risk)"],
            "ibuprofen": ["nsaids", "aspirin", "naproxen"],
            "aspirin": ["nsaids", "ibuprofen", "naproxen"],
            "ace inhibitor": ["lisinopril", "enalapril", "ramipril", "captopril"],
            "lisinopril": ["other ace inhibitors"],
            "codeine": ["morphine", "hydrocodone", "oxycodone (possible cross-reactivity)"],
            "morphine": ["codeine", "hydromorphone"],
            "latex": ["banana", "avocado", "kiwi", "chestnut (latex-fruit syndrome)"],
            "egg": ["influenza vaccine (most now egg-free or low-egg)"],
            "shellfish": ["iodinated contrast (historically cited but evidence is weak)"],
        }

        allergies = []
        for allergy in patient.allergies:
            allergen_lower = allergy.allergen.lower()
            cross_alerts = []
            for key, related in cross_reactivity_map.items():
                if key in allergen_lower:
                    cross_alerts = related
                    break

            allergies.append({
                "allergen": allergy.allergen,
                "reaction": allergy.reaction,
                "severity": allergy.severity,
                "cross_reactivity_alerts": cross_alerts if cross_alerts else ["No known major cross-reactivity"],
            })

        return {
            "patient_id": patient_id,
            "allergy_count": len(allergies),
            "allergies": allergies,
            "note": "Always verify allergies with the patient before prescribing. Cross-reactivity data is advisory.",
        }

    @is_tool(ToolType.READ)
    def calculate_bmi(self, patient_id: str) -> dict:
        """Calculate Body Mass Index (BMI) from patient height and weight.

        BMI = weight (kg) / height (m)^2

        Args:
            patient_id: The unique patient identifier

        Returns:
            BMI value, category, and health implications

        Raises:
            ValueError: If the patient is not found
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")

        patient = self.db.patients[patient_id]

        # Get weight from latest vital signs
        weight = None
        for v in reversed(patient.vital_signs):
            if v.weight is not None:
                weight = v.weight
                break

        # Get height from lab results or notes (often stored as a measurement)
        height_m = None
        for lab in patient.lab_results:
            if lab.test_name.lower() in ("height", "body height"):
                try:
                    val = float(lab.value)
                    # If value > 3, assume centimeters
                    height_m = val / 100.0 if val > 3 else val
                except (ValueError, TypeError):
                    pass
                break

        # Simulated defaults if not available
        if weight is None:
            weight = 78.0 if patient.sex == "male" else 65.0
        if height_m is None:
            height_m = 1.75 if patient.sex == "male" else 1.63

        bmi = round(weight / (height_m ** 2), 1)

        # Classification
        if bmi < 18.5:
            category = "Underweight"
            implications = (
                "Increased risk of nutritional deficiency, osteoporosis, and decreased immune function. "
                "Consider nutritional assessment and supplementation."
            )
        elif bmi < 25.0:
            category = "Normal weight"
            implications = "Healthy weight range. Encourage maintenance through balanced diet and regular exercise."
        elif bmi < 30.0:
            category = "Overweight"
            implications = (
                "Increased risk of cardiovascular disease, type 2 diabetes, and hypertension. "
                "Recommend lifestyle modifications: diet and exercise counseling."
            )
        elif bmi < 35.0:
            category = "Obese (Class I)"
            implications = (
                "Significantly elevated risk of metabolic syndrome, cardiovascular disease, and sleep apnea. "
                "Consider structured weight management program."
            )
        elif bmi < 40.0:
            category = "Obese (Class II)"
            implications = (
                "High risk of obesity-related complications. Pharmacotherapy may be considered alongside "
                "intensive lifestyle interventions."
            )
        else:
            category = "Obese (Class III / Morbid)"
            implications = (
                "Very high risk. Consider multidisciplinary approach including possible bariatric surgery evaluation, "
                "pharmacotherapy, and intensive lifestyle interventions."
            )

        return {
            "patient_id": patient_id,
            "bmi": bmi,
            "category": category,
            "weight_kg": weight,
            "height_m": height_m,
            "health_implications": implications,
        }

    @is_tool(ToolType.READ)
    def get_immunization_history(self, patient_id: str) -> dict:
        """Get vaccination history for a patient and recommend any due or overdue immunizations.

        Args:
            patient_id: The unique patient identifier

        Returns:
            Vaccination records and recommended immunizations

        Raises:
            ValueError: If the patient is not found
        """
        if patient_id not in self.db.patients:
            raise ValueError(f"Patient '{patient_id}' not found.")

        patient = self.db.patients[patient_id]
        age = patient.age

        # Check clinical notes for vaccination history
        vaccine_records = []
        for note in patient.clinical_notes:
            content_lower = note.content.lower()
            if any(kw in content_lower for kw in ("vaccine", "vaccination", "immuniz", "immunis")):
                vaccine_records.append({
                    "date": note.date,
                    "details": note.content,
                    "source": note.note_id,
                })

        # Simulated vaccination history based on age (realistic defaults)
        if not vaccine_records:
            base_vaccines = [
                {"vaccine": "DTaP (Diphtheria, Tetanus, Pertussis)", "date": "childhood", "status": "completed"},
                {"vaccine": "IPV (Polio)", "date": "childhood", "status": "completed"},
                {"vaccine": "MMR (Measles, Mumps, Rubella)", "date": "childhood", "status": "completed"},
                {"vaccine": "Varicella", "date": "childhood", "status": "completed"},
                {"vaccine": "Hepatitis B", "date": "childhood", "status": "completed"},
            ]
            if age >= 18:
                base_vaccines.append(
                    {"vaccine": "Tdap booster", "date": "adulthood", "status": "completed"}
                )
            if age >= 50:
                base_vaccines.append(
                    {"vaccine": "Influenza (annual)", "date": "2024-2025 season", "status": "completed"}
                )
            vaccine_records = base_vaccines

        # Determine recommended immunizations based on age and guidelines
        recommended = []
        if age >= 65:
            recommended.append({
                "vaccine": "Pneumococcal (PCV20 or PCV15 + PPSV23)",
                "reason": "Recommended for adults >= 65 years",
                "priority": "high",
            })
            recommended.append({
                "vaccine": "Zoster (Shingrix, 2-dose series)",
                "reason": "Recommended for adults >= 50 years; especially important >= 65",
                "priority": "high",
            })
        elif age >= 50:
            recommended.append({
                "vaccine": "Zoster (Shingrix, 2-dose series)",
                "reason": "Recommended for adults >= 50 years",
                "priority": "moderate",
            })

        # Annual influenza for all adults
        recommended.append({
            "vaccine": "Influenza (annual)",
            "reason": "Recommended annually for all adults",
            "priority": "routine",
        })

        # Td/Tdap every 10 years
        recommended.append({
            "vaccine": "Td/Tdap booster",
            "reason": "Recommended every 10 years for all adults",
            "priority": "routine",
        })

        # COVID-19 for high-risk
        if age >= 65 or any(
            kw in cond.condition_name.lower()
            for cond in patient.conditions
            for kw in ("diabetes", "heart failure", "copd", "immunocompromised", "chronic kidney")
        ):
            recommended.append({
                "vaccine": "COVID-19 (updated booster)",
                "reason": "Recommended for adults >= 65 or those with chronic conditions",
                "priority": "high",
            })
        else:
            recommended.append({
                "vaccine": "COVID-19 (updated booster)",
                "reason": "Recommended for all adults; annual update",
                "priority": "routine",
            })

        # Hepatitis A/B for at-risk adults
        if age >= 19:
            recommended.append({
                "vaccine": "Hepatitis A & B (if not previously vaccinated / no immunity)",
                "reason": "Recommended for adults without documented immunity",
                "priority": "low",
            })

        return {
            "patient_id": patient_id,
            "age": age,
            "vaccination_history": vaccine_records,
            "recommended_immunizations": recommended,
            "note": "Immunization recommendations based on CDC/ACIP adult schedule. Verify patient history and contraindications.",
        }

    # ==========================================
    # Category 11: Internal Reasoning & Final Answer
    # ==========================================

    @is_tool(ToolType.GENERIC)
    def think(self, thought: str) -> str:
        """Internal reasoning tool. Use this to think through complex clinical decisions before acting.

        Args:
            thought: Your clinical reasoning process

        Returns:
            Empty string (thinking is logged but produces no output)
        """
        return ""

    # ==========================================
    # Assertion helpers (for evaluation, not tools)
    # ==========================================

    def assert_diagnosis_recorded(self, patient_id: str, expected_diagnosis: str) -> bool:
        """Check if a specific diagnosis was recorded for a patient."""
        if patient_id not in self.db.patients:
            return False
        patient = self.db.patients[patient_id]
        return any(
            c.condition_name.lower() == expected_diagnosis.lower()
            for c in patient.conditions
            if c.status == "active"
        )

    def assert_lab_ordered(self, patient_id: str, expected_test: str) -> bool:
        """Check if a specific lab test was ordered for a patient."""
        return any(
            order.patient_id == patient_id and order.test_name.lower() == expected_test.lower()
            for order in self.db.lab_orders.values()
        )

    def assert_medication_prescribed(self, patient_id: str, expected_drug: str) -> bool:
        """Check if a specific medication was prescribed for a patient."""
        if patient_id not in self.db.patients:
            return False
        patient = self.db.patients[patient_id]
        return any(
            m.drug_name.lower() == expected_drug.lower() and m.end_date is None
            for m in patient.medications
        )
