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
