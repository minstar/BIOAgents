"""Medical tools for the Drug Interaction domain.

Provides tools for:
- Drug information lookup
- Drug-drug interaction checking
- Patient medication profile review
- Alternative drug search
- Dosage verification
- Interaction risk assessment
"""

import re
from typing import List, Optional

from bioagents.environment.toolkit import ToolKitBase, ToolType, is_tool
from bioagents.domains.drug_interaction.data_model import (
    DrugInteractionDB,
    DrugInfo,
    Interaction,
    PatientMedProfile,
)


class DrugInteractionTools(ToolKitBase):
    """Tools available to the drug interaction agent."""

    db: DrugInteractionDB

    def __init__(self, db: DrugInteractionDB) -> None:
        super().__init__(db)

    # ==========================================
    # Category 1: Drug Information
    # ==========================================

    @is_tool(ToolType.READ)
    def get_drug_info(self, drug_name: str) -> dict:
        """Get comprehensive information about a drug including mechanism, indications, side effects, and metabolism.

        Args:
            drug_name: The drug name (generic or brand)

        Returns:
            Drug information including mechanism, indications, side effects, and dosage
        """
        drug_lower = drug_name.lower()

        for drug_id, drug in self.db.drugs.items():
            if drug.name.lower() == drug_lower:
                return drug.model_dump()
            if any(bn.lower() == drug_lower for bn in drug.brand_names):
                return drug.model_dump()

        return {
            "error": f"Drug '{drug_name}' not found in the database.",
            "suggestion": "Try the generic name or check spelling.",
        }

    @is_tool(ToolType.READ)
    def search_drugs_by_class(self, drug_class: str) -> list:
        """Search for drugs by their class (e.g., 'SSRI', 'ACE inhibitor', 'beta-blocker').

        Args:
            drug_class: The drug class to search for

        Returns:
            List of drugs in the specified class
        """
        class_lower = drug_class.lower()
        results = []
        for drug in self.db.drugs.values():
            if class_lower in drug.drug_class.lower():
                results.append({
                    "drug_id": drug.drug_id,
                    "name": drug.name,
                    "drug_class": drug.drug_class,
                    "mechanism": drug.mechanism[:150],
                })
        if not results:
            return [{"message": f"No drugs found in class '{drug_class}'."}]
        return results

    # ==========================================
    # Category 2: Interaction Checking
    # ==========================================

    @is_tool(ToolType.READ)
    def check_interaction(self, drug_a: str, drug_b: str) -> dict:
        """Check for known interactions between two drugs.

        Args:
            drug_a: First drug name
            drug_b: Second drug name

        Returns:
            Interaction details including severity, mechanism, effect, and management
        """
        a_lower = drug_a.lower()
        b_lower = drug_b.lower()

        for interaction in self.db.interactions.values():
            ia = interaction.drug_a.lower()
            ib = interaction.drug_b.lower()
            if (ia == a_lower and ib == b_lower) or (ia == b_lower and ib == a_lower):
                result = interaction.model_dump()
                self.db.interaction_check_log.append({
                    "action": "check_interaction",
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "found": True,
                    "severity": interaction.severity,
                })
                return result

        self.db.interaction_check_log.append({
            "action": "check_interaction",
            "drug_a": drug_a,
            "drug_b": drug_b,
            "found": False,
        })

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": "none",
            "effect": "No known interaction found.",
            "management": "No special precautions needed based on available data.",
        }

    @is_tool(ToolType.READ)
    def check_all_interactions(self, patient_id: str, new_drug: str = "") -> list:
        """Check all drug interactions for a patient's medication list. Optionally checks a new drug against existing medications.

        Args:
            patient_id: Patient identifier to check medication profile
            new_drug: Optional new drug to check against existing medications

        Returns:
            List of all found interactions with severity and management
        """
        if patient_id not in self.db.patient_profiles:
            raise ValueError(f"Patient profile '{patient_id}' not found.")

        profile = self.db.patient_profiles[patient_id]
        meds = list(profile.current_medications)
        if new_drug:
            meds.append(new_drug)

        found_interactions = []
        checked_pairs = set()

        for i, med_a in enumerate(meds):
            for j, med_b in enumerate(meds):
                if i >= j:
                    continue
                pair_key = tuple(sorted([med_a.lower(), med_b.lower()]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                result = self.check_interaction(med_a, med_b)
                if result.get("severity", "none") != "none":
                    found_interactions.append(result)

        if not found_interactions:
            return [{
                "message": "No interactions found among the patient's medications.",
                "medications_checked": meds,
            }]

        # Sort by severity
        severity_order = {"contraindicated": 0, "major": 1, "moderate": 2, "minor": 3}
        found_interactions.sort(
            key=lambda x: severity_order.get(x.get("severity", "minor"), 4)
        )

        return found_interactions

    # ==========================================
    # Category 3: Patient Profile
    # ==========================================

    @is_tool(ToolType.READ)
    def get_patient_medications(self, patient_id: str) -> dict:
        """Get a patient's current medication list and relevant health information.

        Args:
            patient_id: Patient identifier

        Returns:
            Patient medication profile including allergies, conditions, and organ function

        Raises:
            ValueError: If the patient profile is not found
        """
        if patient_id not in self.db.patient_profiles:
            raise ValueError(f"Patient profile '{patient_id}' not found.")

        return self.db.patient_profiles[patient_id].model_dump()

    # ==========================================
    # Category 4: Alternatives & Dosage
    # ==========================================

    @is_tool(ToolType.READ)
    def search_alternatives(self, drug_name: str) -> list:
        """Search for alternative drugs when an interaction or allergy is identified.

        Args:
            drug_name: The drug to find alternatives for

        Returns:
            List of alternative drugs with rationale and interaction risk
        """
        drug_lower = drug_name.lower()

        # Find the drug ID
        drug_id = None
        for did, drug in self.db.drugs.items():
            if drug.name.lower() == drug_lower:
                drug_id = did
                break

        if drug_id and drug_id in self.db.alternatives:
            return [a.model_dump() for a in self.db.alternatives[drug_id]]

        # Fallback: find drugs in the same class
        original_drug = None
        for drug in self.db.drugs.values():
            if drug.name.lower() == drug_lower:
                original_drug = drug
                break

        if original_drug and original_drug.drug_class:
            same_class = [
                {
                    "drug_name": d.name,
                    "drug_class": d.drug_class,
                    "reason": f"Same drug class ({d.drug_class})",
                    "interaction_risk": "unknown",
                }
                for d in self.db.drugs.values()
                if d.drug_class == original_drug.drug_class and d.name.lower() != drug_lower
            ]
            if same_class:
                return same_class

        return [{"message": f"No alternatives found for '{drug_name}'."}]

    @is_tool(ToolType.READ)
    def check_dosage(self, drug_name: str, patient_id: str = "") -> dict:
        """Check dosage guidelines for a drug, considering patient-specific factors if available.

        Args:
            drug_name: The drug to check dosage for
            patient_id: Optional patient ID for personalized dosing

        Returns:
            Dosage information including adjustments for renal/hepatic function
        """
        drug_lower = drug_name.lower()
        drug_info = None
        for drug in self.db.drugs.values():
            if drug.name.lower() == drug_lower:
                drug_info = drug
                break

        if not drug_info:
            return {"error": f"Drug '{drug_name}' not found."}

        result = {
            "drug_name": drug_info.name,
            "typical_dosage": drug_info.typical_dosage,
            "dosage_forms": drug_info.dosage_forms,
        }

        if patient_id and patient_id in self.db.patient_profiles:
            profile = self.db.patient_profiles[patient_id]
            if profile.renal_function:
                result["renal_adjustment"] = drug_info.renal_adjustment or "No specific renal adjustment guidelines."
                result["patient_renal_function"] = profile.renal_function
            if profile.hepatic_function:
                result["hepatic_adjustment"] = drug_info.hepatic_adjustment or "No specific hepatic adjustment guidelines."
                result["patient_hepatic_function"] = profile.hepatic_function

        return result

    # ==========================================
    # Category 5: Reasoning & Answer
    # ==========================================

    @is_tool(ToolType.GENERIC)
    def think(self, thought: str) -> str:
        """Internal reasoning tool. Use to reason through drug interaction decisions.

        Args:
            thought: Your pharmacological reasoning process

        Returns:
            Empty string (thinking is logged)
        """
        return ""

    @is_tool(ToolType.GENERIC)
    def submit_answer(self, answer: str, reasoning: str = "") -> str:
        """Submit your final recommendation regarding drug interactions.

        Args:
            answer: Your recommendation (e.g., 'safe to prescribe', 'contraindicated', 'dose adjustment needed')
            reasoning: Your pharmacological reasoning

        Returns:
            Confirmation of the submitted answer
        """
        return f"Recommendation '{answer}' submitted. Reasoning: {reasoning}"

    # ==========================================
    # Assertion helpers
    # ==========================================

    def assert_interaction_found(self, drug_a: str, drug_b: str) -> bool:
        """Check if a specific interaction was checked during the session."""
        return any(
            log.get("action") == "check_interaction"
            and log.get("drug_a", "").lower() in [drug_a.lower(), drug_b.lower()]
            and log.get("drug_b", "").lower() in [drug_a.lower(), drug_b.lower()]
            for log in self.db.interaction_check_log
        )
