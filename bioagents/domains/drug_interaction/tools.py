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
    # Category 5: Clinical Pharmacology
    # ==========================================

    @is_tool(ToolType.READ)
    def check_cyp450_metabolism(self, drug_name: str) -> dict:
        """Check CYP450 enzyme metabolism pathway for a drug.

        Returns substrate, inhibitor, and inducer status across major CYP
        isoenzymes (CYP3A4, CYP2D6, CYP2C9, CYP2C19, CYP1A2, CYP2B6).

        Args:
            drug_name: The drug name (generic or brand)

        Returns:
            CYP450 metabolism profile including substrate/inhibitor/inducer status
        """
        drug_lower = drug_name.lower()

        for drug in self.db.drugs.values():
            if drug.name.lower() == drug_lower or any(
                bn.lower() == drug_lower for bn in drug.brand_names
            ):
                cyp_data = getattr(drug, "cyp450_metabolism", None)
                if cyp_data:
                    return {
                        "drug_name": drug.name,
                        "cyp450_metabolism": cyp_data,
                    }
                # Drug found but no CYP data stored – fall through to simulated
                break

        # Simulated realistic CYP450 data for drugs not in the DB
        return {
            "drug_name": drug_name,
            "cyp450_metabolism": {
                "substrates": ["CYP3A4"],
                "inhibitors": [],
                "inducers": [],
            },
            "clinical_significance": (
                "This drug is primarily metabolized by CYP3A4. "
                "Co-administration with strong CYP3A4 inhibitors (e.g., ketoconazole, "
                "clarithromycin) may increase plasma levels. Strong CYP3A4 inducers "
                "(e.g., rifampin, phenytoin) may decrease efficacy."
            ),
            "note": "Simulated data – not found in local database.",
        }

    @is_tool(ToolType.READ)
    def calculate_renal_dose_adjustment(self, drug_name: str, gfr: float) -> dict:
        """Calculate dose adjustment for renal impairment based on GFR.

        Uses standard GFR staging:
        - Normal: GFR >= 90 mL/min
        - Mild impairment: GFR 60-89 mL/min
        - Moderate impairment: GFR 30-59 mL/min
        - Severe impairment: GFR 15-29 mL/min
        - Kidney failure: GFR < 15 mL/min

        Args:
            drug_name: The drug name (generic or brand)
            gfr: Glomerular filtration rate in mL/min

        Returns:
            Adjusted dose, dosing interval, and warnings for the given GFR
        """
        drug_lower = drug_name.lower()

        drug_info = None
        for drug in self.db.drugs.values():
            if drug.name.lower() == drug_lower or any(
                bn.lower() == drug_lower for bn in drug.brand_names
            ):
                drug_info = drug
                break

        # Determine renal stage
        if gfr >= 90:
            stage = "normal"
        elif gfr >= 60:
            stage = "mild_impairment"
        elif gfr >= 30:
            stage = "moderate_impairment"
        elif gfr >= 15:
            stage = "severe_impairment"
        else:
            stage = "kidney_failure"

        if drug_info:
            typical_dosage = drug_info.typical_dosage
            renal_adj = drug_info.renal_adjustment or "No specific renal adjustment guidelines available."
        else:
            typical_dosage = "Refer to prescribing information"
            renal_adj = "No specific renal adjustment guidelines available."

        # Simulated adjustment schedule
        adjustment_map = {
            "normal": {"dose_percent": 100, "interval": "standard", "warnings": []},
            "mild_impairment": {
                "dose_percent": 100,
                "interval": "standard",
                "warnings": ["Monitor renal function periodically."],
            },
            "moderate_impairment": {
                "dose_percent": 75,
                "interval": "extend by 25%",
                "warnings": [
                    "Reduce dose or extend interval.",
                    "Monitor drug levels if applicable.",
                ],
            },
            "severe_impairment": {
                "dose_percent": 50,
                "interval": "extend by 50-100%",
                "warnings": [
                    "Significant dose reduction required.",
                    "Consider therapeutic drug monitoring.",
                    "Watch for accumulation and toxicity.",
                ],
            },
            "kidney_failure": {
                "dose_percent": 25,
                "interval": "individualize",
                "warnings": [
                    "Use with extreme caution.",
                    "Consult nephrology.",
                    "Consider dialyzability of the drug.",
                ],
            },
        }

        adj = adjustment_map[stage]
        return {
            "drug_name": drug_info.name if drug_info else drug_name,
            "gfr": gfr,
            "renal_stage": stage,
            "typical_dosage": typical_dosage,
            "renal_adjustment_guideline": renal_adj,
            "adjusted_dose_percent": adj["dose_percent"],
            "adjusted_interval": adj["interval"],
            "warnings": adj["warnings"],
        }

    @is_tool(ToolType.READ)
    def calculate_hepatic_dose_adjustment(self, drug_name: str, child_pugh: str) -> dict:
        """Calculate dose adjustment for hepatic impairment using Child-Pugh classification.

        Child-Pugh classes:
        - A (mild): score 5-6
        - B (moderate): score 7-9
        - C (severe): score 10-15

        Args:
            drug_name: The drug name (generic or brand)
            child_pugh: Child-Pugh class ('A', 'B', or 'C')

        Returns:
            Adjusted dose, clinical guidance, and warnings for hepatic impairment
        """
        child_pugh_upper = child_pugh.strip().upper()
        if child_pugh_upper not in ("A", "B", "C"):
            return {
                "error": f"Invalid Child-Pugh class '{child_pugh}'. Must be 'A', 'B', or 'C'.",
            }

        drug_lower = drug_name.lower()

        drug_info = None
        for drug in self.db.drugs.values():
            if drug.name.lower() == drug_lower or any(
                bn.lower() == drug_lower for bn in drug.brand_names
            ):
                drug_info = drug
                break

        if drug_info:
            typical_dosage = drug_info.typical_dosage
            hepatic_adj = drug_info.hepatic_adjustment or "No specific hepatic adjustment guidelines available."
        else:
            typical_dosage = "Refer to prescribing information"
            hepatic_adj = "No specific hepatic adjustment guidelines available."

        adjustment_map = {
            "A": {
                "dose_percent": 100,
                "recommendation": "No routine dose adjustment; use with caution.",
                "warnings": ["Monitor liver function tests periodically."],
            },
            "B": {
                "dose_percent": 50,
                "recommendation": "Reduce dose by approximately 50% or extend dosing interval.",
                "warnings": [
                    "Monitor for signs of hepatotoxicity.",
                    "Avoid hepatotoxic combinations.",
                    "Consider therapeutic drug monitoring.",
                ],
            },
            "C": {
                "dose_percent": 25,
                "recommendation": "Use only if benefits outweigh risks. Significant dose reduction required.",
                "warnings": [
                    "Severe hepatic impairment – risk of drug accumulation.",
                    "Consult hepatology before prescribing.",
                    "Contraindicated for many hepatically-metabolized drugs.",
                ],
            },
        }

        adj = adjustment_map[child_pugh_upper]
        return {
            "drug_name": drug_info.name if drug_info else drug_name,
            "child_pugh_class": child_pugh_upper,
            "typical_dosage": typical_dosage,
            "hepatic_adjustment_guideline": hepatic_adj,
            "adjusted_dose_percent": adj["dose_percent"],
            "recommendation": adj["recommendation"],
            "warnings": adj["warnings"],
        }

    @is_tool(ToolType.READ)
    def check_pregnancy_safety(self, drug_name: str) -> dict:
        """Check FDA pregnancy category and lactation safety for a drug.

        FDA Pregnancy Categories:
        - A: Adequate studies show no risk
        - B: Animal studies show no risk; no adequate human studies
        - C: Animal studies show adverse effects; no adequate human studies
        - D: Evidence of human fetal risk; benefits may outweigh risks
        - X: Contraindicated in pregnancy

        Args:
            drug_name: The drug name (generic or brand)

        Returns:
            Pregnancy category, risk summary, lactation safety, and safer alternatives
        """
        drug_lower = drug_name.lower()

        drug_info = None
        for drug in self.db.drugs.values():
            if drug.name.lower() == drug_lower or any(
                bn.lower() == drug_lower for bn in drug.brand_names
            ):
                drug_info = drug
                break

        pregnancy_data = getattr(drug_info, "pregnancy_category", None) if drug_info else None
        if pregnancy_data and isinstance(pregnancy_data, dict):
            return {
                "drug_name": drug_info.name,
                "pregnancy_safety": pregnancy_data,
            }

        # Simulated realistic pregnancy safety data
        return {
            "drug_name": drug_info.name if drug_info else drug_name,
            "pregnancy_category": "C",
            "risk_summary": (
                "Animal reproduction studies have shown an adverse effect on the fetus "
                "and there are no adequate and well-controlled studies in humans. "
                "Potential benefits may warrant use despite potential risks."
            ),
            "lactation_safety": {
                "excreted_in_breast_milk": "unknown",
                "recommendation": "Caution advised; consult physician before use during breastfeeding.",
            },
            "alternatives_in_pregnancy": [
                "Consult prescribing information for category A/B alternatives in the same class.",
            ],
            "note": "Simulated data – verify with current FDA labeling.",
        }

    @is_tool(ToolType.READ)
    def calculate_interaction_severity_score(self, drug_a: str, drug_b: str) -> dict:
        """Calculate a quantitative severity score (0-10) for a drug interaction pair.

        Score interpretation:
        - 0-2: Minimal / no clinical significance
        - 3-4: Minor – monitor patient
        - 5-6: Moderate – may require dose adjustment
        - 7-8: Major – use alternative if possible
        - 9-10: Contraindicated – avoid combination

        Args:
            drug_a: First drug name
            drug_b: Second drug name

        Returns:
            Severity score, clinical significance level, and management recommendation
        """
        # Check if a known interaction exists in the database
        interaction_result = self.check_interaction(drug_a, drug_b)
        severity = interaction_result.get("severity", "none")

        severity_score_map = {
            "contraindicated": 10,
            "major": 8,
            "moderate": 5,
            "minor": 3,
            "none": 0,
        }

        significance_map = {
            "contraindicated": "Contraindicated – avoid combination",
            "major": "Major – use alternative if possible",
            "moderate": "Moderate – may require dose adjustment or monitoring",
            "minor": "Minor – monitor patient",
            "none": "No clinically significant interaction expected",
        }

        score = severity_score_map.get(severity, 0)
        significance = significance_map.get(severity, "Unknown")

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity_score": score,
            "severity_label": severity,
            "clinical_significance": significance,
            "effect": interaction_result.get("effect", "No known interaction found."),
            "management": interaction_result.get(
                "management", "No special precautions needed based on available data."
            ),
        }

    @is_tool(ToolType.READ)
    def check_food_interactions(self, drug_name: str) -> dict:
        """Check food-drug interactions for a given drug.

        Common food interaction categories include grapefruit juice, dairy
        products, alcohol, high-fat meals, caffeine, and vitamin-K-rich foods.

        Args:
            drug_name: The drug name (generic or brand)

        Returns:
            Known food interactions, timing recommendations, and clinical advice
        """
        drug_lower = drug_name.lower()

        drug_info = None
        for drug in self.db.drugs.values():
            if drug.name.lower() == drug_lower or any(
                bn.lower() == drug_lower for bn in drug.brand_names
            ):
                drug_info = drug
                break

        food_data = getattr(drug_info, "food_interactions", None) if drug_info else None
        if food_data and isinstance(food_data, list):
            return {
                "drug_name": drug_info.name,
                "food_interactions": food_data,
            }

        # Simulated realistic food interaction data
        return {
            "drug_name": drug_info.name if drug_info else drug_name,
            "food_interactions": [
                {
                    "food": "Grapefruit juice",
                    "effect": "May inhibit CYP3A4, increasing drug plasma levels.",
                    "recommendation": "Avoid grapefruit juice during therapy.",
                    "severity": "moderate",
                },
                {
                    "food": "Alcohol",
                    "effect": "May potentiate CNS depression or increase hepatotoxicity risk.",
                    "recommendation": "Limit or avoid alcohol consumption.",
                    "severity": "moderate",
                },
                {
                    "food": "High-fat meals",
                    "effect": "May alter absorption rate and bioavailability.",
                    "recommendation": "Take consistently with or without food.",
                    "severity": "minor",
                },
            ],
            "general_advice": "Take with a full glass of water. Follow prescribing instructions regarding food timing.",
            "note": "Simulated data – verify with current prescribing information.",
        }

    @is_tool(ToolType.READ)
    def get_pharmacokinetics(self, drug_name: str) -> dict:
        """Get pharmacokinetic (PK) parameters for a drug.

        Parameters returned include bioavailability, half-life, volume of
        distribution (Vd), protein binding, and route of elimination.

        Args:
            drug_name: The drug name (generic or brand)

        Returns:
            Pharmacokinetic profile including absorption, distribution, metabolism, and excretion
        """
        drug_lower = drug_name.lower()

        drug_info = None
        for drug in self.db.drugs.values():
            if drug.name.lower() == drug_lower or any(
                bn.lower() == drug_lower for bn in drug.brand_names
            ):
                drug_info = drug
                break

        pk_data = getattr(drug_info, "pharmacokinetics", None) if drug_info else None
        if pk_data and isinstance(pk_data, dict):
            return {
                "drug_name": drug_info.name,
                "pharmacokinetics": pk_data,
            }

        # Simulated realistic PK data
        return {
            "drug_name": drug_info.name if drug_info else drug_name,
            "pharmacokinetics": {
                "bioavailability": "~60-80% (oral)",
                "half_life": "6-12 hours",
                "volume_of_distribution": "1.0-2.0 L/kg",
                "protein_binding": "85-95%",
                "onset_of_action": "0.5-2 hours",
                "time_to_peak": "1-3 hours",
                "metabolism": "Hepatic (primarily CYP3A4)",
                "route_of_elimination": "Renal (~60%), Fecal (~30%)",
                "active_metabolites": "Unknown / refer to prescribing information",
            },
            "note": "Simulated data – verify with current prescribing information.",
        }

    @is_tool(ToolType.READ)
    def check_therapeutic_drug_monitoring(self, drug_name: str) -> dict:
        """Check if therapeutic drug monitoring (TDM) is recommended for a drug.

        Returns therapeutic range, toxic concentration thresholds, recommended
        sampling times, and monitoring frequency.

        Args:
            drug_name: The drug name (generic or brand)

        Returns:
            TDM recommendation including therapeutic range, toxic levels, and monitoring schedule
        """
        drug_lower = drug_name.lower()

        drug_info = None
        for drug in self.db.drugs.values():
            if drug.name.lower() == drug_lower or any(
                bn.lower() == drug_lower for bn in drug.brand_names
            ):
                drug_info = drug
                break

        tdm_data = getattr(drug_info, "tdm", None) if drug_info else None
        if tdm_data and isinstance(tdm_data, dict):
            return {
                "drug_name": drug_info.name,
                "tdm": tdm_data,
            }

        # Well-known narrow therapeutic index drugs
        narrow_ti_drugs = {
            "warfarin", "digoxin", "lithium", "phenytoin", "carbamazepine",
            "valproic acid", "theophylline", "aminophylline", "vancomycin",
            "gentamicin", "tobramycin", "amikacin", "cyclosporine",
            "tacrolimus", "sirolimus", "methotrexate",
        }

        tdm_recommended = drug_lower in narrow_ti_drugs

        if tdm_recommended:
            return {
                "drug_name": drug_info.name if drug_info else drug_name,
                "tdm_recommended": True,
                "reason": "Narrow therapeutic index – routine monitoring advised.",
                "therapeutic_range": "Refer to institution-specific protocol.",
                "toxic_level": "Refer to institution-specific protocol.",
                "sampling_time": "Trough level immediately before next dose (steady-state).",
                "monitoring_frequency": (
                    "At initiation, after dose changes, when interacting drugs are "
                    "added/removed, and periodically during maintenance therapy."
                ),
            }

        return {
            "drug_name": drug_info.name if drug_info else drug_name,
            "tdm_recommended": False,
            "reason": "Not a narrow therapeutic index drug; routine TDM is generally not required.",
            "guidance": (
                "Monitor clinically for efficacy and adverse effects. "
                "Consider TDM if toxicity is suspected or response is suboptimal."
            ),
        }

    # ==========================================
    # Category 6: Reasoning & Answer
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
