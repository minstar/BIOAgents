"""Data models for the Drug Interaction domain.

Defines the drug database schema including:
- Drug information (mechanism, side effects, contraindications)
- Drug-drug interactions
- Dosage guidelines
- Patient medication profiles
"""

import os
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from bioagents.environment.db import DB


# --- Sub-models ---


class DrugInfo(BaseModel):
    """Comprehensive drug information."""
    drug_id: str = Field(description="Unique drug identifier")
    name: str = Field(description="Generic drug name")
    brand_names: List[str] = Field(default_factory=list, description="Brand names")
    drug_class: str = Field(default="", description="Drug class (e.g., 'SSRI', 'Beta-blocker')")
    mechanism: str = Field(default="", description="Mechanism of action")
    indications: List[str] = Field(default_factory=list, description="Approved indications")
    contraindications: List[str] = Field(default_factory=list, description="Contraindications")
    common_side_effects: List[str] = Field(default_factory=list, description="Common side effects")
    serious_side_effects: List[str] = Field(default_factory=list, description="Serious/black box warnings")
    dosage_forms: List[str] = Field(default_factory=list, description="Available forms (tablet, injection, etc.)")
    typical_dosage: str = Field(default="", description="Typical adult dosage range")
    renal_adjustment: str = Field(default="", description="Renal dose adjustment guidelines")
    hepatic_adjustment: str = Field(default="", description="Hepatic dose adjustment guidelines")
    pregnancy_category: str = Field(default="", description="Pregnancy risk category")
    metabolism: str = Field(default="", description="Metabolic pathway (e.g., 'CYP3A4', 'CYP2D6')")
    half_life: str = Field(default="", description="Elimination half-life")


class Interaction(BaseModel):
    """A drug-drug interaction entry."""
    interaction_id: str = Field(description="Unique interaction identifier")
    drug_a: str = Field(description="First drug name")
    drug_b: str = Field(description="Second drug name")
    severity: Literal["minor", "moderate", "major", "contraindicated"] = Field(
        description="Interaction severity"
    )
    mechanism: str = Field(default="", description="Mechanism of the interaction")
    effect: str = Field(description="Clinical effect of the interaction")
    management: str = Field(default="", description="Clinical management recommendation")
    evidence_level: Literal["theoretical", "case_report", "clinical_study", "well_established"] = Field(
        default="clinical_study"
    )


class PatientMedProfile(BaseModel):
    """A patient's medication profile for interaction checking."""
    patient_id: str = Field(description="Patient identifier")
    current_medications: List[str] = Field(default_factory=list, description="Current drug names")
    allergies: List[str] = Field(default_factory=list, description="Known drug allergies")
    conditions: List[str] = Field(default_factory=list, description="Active medical conditions")
    age: Optional[int] = Field(default=None)
    weight_kg: Optional[float] = Field(default=None)
    renal_function: Optional[str] = Field(default=None, description="eGFR or CrCl")
    hepatic_function: Optional[str] = Field(default=None, description="Child-Pugh score if applicable")


class DrugAlternative(BaseModel):
    """An alternative drug recommendation."""
    drug_name: str = Field(description="Alternative drug name")
    drug_class: str = Field(default="", description="Drug class")
    reason: str = Field(default="", description="Reason for suggesting this alternative")
    interaction_risk: str = Field(default="low", description="Interaction risk with current medications")


# --- Main Database ---


class DrugInteractionDB(DB):
    """Drug Interaction domain database.

    Contains drug information, interaction data, and patient profiles
    for the drug interaction verification simulation.
    """
    drugs: Dict[str, DrugInfo] = Field(
        default_factory=dict,
        description="Drug information indexed by drug_id",
    )
    interactions: Dict[str, Interaction] = Field(
        default_factory=dict,
        description="Drug interactions indexed by interaction_id",
    )
    patient_profiles: Dict[str, PatientMedProfile] = Field(
        default_factory=dict,
        description="Patient medication profiles indexed by patient_id",
    )
    alternatives: Dict[str, List[DrugAlternative]] = Field(
        default_factory=dict,
        description="Drug alternatives indexed by drug_id",
    )
    interaction_check_log: List[dict] = Field(
        default_factory=list,
        description="Log of interaction checks performed",
    )


# --- Data paths ---

_DOMAIN_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "data", "domains", "drug_interaction",
)
DB_PATH = os.path.join(_DOMAIN_DATA_DIR, "db.json")
POLICY_PATH = os.path.join(_DOMAIN_DATA_DIR, "policy.md")
TASKS_PATH = os.path.join(_DOMAIN_DATA_DIR, "tasks.json")


def get_db() -> DrugInteractionDB:
    """Load the drug interaction database."""
    return DrugInteractionDB.load(DB_PATH)
