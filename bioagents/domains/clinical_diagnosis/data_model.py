"""Data models for the Clinical Diagnosis domain.

Defines the patient database schema including:
- Patient demographics and medical history
- Lab test results and vital signs
- Medications and allergies
- Clinical notes and diagnoses
"""

from typing import Dict, List, Literal, Optional
from datetime import datetime

from pydantic import BaseModel, Field

from bioagents.environment.db import DB

# --- Sub-models ---

class Allergy(BaseModel):
    allergen: str = Field(description="Name of the allergen")
    reaction: str = Field(description="Type of reaction (e.g., anaphylaxis, rash)")
    severity: Literal["mild", "moderate", "severe"] = Field(description="Severity of the allergic reaction")


class Medication(BaseModel):
    drug_name: str = Field(description="Name of the medication")
    dosage: str = Field(description="Dosage (e.g., '500mg')")
    frequency: str = Field(description="Frequency (e.g., 'twice daily')")
    route: str = Field(default="oral", description="Route of administration")
    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date if discontinued")
    prescriber: str = Field(default="", description="Prescribing physician")


class VitalSigns(BaseModel):
    timestamp: str = Field(description="Time of measurement")
    blood_pressure_systolic: int = Field(description="Systolic BP (mmHg)")
    blood_pressure_diastolic: int = Field(description="Diastolic BP (mmHg)")
    heart_rate: int = Field(description="Heart rate (bpm)")
    temperature: float = Field(description="Body temperature (°C)")
    respiratory_rate: int = Field(description="Respiratory rate (breaths/min)")
    spo2: int = Field(description="Oxygen saturation (%)")
    weight: Optional[float] = Field(default=None, description="Weight (kg)")


class LabResult(BaseModel):
    test_id: str = Field(description="Unique test identifier")
    test_name: str = Field(description="Name of the lab test")
    category: str = Field(description="Category (e.g., 'hematology', 'chemistry', 'microbiology')")
    value: str = Field(description="Test result value")
    unit: str = Field(description="Unit of measurement")
    reference_range: str = Field(description="Normal reference range")
    flag: Optional[Literal["normal", "low", "high", "critical"]] = Field(default="normal")
    order_date: str = Field(description="Date ordered")
    result_date: str = Field(description="Date results available")
    status: Literal["pending", "completed", "cancelled"] = Field(default="completed")


class ClinicalNote(BaseModel):
    note_id: str = Field(description="Unique note identifier")
    note_type: Literal["admission", "progress", "discharge", "consultation", "procedure"] = Field(
        description="Type of clinical note"
    )
    author: str = Field(description="Author of the note")
    date: str = Field(description="Date of the note")
    content: str = Field(description="Note content")
    diagnosis_codes: Optional[List[str]] = Field(default=None, description="ICD-10 codes")


class Condition(BaseModel):
    condition_name: str = Field(description="Name of the condition")
    icd10_code: Optional[str] = Field(default=None, description="ICD-10 code")
    status: Literal["active", "resolved", "chronic"] = Field(description="Current status")
    onset_date: Optional[str] = Field(default=None, description="Date of onset")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class Patient(BaseModel):
    patient_id: str = Field(description="Unique patient identifier")
    name: str = Field(description="Patient full name")
    age: int = Field(description="Patient age")
    sex: Literal["male", "female", "other"] = Field(description="Patient sex")
    date_of_birth: str = Field(description="Date of birth (YYYY-MM-DD)")
    blood_type: Optional[str] = Field(default=None, description="Blood type (e.g., 'A+')")
    allergies: List[Allergy] = Field(default_factory=list, description="Known allergies")
    medications: List[Medication] = Field(default_factory=list, description="Current medications")
    conditions: List[Condition] = Field(default_factory=list, description="Medical conditions")
    vital_signs: List[VitalSigns] = Field(default_factory=list, description="Vital signs history")
    lab_results: List[LabResult] = Field(default_factory=list, description="Lab test results")
    clinical_notes: List[ClinicalNote] = Field(default_factory=list, description="Clinical notes")
    chief_complaint: Optional[str] = Field(default=None, description="Current chief complaint")
    family_history: Optional[str] = Field(default=None, description="Family medical history")
    social_history: Optional[str] = Field(default=None, description="Social history")


# --- Drug Interaction Database ---

class DrugInteraction(BaseModel):
    drug_a: str = Field(description="First drug")
    drug_b: str = Field(description="Second drug")
    severity: Literal["minor", "moderate", "major", "contraindicated"] = Field(
        description="Interaction severity"
    )
    description: str = Field(description="Description of the interaction")
    recommendation: str = Field(description="Clinical recommendation")


# --- Clinical Guidelines ---

class ClinicalGuideline(BaseModel):
    guideline_id: str = Field(description="Unique guideline identifier")
    condition: str = Field(description="Medical condition")
    title: str = Field(description="Guideline title")
    source: str = Field(description="Source organization (e.g., 'AHA', 'WHO')")
    summary: str = Field(description="Brief summary of the guideline")
    recommendations: List[str] = Field(description="Key recommendations")
    last_updated: str = Field(description="Last update date")


# --- Lab Test Order ---

class LabOrder(BaseModel):
    order_id: str = Field(description="Unique order identifier")
    patient_id: str = Field(description="Patient the order is for")
    test_name: str = Field(description="Name of the test ordered")
    ordered_by: str = Field(default="agent", description="Who ordered the test")
    order_date: str = Field(description="Date of order")
    status: Literal["ordered", "in_progress", "completed", "cancelled"] = Field(default="ordered")
    priority: Literal["routine", "urgent", "stat"] = Field(default="routine")


# --- Main Database ---

class ClinicalDB(DB):
    """Clinical Diagnosis domain database.
    
    Contains patient records, drug interactions, clinical guidelines,
    and lab orders for the simulation.
    """
    patients: Dict[str, Patient] = Field(
        default_factory=dict, 
        description="Patient records indexed by patient_id"
    )
    drug_interactions: List[DrugInteraction] = Field(
        default_factory=list,
        description="Known drug interactions"
    )
    clinical_guidelines: Dict[str, ClinicalGuideline] = Field(
        default_factory=dict,
        description="Clinical guidelines indexed by guideline_id"
    )
    lab_orders: Dict[str, LabOrder] = Field(
        default_factory=dict,
        description="Active lab orders indexed by order_id"
    )
    diagnosis_log: List[dict] = Field(
        default_factory=list,
        description="Log of diagnoses made by the agent"
    )


# --- Data path ---

import os

_DOMAIN_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "data", "domains", "clinical_diagnosis"
)
DB_PATH = os.path.join(_DOMAIN_DATA_DIR, "db.json")
POLICY_PATH = os.path.join(_DOMAIN_DATA_DIR, "policy.md")
TASKS_PATH = os.path.join(_DOMAIN_DATA_DIR, "tasks.json")


def get_db() -> ClinicalDB:
    """Load the clinical diagnosis database."""
    return ClinicalDB.load(DB_PATH)
