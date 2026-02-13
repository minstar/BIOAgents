"""Data models for the Triage & Emergency domain.

Simulates an Emergency Department environment with:
- Patient arrivals with chief complaints
- Vital signs and initial assessments
- ESI (Emergency Severity Index) triage levels
- Emergency protocols and order sets
- Resource allocation (beds, CT, labs)
"""

import os
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from bioagents.environment.db import DB


# ── Sub-models ────────────────────────────────────────────────

class EmergencyPatient(BaseModel):
    """Patient presenting to the Emergency Department."""
    patient_id: str = Field(description="Unique patient identifier")
    name: str = Field(default="", description="Patient name (anonymized)")
    age: int = Field(description="Patient age")
    sex: Literal["M", "F"] = Field(description="Patient sex")
    chief_complaint: str = Field(description="Primary reason for ED visit")
    arrival_mode: Literal["ambulatory", "ambulance", "helicopter", "transfer"] = Field(
        default="ambulatory"
    )
    onset_time: str = Field(default="", description="Symptom onset (e.g., '2 hours ago')")
    triage_vitals: Dict = Field(default_factory=dict, description="Initial vital signs")
    allergies: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    medical_history: List[str] = Field(default_factory=list)
    presenting_symptoms: List[str] = Field(default_factory=list)
    pain_score: Optional[int] = Field(default=None, description="Pain score 0-10")
    gcs: Optional[int] = Field(default=None, description="Glasgow Coma Scale 3-15")
    correct_esi_level: Optional[int] = Field(default=None, description="Ground truth ESI 1-5")
    correct_disposition: Optional[str] = Field(default=None, description="Expected disposition")


class EDResource(BaseModel):
    """Emergency Department resource tracking."""
    resource_id: str = Field(description="Resource identifier")
    resource_type: Literal[
        "bed_resus", "bed_acute", "bed_minor", "bed_fast_track",
        "ct_scanner", "xray", "ultrasound", "lab_stat", "or_slot"
    ] = Field(description="Type of resource")
    total_capacity: int = Field(default=1)
    current_available: int = Field(default=1)
    wait_time_minutes: int = Field(default=0)


class EmergencyProtocol(BaseModel):
    """Emergency treatment protocol / order set."""
    protocol_id: str = Field(description="Protocol identifier")
    name: str = Field(description="Protocol name (e.g., 'STEMI Alert')")
    trigger_conditions: List[str] = Field(default_factory=list)
    immediate_actions: List[str] = Field(default_factory=list)
    medications: List[Dict] = Field(default_factory=list)
    diagnostics: List[str] = Field(default_factory=list)
    time_critical: bool = Field(default=False)
    time_target_minutes: Optional[int] = Field(default=None)


class TriageDecision(BaseModel):
    """A triage decision record."""
    patient_id: str
    esi_level: int = Field(description="ESI level 1-5")
    reasoning: str = Field(default="")
    bed_assignment: Optional[str] = Field(default=None)
    time_to_physician_minutes: Optional[int] = Field(default=None)


class EDStatus(BaseModel):
    """Current ED operational status."""
    total_patients: int = Field(default=0)
    waiting_patients: int = Field(default=0)
    beds_available: int = Field(default=0)
    beds_total: int = Field(default=0)
    average_wait_minutes: int = Field(default=0)
    diversion_status: bool = Field(default=False)
    pending_admissions: int = Field(default=0)


# ── Main Database ─────────────────────────────────────────────

class TriageEmergencyDB(DB):
    """Triage & Emergency domain database."""
    patients: Dict[str, EmergencyPatient] = Field(
        default_factory=dict, description="ED patients indexed by patient_id"
    )
    resources: Dict[str, EDResource] = Field(
        default_factory=dict, description="ED resources indexed by resource_id"
    )
    protocols: Dict[str, EmergencyProtocol] = Field(
        default_factory=dict, description="Emergency protocols indexed by protocol_id"
    )
    triage_decisions: List[TriageDecision] = Field(
        default_factory=list, description="Log of triage decisions"
    )
    ed_status: EDStatus = Field(
        default_factory=EDStatus, description="Current ED status"
    )


# ── Data paths ────────────────────────────────────────────────

_DOMAIN_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "data", "domains", "triage_emergency",
)
DB_PATH = os.path.join(_DOMAIN_DATA_DIR, "db.json")
POLICY_PATH = os.path.join(_DOMAIN_DATA_DIR, "policy.md")
TASKS_PATH = os.path.join(_DOMAIN_DATA_DIR, "tasks.json")


def get_db() -> TriageEmergencyDB:
    """Load the triage & emergency database."""
    return TriageEmergencyDB.load(DB_PATH)
