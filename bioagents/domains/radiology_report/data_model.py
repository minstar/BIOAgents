"""Data models for the Radiology Report Generation domain.

Simulates a radiology reporting workflow with:
- Medical images with findings
- Structured report templates
- Clinical history and prior studies
- Report quality assessment
"""

import os
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from bioagents.environment.db import DB


# ── Sub-models ────────────────────────────────────────────────

class RadiologyStudy(BaseModel):
    """A radiology study/exam."""
    study_id: str = Field(description="Unique study identifier")
    modality: Literal[
        "CR", "CT", "MRI", "US", "NM", "XR", "MG", "FL"
    ] = Field(description="Imaging modality (DICOM abbreviation)")
    body_part: str = Field(description="Body part examined")
    indication: str = Field(description="Clinical indication")
    technique: str = Field(default="", description="Imaging technique/protocol")
    patient_id: str = Field(default="", description="Patient identifier")
    patient_age: Optional[int] = Field(default=None)
    patient_sex: Optional[str] = Field(default=None)
    clinical_history: str = Field(default="", description="Relevant clinical history")
    prior_studies: List[str] = Field(default_factory=list, description="Prior study IDs")
    findings_description: str = Field(default="", description="Ground truth findings text")
    impression_reference: str = Field(default="", description="Reference impression")
    urgency: Literal["routine", "urgent", "stat"] = Field(default="routine")


class ReportTemplate(BaseModel):
    """Structured radiology report template."""
    template_id: str
    modality: str
    body_part: str
    sections: List[str] = Field(
        default_factory=lambda: [
            "clinical_indication",
            "technique",
            "comparison",
            "findings",
            "impression",
        ]
    )
    checklist: List[str] = Field(
        default_factory=list,
        description="Key items to check/mention for this study type",
    )


class PriorReport(BaseModel):
    """A prior radiology report for comparison."""
    report_id: str
    study_id: str
    date: str
    impression: str
    key_findings: List[str] = Field(default_factory=list)


class RadiologyKnowledge(BaseModel):
    """Reference knowledge for radiological interpretation."""
    topic: str
    content: str
    category: str = Field(default="general")


# ── Main Database ─────────────────────────────────────────────

class RadiologyReportDB(DB):
    """Radiology Report Generation database."""
    studies: Dict[str, RadiologyStudy] = Field(
        default_factory=dict, description="Radiology studies"
    )
    templates: Dict[str, ReportTemplate] = Field(
        default_factory=dict, description="Report templates"
    )
    prior_reports: Dict[str, PriorReport] = Field(
        default_factory=dict, description="Prior reports for comparison"
    )
    knowledge_base: Dict[str, RadiologyKnowledge] = Field(
        default_factory=dict, description="Radiology reference knowledge"
    )
    generated_reports: List[dict] = Field(
        default_factory=list, description="Log of generated reports"
    )


# ── Data paths ────────────────────────────────────────────────

_DOMAIN_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "data", "domains", "radiology_report",
)
DB_PATH = os.path.join(_DOMAIN_DATA_DIR, "db.json")
POLICY_PATH = os.path.join(_DOMAIN_DATA_DIR, "policy.md")
TASKS_PATH = os.path.join(_DOMAIN_DATA_DIR, "tasks.json")


def get_db() -> RadiologyReportDB:
    """Load the radiology report database."""
    return RadiologyReportDB.load(DB_PATH)
