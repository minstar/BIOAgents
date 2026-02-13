"""Agent module for BIOAgents Healthcare AI GYM.

Provides:
- PatientAgent: Simulates realistic patients for multi-agent encounters
- (Future) DoctorAgent: The model being trained/evaluated
- (Future) NurseAgent: Assists with triage and vitals
"""

from bioagents.agents.patient_agent import (
    PatientAgent,
    PatientPersonality,
    PatientBias,
    ClinicalCase,
    SymptomLayer,
    get_clinical_cases,
    evaluate_doctor_performance,
)

__all__ = [
    "PatientAgent",
    "PatientPersonality",
    "PatientBias",
    "ClinicalCase",
    "SymptomLayer",
    "get_clinical_cases",
    "evaluate_doctor_performance",
]
