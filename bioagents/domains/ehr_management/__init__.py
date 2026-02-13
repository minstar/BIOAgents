"""EHR Management Domain — Electronic Health Record analysis and clinical decision support.

Tools: get_patient_summary, get_admission_history, get_lab_results, get_lab_trend,
       get_vital_signs, detect_vital_alerts, get_medication_orders, get_clinical_scores,
       get_quality_indicators, get_procedures, get_discharge_summary, lookup_icd_code,
       think, submit_answer
"""

from bioagents.domains.ehr_management.data_model import EHRDB, get_db
from bioagents.domains.ehr_management.tools import EHRTools
from bioagents.domains.ehr_management.environment import get_environment, get_tasks

__all__ = ["EHRDB", "EHRTools", "get_db", "get_environment", "get_tasks"]
