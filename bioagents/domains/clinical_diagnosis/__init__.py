"""Clinical Diagnosis Domain for BIOAgents.

This domain simulates a clinical diagnostic environment where an agent
acts as a physician assistant, using tools to:
- Look up patient information
- Order and review lab tests
- Check vital signs
- Search medical literature
- Make differential diagnoses
- Prescribe treatments
"""

from bioagents.domains.clinical_diagnosis.data_model import ClinicalDB, get_db
from bioagents.domains.clinical_diagnosis.tools import ClinicalTools
from bioagents.domains.clinical_diagnosis.environment import get_environment, get_tasks

__all__ = ["ClinicalDB", "ClinicalTools", "get_db", "get_environment", "get_tasks"]
