"""BIOAgents Medical Tools for veRL Multi-Turn GRPO Training.

Each tool_config.yaml entry creates a BIOAgentsMedicalTool instance with a specific
tool name (e.g., search_pubmed, get_patient_info). On execute, uses self.name
to dispatch to the appropriate BIOAgents domain toolkit method.

The domain is determined from the dataset's extra_info.domain field via create().
"""

import json
import logging
import os
import sys
from typing import Any, Optional
from uuid import uuid4

sys.path.insert(0, "/data/project/private/minstar/workspace/BIOAgents")

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)

# Map data domains to BIOAgents module names
DATA_TO_BIO_DOMAIN = {
    "text_qa": "medical_qa",
    "multimodal_vqa": "visual_diagnosis",
    "clinical_diagnosis": "clinical_diagnosis",
    "drug_interaction": "drug_interaction",
    "ehr_management": "ehr_management",
    "triage_emergency": "triage_emergency",
    "psychiatry": "psychiatry",
    "obstetrics": "obstetrics",
    "radiology_report": "radiology_report",
}

# Toolkit cache keyed by (domain, instance_id) to avoid stale data across samples
_toolkit_cache: dict[tuple[str, str], Any] = {}
_knowledge_backend = None


def _get_knowledge_backend():
    """Get or init shared knowledge search backend."""
    global _knowledge_backend
    if _knowledge_backend is None:
        try:
            from bioagents.tools.knowledge_tools import MedicalKnowledgeBackend
            db_path = "/data/project/private/minstar/workspace/BIOAgents/data/medical_knowledge.db"
            if os.path.exists(db_path):
                _knowledge_backend = MedicalKnowledgeBackend(db_path=db_path)
            else:
                _knowledge_backend = MedicalKnowledgeBackend()
        except Exception as e:
            logger.warning(f"Knowledge backend init failed: {e}")
    return _knowledge_backend


def _get_toolkit(bio_domain: str, db_data: dict = None, instance_id: str = ""):
    """Get or create a domain toolkit, keyed by (domain, instance) to avoid stale data."""
    cache_key = (bio_domain, instance_id)
    if cache_key in _toolkit_cache:
        return _toolkit_cache[cache_key]

    toolkit = None
    try:
        if bio_domain == "medical_qa":
            from bioagents.domains.medical_qa.tools import MedicalQATools
            from bioagents.domains.medical_qa.data_model import MedicalQADB
            db = MedicalQADB.from_dict(db_data) if db_data else MedicalQADB()
            toolkit = MedicalQATools(db)
        elif bio_domain == "clinical_diagnosis":
            from bioagents.domains.clinical_diagnosis.tools import ClinicalTools
            from bioagents.domains.clinical_diagnosis.data_model import ClinicalDB
            db = ClinicalDB.from_dict(db_data) if db_data else ClinicalDB()
            toolkit = ClinicalTools(db)
        elif bio_domain == "drug_interaction":
            from bioagents.domains.drug_interaction.tools import DrugInteractionTools
            from bioagents.domains.drug_interaction.data_model import DrugInteractionDB
            db = DrugInteractionDB.from_dict(db_data) if db_data else DrugInteractionDB()
            toolkit = DrugInteractionTools(db)
        elif bio_domain == "visual_diagnosis":
            from bioagents.domains.visual_diagnosis.tools import VisualDiagnosisTools
            from bioagents.domains.visual_diagnosis.data_model import VisualDiagnosisDB
            db = VisualDiagnosisDB.from_dict(db_data) if db_data else VisualDiagnosisDB()
            toolkit = VisualDiagnosisTools(db)
        elif bio_domain == "ehr_management":
            from bioagents.domains.ehr_management.tools import EHRTools
            from bioagents.domains.ehr_management.data_model import EHRDB
            db = EHRDB.from_dict(db_data) if db_data else EHRDB()
            toolkit = EHRTools(db)
        elif bio_domain == "triage_emergency":
            from bioagents.domains.triage_emergency.tools import TriageEmergencyTools
            from bioagents.domains.triage_emergency.data_model import TriageEmergencyDB
            db = TriageEmergencyDB.from_dict(db_data) if db_data else TriageEmergencyDB()
            toolkit = TriageEmergencyTools(db)
        elif bio_domain == "psychiatry":
            from bioagents.domains.psychiatry.tools import PsychiatryTools
            from bioagents.domains.psychiatry.data_model import PsychiatryDB
            db = PsychiatryDB.from_dict(db_data) if db_data else PsychiatryDB()
            toolkit = PsychiatryTools(db)
        elif bio_domain == "obstetrics":
            from bioagents.domains.obstetrics.tools import ObstetricsTools
            from bioagents.domains.obstetrics.data_model import ObstetricsDB
            db = ObstetricsDB.from_dict(db_data) if db_data else ObstetricsDB()
            toolkit = ObstetricsTools(db)
        elif bio_domain == "radiology_report":
            from bioagents.domains.radiology_report.tools import RadiologyReportTools
            from bioagents.domains.radiology_report.data_model import RadiologyReportDB
            db = RadiologyReportDB.from_dict(db_data) if db_data else RadiologyReportDB()
            toolkit = RadiologyReportTools(db)
    except Exception as e:
        logger.warning(f"Failed to load {bio_domain} tools: {e}")

    # Limit cache size to prevent memory leak (per-instance keying)
    if len(_toolkit_cache) > 256:
        _toolkit_cache.clear()

    _toolkit_cache[cache_key] = toolkit
    return toolkit


# Knowledge search tool names
KNOWLEDGE_TOOLS = frozenset({
    "search", "search_pubmed", "search_medical_wiki",
    "retrieve_evidence", "search_knowledge", "search_medical_knowledge",
    "browse_article", "browse_wiki_entry", "search_evidence",
    "search_guidelines", "cross_reference", "get_citation",
    "search_clinical_trials", "search_drug_interactions",
    "search_adverse_effects", "search_diagnostic_accuracy",
})

# Tools that handle 'think' and 'submit_answer' locally
SPECIAL_TOOLS = frozenset({"think", "submit_answer", "submit_report"})


class BIOAgentsMedicalTool(BaseTool):
    """Medical tool that dispatches to BIOAgents domain toolkits.

    Each instance represents one specific tool (e.g., search_pubmed, get_vital_signs).
    self.name is set from tool_schema.function.name by BaseTool.__init__.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances = {}

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """Create a tool instance for a trajectory."""
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        extra_info = create_kwargs.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except (json.JSONDecodeError, TypeError):
                extra_info = {}

        data_domain = extra_info.get("domain", "text_qa")
        bio_domain = DATA_TO_BIO_DOMAIN.get(data_domain, "medical_qa")

        self._instances[instance_id] = {
            "bio_domain": bio_domain,
            "extra_info": extra_info,
        }

        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """Execute this tool with the given parameters.

        self.name identifies which tool function to call.
        parameters are the arguments parsed from the model's tool call.
        """
        instance = self._instances.get(instance_id, {})
        bio_domain = instance.get("bio_domain", "medical_qa")
        tool_name = self.name  # From tool_schema.function.name

        # Handle special tools
        if tool_name in SPECIAL_TOOLS:
            return self._handle_special(tool_name, parameters)

        # Handle knowledge search tools
        if tool_name in KNOWLEDGE_TOOLS:
            return await self._execute_knowledge_search(tool_name, parameters)

        # Dispatch to domain toolkit (per-instance to avoid stale data)
        db_data = instance.get("extra_info", {}).get("db_data")
        toolkit = _get_toolkit(bio_domain, db_data, instance_id=instance_id)
        if toolkit is None:
            return ToolResponse(text=f"Domain '{bio_domain}' tools unavailable."), 0.0, {}

        method = getattr(toolkit, tool_name, None)
        if method is None:
            return ToolResponse(text=f"Tool '{tool_name}' not found in {bio_domain}."), 0.0, {}

        try:
            result = method(**parameters)
            if isinstance(result, (dict, list)):
                text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
            else:
                text = str(result)
            return ToolResponse(text=text), 0.0, {"tool_name": tool_name}
        except Exception as e:
            return ToolResponse(text=f"Error: {e}"), 0.0, {"error": str(e)}

    def _handle_special(
        self, tool_name: str, parameters: dict
    ) -> tuple[ToolResponse, float, dict]:
        """Handle think and submit_answer tools."""
        if tool_name == "think":
            thought = parameters.get("thought", parameters.get("reasoning", ""))
            return ToolResponse(text=f"Recorded: {thought[:200]}"), 0.0, {}
        elif tool_name in ("submit_answer", "submit_report"):
            answer = parameters.get("answer", "")
            reasoning = parameters.get("reasoning", "")
            return ToolResponse(text=f"Answer submitted: {answer}"), 0.0, {
                "answer": answer, "reasoning": reasoning
            }
        return ToolResponse(text="Unknown special tool."), 0.0, {}

    async def _execute_knowledge_search(
        self, tool_name: str, params: dict
    ) -> tuple[ToolResponse, float, dict]:
        """Execute knowledge search tools."""
        query = params.get("query", "")
        max_results = int(params.get("max_results", 5))

        backend = _get_knowledge_backend()
        if backend is None:
            return ToolResponse(text="Knowledge search unavailable."), 0.0, {}

        try:
            # Route to specific search method if available
            if hasattr(backend, tool_name):
                method = getattr(backend, tool_name)
                result = method(**params)
            else:
                result = backend.search(query, limit=max_results)

            if result:
                text = json.dumps(
                    result[:max_results] if isinstance(result, list) else result,
                    indent=2, ensure_ascii=False, default=str,
                )
            else:
                text = f"No results for: {query}"
            return ToolResponse(text=text), 0.0, {"tool_name": tool_name}
        except Exception as e:
            return ToolResponse(text=f"Search error: {e}"), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Tool reward is handled by reward_fn.py. Return 0 here."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release tool instance."""
        self._instances.pop(instance_id, None)
