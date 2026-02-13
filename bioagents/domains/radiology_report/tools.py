"""Tools for the Radiology Report Generation domain.

Provides 11 tools for structured radiology report generation:
1.  get_study_info           — Study details, indication, technique
2.  get_clinical_history     — Patient clinical history
3.  get_prior_reports        — Comparison with prior studies
4.  get_report_template      — Structured template for study type
5.  analyze_findings         — Simulated image analysis (findings)
6.  search_radiology_knowledge — Reference knowledge lookup
7.  get_reporting_checklist  — Quality checklist for study type
8.  calculate_measurements   — Standardized measurements
9.  think                    — Internal reasoning
10. submit_report            — Submit structured radiology report
11. submit_answer            — Submit answer (alias for compatibility)
"""

import json
from typing import Any

from bioagents.domains.radiology_report.data_model import RadiologyReportDB
from bioagents.environment.toolkit import ToolKitBase, is_tool


class RadiologyReportTools(ToolKitBase):
    """Tool kit for Radiology Report Generation domain."""

    def __init__(self, db: RadiologyReportDB):
        super().__init__()
        self.db = db

    @is_tool
    def get_study_info(self, study_id: str) -> str:
        """Get details about a radiology study including modality, body part, indication, and technique.

        Args:
            study_id: Study identifier.

        Returns:
            JSON string with study information.
        """
        study = self.db.studies.get(study_id)
        if not study:
            return json.dumps({"error": f"Study {study_id} not found"})

        return json.dumps({
            "study_id": study.study_id,
            "modality": study.modality,
            "body_part": study.body_part,
            "indication": study.indication,
            "technique": study.technique,
            "patient_age": study.patient_age,
            "patient_sex": study.patient_sex,
            "urgency": study.urgency,
            "prior_studies": study.prior_studies,
        }, indent=2)

    @is_tool
    def get_clinical_history(self, study_id: str) -> str:
        """Get the patient's clinical history relevant to this study.

        Args:
            study_id: Study identifier.

        Returns:
            JSON string with clinical history.
        """
        study = self.db.studies.get(study_id)
        if not study:
            return json.dumps({"error": f"Study {study_id} not found"})

        return json.dumps({
            "study_id": study_id,
            "clinical_history": study.clinical_history,
            "indication": study.indication,
            "patient_age": study.patient_age,
            "patient_sex": study.patient_sex,
        }, indent=2)

    @is_tool
    def get_prior_reports(self, study_id: str) -> str:
        """Get prior radiology reports for comparison.

        Args:
            study_id: Study identifier.

        Returns:
            JSON string with prior reports.
        """
        study = self.db.studies.get(study_id)
        if not study:
            return json.dumps({"error": f"Study {study_id} not found"})

        priors = []
        for prior_id in study.prior_studies:
            prior = self.db.prior_reports.get(prior_id)
            if prior:
                priors.append({
                    "report_id": prior.report_id,
                    "date": prior.date,
                    "impression": prior.impression,
                    "key_findings": prior.key_findings,
                })

        if not priors:
            return json.dumps({"message": "No prior studies available for comparison"})

        return json.dumps({"prior_reports": priors}, indent=2)

    @is_tool
    def get_report_template(self, modality: str, body_part: str) -> str:
        """Get a structured report template for the given study type.

        Args:
            modality: Imaging modality (e.g., 'CT', 'XR', 'MRI').
            body_part: Body part (e.g., 'chest', 'head', 'abdomen').

        Returns:
            JSON string with report template and expected sections.
        """
        key = f"{modality.lower()}_{body_part.lower()}"

        template = self.db.templates.get(key)
        if not template:
            # Default template
            return json.dumps({
                "template": {
                    "sections": ["clinical_indication", "technique", "comparison", "findings", "impression"],
                    "note": f"Using default template for {modality} {body_part}",
                },
            }, indent=2)

        return json.dumps({
            "template": {
                "template_id": template.template_id,
                "sections": template.sections,
                "checklist": template.checklist,
            },
        }, indent=2)

    @is_tool
    def analyze_findings(self, study_id: str) -> str:
        """Analyze the radiology study and return detailed findings (simulated image analysis).

        Args:
            study_id: Study identifier.

        Returns:
            JSON string with findings from the study.
        """
        study = self.db.studies.get(study_id)
        if not study:
            return json.dumps({"error": f"Study {study_id} not found"})

        return json.dumps({
            "study_id": study_id,
            "modality": study.modality,
            "body_part": study.body_part,
            "findings": study.findings_description,
        }, indent=2)

    @is_tool
    def search_radiology_knowledge(self, query: str) -> str:
        """Search the radiology knowledge base for reference information.

        Args:
            query: Search query (e.g., 'pneumothorax classification', 'Fleischner criteria').

        Returns:
            JSON string with matching knowledge entries.
        """
        query_lower = query.lower()
        matches = []

        for kid, entry in self.db.knowledge_base.items():
            if (query_lower in entry.topic.lower() or
                query_lower in entry.content.lower() or
                query_lower in entry.category.lower()):
                matches.append({
                    "topic": entry.topic,
                    "content": entry.content,
                    "category": entry.category,
                })

        if not matches:
            return json.dumps({"message": f"No knowledge found for '{query}'"})

        return json.dumps({"results": matches[:5]}, indent=2)

    @is_tool
    def get_reporting_checklist(self, modality: str, body_part: str) -> str:
        """Get a quality checklist for the given study type.

        Args:
            modality: Imaging modality.
            body_part: Body part.

        Returns:
            JSON string with checklist items.
        """
        key = f"{modality.lower()}_{body_part.lower()}"
        template = self.db.templates.get(key)

        if template and template.checklist:
            return json.dumps({"checklist": template.checklist}, indent=2)

        # Default checklists by modality
        defaults = {
            "XR": ["Alignment", "Bones", "Cartilage/joints", "Soft tissues", "Comparison with prior"],
            "CT": ["Systematic review of all organs", "Vascular structures", "Lymph nodes", "Bones", "Comparison with prior"],
            "MRI": ["Signal characteristics", "Enhancement pattern", "Diffusion changes", "Mass effect", "Comparison with prior"],
        }

        checklist = defaults.get(modality.upper(), ["Systematic review", "Comparison with prior"])
        return json.dumps({"checklist": checklist, "note": "Default checklist"}, indent=2)

    @is_tool
    def calculate_measurements(self, study_id: str, measurement_type: str) -> str:
        """Get standardized measurements from the study (simulated).

        Args:
            study_id: Study identifier.
            measurement_type: Type of measurement (e.g., 'cardiac_size', 'nodule_size', 'aortic_diameter').

        Returns:
            JSON string with measurements.
        """
        study = self.db.studies.get(study_id)
        if not study:
            return json.dumps({"error": f"Study {study_id} not found"})

        # Simulated measurements based on study findings
        return json.dumps({
            "study_id": study_id,
            "measurement_type": measurement_type,
            "note": f"Measurements for {measurement_type} should be derived from image analysis",
            "recommendation": "Use RECIST criteria for tumor measurements, Fleischner for nodules",
        }, indent=2)

    @is_tool
    def think(self, thought: str) -> str:
        """Record internal reasoning about the radiology findings.

        Args:
            thought: Your interpretation and reasoning.

        Returns:
            Acknowledgment.
        """
        return json.dumps({"thought_recorded": True, "content": thought[:500]})

    @is_tool
    def submit_report(self, study_id: str, indication: str, technique: str, comparison: str, findings: str, impression: str) -> str:
        """Submit a structured radiology report.

        Args:
            study_id: Study identifier.
            indication: Clinical indication section.
            technique: Technique/protocol section.
            comparison: Comparison with prior studies section.
            findings: Detailed findings section.
            impression: Impression/conclusion section.

        Returns:
            Confirmation of submitted report.
        """
        report = {
            "study_id": study_id,
            "indication": indication,
            "technique": technique,
            "comparison": comparison,
            "findings": findings,
            "impression": impression,
        }
        self.db.generated_reports.append(report)

        return json.dumps({
            "submitted": True,
            "report_sections": list(report.keys()),
            "findings_length": len(findings),
            "impression_length": len(impression),
        })

    @is_tool
    def submit_answer(self, answer: str) -> str:
        """Submit a text answer (compatibility alias for submit_report).

        Args:
            answer: Your complete radiology report or assessment.

        Returns:
            Confirmation.
        """
        return json.dumps({"submitted": True, "answer": answer[:200] + "..."})
