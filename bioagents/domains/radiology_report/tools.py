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
    def apply_classification_system(self, study_id: str, system: str = "BI-RADS") -> str:
        """Apply a standardized imaging classification system to findings.

        Args:
            study_id: Study identifier.
            system: Classification system (BI-RADS, TI-RADS, LI-RADS, Lung-RADS, PI-RADS).

        Returns:
            Classification category with management recommendation.
        """
        classifications = {
            "BI-RADS": {"category": "BI-RADS 4A", "description": "Low suspicion for malignancy", "management": "Tissue diagnosis recommended (biopsy)", "cancer_risk": "2-10%"},
            "TI-RADS": {"category": "TI-RADS TR3", "description": "Mildly suspicious", "management": "FNA if >= 2.5cm, follow-up if 1.5-2.4cm", "points": 3},
            "LI-RADS": {"category": "LI-RADS 4", "description": "Probably HCC", "management": "Multidisciplinary discussion, consider biopsy or treatment", "diagnostic_certainty": "high"},
            "Lung-RADS": {"category": "Lung-RADS 3", "description": "Probably benign", "management": "6-month LDCT follow-up", "nodule_risk": "<5%"},
            "PI-RADS": {"category": "PI-RADS 4", "description": "High likelihood of clinically significant cancer", "management": "MRI-targeted biopsy recommended", "cancer_probability": "60-70%"},
        }
        result = classifications.get(system, {"error": f"Unknown system: {system}. Supported: BI-RADS, TI-RADS, LI-RADS, Lung-RADS, PI-RADS"})
        result["study_id"] = study_id
        result["system"] = system
        return json.dumps(result, indent=2)

    @is_tool
    def compare_with_prior_study(self, study_id: str, prior_study_id: str = "") -> str:
        """Detailed comparison with prior imaging study.

        Args:
            study_id: Current study identifier.
            prior_study_id: Prior study identifier for comparison.

        Returns:
            Comparison results with interval changes.
        """
        return json.dumps({"study_id": study_id, "prior_study_id": prior_study_id or "PRIOR_001", "interval": "6 months", "comparison": {"new_findings": [], "resolved_findings": [], "stable_findings": ["Right lower lobe nodule (5mm, unchanged)"], "changed_findings": [{"finding": "Left pleural effusion", "change": "decreased", "prior": "moderate", "current": "small", "clinical_significance": "improving"}]}, "overall_impression": "Interval improvement with decreasing left pleural effusion. Stable right lower lobe nodule."}, indent=2)

    @is_tool
    def measure_lesion_size(self, study_id: str, lesion_description: str = "") -> str:
        """Measure lesion dimensions using RECIST criteria.

        Args:
            study_id: Study identifier.
            lesion_description: Description of the lesion to measure.

        Returns:
            Lesion dimensions and RECIST classification.
        """
        return json.dumps({"study_id": study_id, "lesion": lesion_description or "target lesion", "dimensions_mm": {"long_axis": 22, "short_axis": 15, "depth": 18}, "volume_ml": 3.1, "recist": {"target_lesion": True, "sum_of_diameters": 22, "prior_sum": 20, "change_percent": 10.0, "response_category": "Stable Disease (SD)"}, "measurement_method": "Axial plane, longest diameter"}, indent=2)

    @is_tool
    def generate_critical_finding_alert(self, study_id: str, finding: str = "") -> str:
        """Generate a critical/urgent finding notification for immediate communication.

        Args:
            study_id: Study identifier.
            finding: Description of the critical finding.

        Returns:
            Alert details with communication documentation.
        """
        return json.dumps({"study_id": study_id, "alert_level": "CRITICAL", "finding": finding or "New acute finding requiring immediate attention", "communication": {"method": "Direct verbal communication", "recipient": "Ordering physician", "timestamp": "2026-02-17T10:45:00Z", "acknowledged": False}, "acr_guidelines": "ACR Practice Parameter for Communication of Diagnostic Imaging Findings", "documentation": "Critical result communicated per institutional policy"}, indent=2)

    @is_tool
    def calculate_coronary_calcium_score(self, study_id: str) -> str:
        """Calculate Agatston coronary artery calcium score.

        Args:
            study_id: Study identifier.

        Returns:
            Calcium score with risk category and percentile.
        """
        return json.dumps({"study_id": study_id, "agatston_score": 150, "risk_category": "Moderate", "risk_categories": {"0": "No identifiable disease", "1-10": "Minimal", "11-100": "Mild", "101-400": "Moderate", "401+": "Severe"}, "percentile_for_age_sex": 75, "10_year_cad_risk": "Moderate (consider statin therapy)", "vessel_scores": {"LAD": 80, "LCx": 30, "RCA": 40, "LM": 0}}, indent=2)

    @is_tool
    def get_radiation_dose(self, study_id: str) -> str:
        """Get radiation dose information for the study.

        Args:
            study_id: Study identifier.

        Returns:
            Radiation dose metrics and context.
        """
        return json.dumps({"study_id": study_id, "dlp_mgy_cm": 520, "ctdi_vol_mgy": 12.5, "effective_dose_msv": 7.8, "reference_levels": {"background_annual_msv": 3.0, "chest_xray_msv": 0.02, "ct_chest_msv": 7.0, "ct_abdomen_msv": 10.0}, "dose_comparison": "Approximately equivalent to 2.6 years of background radiation", "optimization": "Dose within diagnostic reference levels", "alara_compliant": True}, indent=2)

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
