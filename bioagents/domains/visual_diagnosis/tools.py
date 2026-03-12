"""Medical tools for the Visual Diagnosis domain.

Provides tools for:
- Medical image analysis (simulated)
- Image report retrieval
- Similar case search
- Patient context lookup
- Visual finding description
- Diagnostic reasoning support
"""

import re
from typing import List, Optional

from bioagents.environment.toolkit import ToolKitBase, ToolType, is_tool
from bioagents.domains.visual_diagnosis.data_model import (
    VisualDiagnosisDB,
    ImageMetadata,
    ImageReport,
    ImageFinding,
    VisualQuestion,
    PatientImageContext,
    SimilarCase,
)


def _keyword_relevance(query: str, text: str) -> float:
    """Compute keyword-overlap relevance score."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    text_tokens = set(re.findall(r"\w+", text.lower()))
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


class VisualDiagnosisTools(ToolKitBase):
    """Tools available to the visual diagnosis agent."""

    db: VisualDiagnosisDB

    def __init__(self, db: VisualDiagnosisDB) -> None:
        super().__init__(db)

    # ==========================================
    # Category 1: Image Analysis (Simulated)
    # ==========================================

    @is_tool(ToolType.READ)
    def analyze_medical_image(self, image_id: str, focus_area: str = "") -> dict:
        """Analyze a medical image and return findings. In simulation mode, returns pre-computed findings from the database.

        Args:
            image_id: The unique image identifier
            focus_area: Optional area to focus on (e.g., 'right lower lobe', 'left ventricle')

        Returns:
            Analysis results including findings, modality, and description

        Raises:
            ValueError: If the image is not found
        """
        if image_id not in self.db.images:
            raise ValueError(f"Image '{image_id}' not found in the database.")

        image = self.db.images[image_id]

        # Check if we have a report for this image
        report = None
        for r in self.db.reports.values():
            if r.image_id == image_id:
                report = r
                break

        result = {
            "image_id": image.image_id,
            "modality": image.modality,
            "body_part": image.body_part,
            "view": image.view,
            "description": image.description,
        }

        if report:
            findings = report.findings
            if focus_area:
                # Filter findings by location
                focus_lower = focus_area.lower()
                findings = [
                    f for f in findings
                    if focus_lower in f.location.lower() or focus_lower in f.description.lower()
                ] or findings  # Fall back to all findings

            result["findings"] = [f.model_dump() for f in findings]
            result["impression"] = report.impression
        else:
            result["findings"] = []
            result["impression"] = "No pre-computed analysis available. Manual review required."

        # Log the analysis
        self.db.analysis_log.append({
            "action": "analyze_medical_image",
            "image_id": image_id,
            "focus_area": focus_area,
        })

        return result

    @is_tool(ToolType.READ)
    def get_image_report(self, image_id: str) -> dict:
        """Get the full radiology/pathology report for a medical image.

        Args:
            image_id: The unique image identifier

        Returns:
            Complete report including indication, findings, impression, and technique
        """
        for report in self.db.reports.values():
            if report.image_id == image_id:
                return report.model_dump()

        return {
            "error": f"No report found for image '{image_id}'.",
            "suggestion": "Try analyze_medical_image to generate an analysis.",
        }

    # ==========================================
    # Category 2: Patient Context
    # ==========================================

    @is_tool(ToolType.READ)
    def get_patient_context(self, patient_id: str) -> dict:
        """Get patient clinical context relevant to image interpretation.

        Args:
            patient_id: The patient identifier

        Returns:
            Patient demographics, clinical history, and presenting complaint

        Raises:
            ValueError: If the patient context is not found
        """
        if patient_id not in self.db.patient_contexts:
            raise ValueError(f"Patient context '{patient_id}' not found.")

        ctx = self.db.patient_contexts[patient_id]
        return ctx.model_dump()

    # ==========================================
    # Category 3: Similar Case Search
    # ==========================================

    @is_tool(ToolType.READ)
    def search_similar_cases(self, image_id: str, max_results: int = 3) -> list:
        """Search for similar cases to compare with the current image findings.

        Args:
            image_id: The image to find similar cases for
            max_results: Maximum number of similar cases to return (default 3)

        Returns:
            List of similar cases with diagnosis, similarity score, and key features
        """
        max_results = int(max_results)

        if image_id in self.db.similar_cases:
            cases = self.db.similar_cases[image_id][:max_results]
            return [c.model_dump() for c in cases]

        # If no pre-computed similar cases, search by modality/body part
        if image_id not in self.db.images:
            return [{"message": f"Image '{image_id}' not found."}]

        current_image = self.db.images[image_id]
        similar = []

        for other_id, other_image in self.db.images.items():
            if other_id == image_id:
                continue
            if (other_image.modality == current_image.modality and
                    other_image.body_part == current_image.body_part):
                # Find the report for this image
                diagnosis = ""
                for r in self.db.reports.values():
                    if r.image_id == other_id:
                        diagnosis = r.impression
                        break
                if diagnosis:
                    similar.append({
                        "case_id": f"case_{other_id}",
                        "image_id": other_id,
                        "diagnosis": diagnosis,
                        "similarity_score": 0.5,
                        "key_features": [current_image.body_part, current_image.modality],
                    })

        return similar[:max_results] if similar else [{
            "message": "No similar cases found.",
            "suggestion": "Try broadening the search or reviewing clinical guidelines.",
        }]

    @is_tool(ToolType.READ)
    def compare_with_prior(self, current_image_id: str, prior_image_id: str) -> dict:
        """Compare current image findings with a prior study for the same patient.

        Args:
            current_image_id: Current image ID
            prior_image_id: Prior image ID for comparison

        Returns:
            Comparison analysis including changes and assessment
        """
        for img_id in [current_image_id, prior_image_id]:
            if img_id not in self.db.images:
                return {"error": f"Image '{img_id}' not found."}

        current_report = None
        prior_report = None
        for r in self.db.reports.values():
            if r.image_id == current_image_id:
                current_report = r
            elif r.image_id == prior_image_id:
                prior_report = r

        result = {
            "current_image": current_image_id,
            "prior_image": prior_image_id,
        }

        if current_report and prior_report:
            result["current_impression"] = current_report.impression
            result["prior_impression"] = prior_report.impression
            result["comparison"] = current_report.comparison or "No formal comparison available."

            # Compare finding counts
            current_findings = len(current_report.findings)
            prior_findings = len(prior_report.findings)
            if current_findings > prior_findings:
                result["assessment"] = "New findings identified compared to prior study."
            elif current_findings < prior_findings:
                result["assessment"] = "Some previously noted findings have resolved."
            else:
                result["assessment"] = "Findings appear stable compared to prior study."
        else:
            result["assessment"] = "Complete comparison not possible. One or both reports are missing."

        return result

    # ==========================================
    # Category 4: Knowledge Search
    # ==========================================

    @is_tool(ToolType.READ)
    def search_imaging_knowledge(self, query: str, modality: str = "") -> list:
        """Search the imaging knowledge base for information about findings, diagnoses, or techniques.

        Args:
            query: Search query (e.g., 'ground glass opacity differential', 'dermoscopy melanoma features')
            modality: Optional modality filter (e.g., 'xray', 'ct', 'pathology')

        Returns:
            List of relevant knowledge entries
        """
        # Search through reports and images for relevant information
        results = []

        for report in self.db.reports.values():
            combined = f"{report.impression} " + " ".join(
                f.description for f in report.findings
            )
            score = _keyword_relevance(query, combined)

            if modality:
                # Check if the image matches the modality
                img = self.db.images.get(report.image_id)
                if img and img.modality != modality:
                    continue

            if score > 0.1:
                results.append({
                    "report_id": report.report_id,
                    "image_id": report.image_id,
                    "impression": report.impression[:200],
                    "key_findings": [f.description[:100] for f in report.findings[:3]],
                    "relevance": round(score, 3),
                })

        results.sort(key=lambda x: x["relevance"], reverse=True)

        if not results:
            return [{
                "message": f"No imaging knowledge found for '{query}'.",
                "suggestion": "Try different terms or check the modality filter.",
            }]

        return results[:5]

    # ==========================================
    # Category 5: Measurement & Quantification
    # ==========================================

    @is_tool(ToolType.READ)
    def measure_lesion(self, image_id: str, lesion_id: str) -> dict:
        """Measure dimensions, area, and volume of a lesion identified in the image.

        Args:
            image_id: The unique image identifier
            lesion_id: The identifier for the specific lesion to measure

        Returns:
            Lesion measurements including length, width, depth (mm), area (mm²), and volume (mm³)
        """
        report = None
        if image_id in self.db.images:
            for r in self.db.reports.values():
                if r.image_id == image_id:
                    report = r
                    break

        # Try to match lesion to a known finding
        matched_finding = None
        if report:
            for f in report.findings:
                if f.finding_id == lesion_id or lesion_id.lower() in f.description.lower():
                    matched_finding = f
                    break

        if matched_finding:
            severity_sizes = {
                "normal": (3.0, 2.5, 2.0),
                "mild": (8.0, 6.0, 5.0),
                "moderate": (15.0, 12.0, 10.0),
                "severe": (28.0, 22.0, 18.0),
                "critical": (45.0, 38.0, 30.0),
            }
            length, width, depth = severity_sizes.get(
                matched_finding.severity, (10.0, 8.0, 6.0)
            )
            area = round(length * width * 0.785, 2)  # elliptical approximation
            volume = round(length * width * depth * 0.524, 2)  # ellipsoid approximation
            result = {
                "image_id": image_id,
                "lesion_id": lesion_id,
                "finding_description": matched_finding.description,
                "location": matched_finding.location,
                "dimensions": {
                    "length_mm": length,
                    "width_mm": width,
                    "depth_mm": depth,
                },
                "area_mm2": area,
                "volume_mm3": volume,
                "measurement_method": "simulated_from_finding_severity",
            }
        else:
            # Return simulated realistic data for unknown lesions
            result = {
                "image_id": image_id,
                "lesion_id": lesion_id,
                "finding_description": "Lesion not matched to a known finding",
                "location": "unspecified",
                "dimensions": {
                    "length_mm": 12.0,
                    "width_mm": 9.5,
                    "depth_mm": 7.0,
                },
                "area_mm2": round(12.0 * 9.5 * 0.785, 2),
                "volume_mm3": round(12.0 * 9.5 * 7.0 * 0.524, 2),
                "measurement_method": "simulated_default",
            }

        self.db.analysis_log.append({
            "action": "measure_lesion",
            "image_id": image_id,
            "lesion_id": lesion_id,
        })

        return result

    @is_tool(ToolType.READ)
    def classify_finding(self, image_id: str, classification_system: str) -> dict:
        """Apply a medical classification system to image findings.

        Supported systems: BI-RADS (breast), TI-RADS (thyroid), LI-RADS (liver),
        Fleischner (lung nodules), ABCDE (melanoma).

        Args:
            image_id: The unique image identifier
            classification_system: The classification system to apply
                (e.g., 'BI-RADS', 'TI-RADS', 'LI-RADS', 'Fleischner', 'ABCDE')

        Returns:
            Classification results including category, description, and recommendation
        """
        system = classification_system.upper().replace("-", "").replace(" ", "")

        classification_schemas = {
            "BIRADS": {
                "full_name": "Breast Imaging Reporting and Data System",
                "categories": {
                    "0": "Incomplete – Need additional imaging",
                    "1": "Negative – No finding",
                    "2": "Benign",
                    "3": "Probably benign – Short-interval follow-up suggested",
                    "4": "Suspicious – Biopsy should be considered",
                    "5": "Highly suggestive of malignancy",
                    "6": "Known biopsy-proven malignancy",
                },
            },
            "TIRADS": {
                "full_name": "Thyroid Imaging Reporting and Data System",
                "categories": {
                    "1": "Benign – No FNA",
                    "2": "Not suspicious – No FNA",
                    "3": "Mildly suspicious – FNA if ≥2.5 cm",
                    "4": "Moderately suspicious – FNA if ≥1.5 cm",
                    "5": "Highly suspicious – FNA if ≥1 cm",
                },
            },
            "LIRADS": {
                "full_name": "Liver Imaging Reporting and Data System",
                "categories": {
                    "1": "Definitely benign",
                    "2": "Probably benign",
                    "3": "Intermediate probability of malignancy",
                    "4": "Probably HCC",
                    "5": "Definitely HCC",
                    "M": "Probably or definitely malignant, not HCC specific",
                },
            },
            "FLEISCHNER": {
                "full_name": "Fleischner Society Guidelines for Pulmonary Nodules",
                "categories": {
                    "no_follow_up": "No routine follow-up (<6mm, low risk)",
                    "optional_ct_12mo": "Optional CT at 12 months (<6mm, high risk)",
                    "ct_6_12mo": "CT at 6-12 months, then 18-24 months (6-8mm)",
                    "ct_3mo_pet": "CT at 3 months, PET/CT, or tissue sampling (>8mm)",
                },
            },
            "ABCDE": {
                "full_name": "ABCDE Criteria for Melanoma Assessment",
                "categories": {
                    "A": "Asymmetry",
                    "B": "Border irregularity",
                    "C": "Color variation",
                    "D": "Diameter >6mm",
                    "E": "Evolving",
                },
            },
        }

        if system not in classification_schemas:
            return {
                "error": f"Unknown classification system '{classification_system}'.",
                "supported_systems": list(classification_schemas.keys()),
                "suggestion": "Use one of: BI-RADS, TI-RADS, LI-RADS, Fleischner, ABCDE",
            }

        schema = classification_schemas[system]

        # Determine classification based on findings
        report = None
        for r in self.db.reports.values():
            if r.image_id == image_id:
                report = r
                break

        if report and report.findings:
            max_severity = max(
                report.findings,
                key=lambda f: ["normal", "mild", "moderate", "severe", "critical"].index(
                    f.severity
                ),
            )
            severity_idx = ["normal", "mild", "moderate", "severe", "critical"].index(
                max_severity.severity
            )
            categories = list(schema["categories"].keys())
            assigned_idx = min(severity_idx, len(categories) - 1)
            assigned_category = categories[assigned_idx]
            assigned_description = schema["categories"][assigned_category]
            basis = max_severity.description
        else:
            categories = list(schema["categories"].keys())
            assigned_category = categories[0]
            assigned_description = schema["categories"][assigned_category]
            basis = "No findings available; defaulting to lowest category"

        result = {
            "image_id": image_id,
            "classification_system": schema["full_name"],
            "system_code": classification_system,
            "assigned_category": assigned_category,
            "category_description": assigned_description,
            "all_categories": schema["categories"],
            "basis": basis,
        }

        self.db.analysis_log.append({
            "action": "classify_finding",
            "image_id": image_id,
            "classification_system": classification_system,
        })

        return result

    @is_tool(ToolType.READ)
    def annotate_regions(self, image_id: str, regions: str) -> dict:
        """Mark regions of interest on the image and return annotated region list with coordinates and labels.

        Args:
            image_id: The unique image identifier
            regions: Comma-separated region descriptions to annotate
                (e.g., 'right lung apex, left costophrenic angle, cardiac silhouette')

        Returns:
            Annotated region list with coordinates and labels
        """
        region_list = [r.strip() for r in regions.split(",") if r.strip()]

        if not region_list:
            return {"error": "No regions specified. Provide comma-separated region descriptions."}

        # Gather findings to enrich annotations
        report = None
        if image_id in self.db.images:
            for r in self.db.reports.values():
                if r.image_id == image_id:
                    report = r
                    break

        annotations = []
        for idx, region_desc in enumerate(region_list):
            # Try to match region to a finding
            matched_finding = None
            if report:
                for f in report.findings:
                    if (region_desc.lower() in f.location.lower()
                            or region_desc.lower() in f.description.lower()):
                        matched_finding = f
                        break

            # Generate simulated bounding box coordinates (normalized 0-1)
            x_start = round(0.1 + (idx * 0.15) % 0.6, 3)
            y_start = round(0.1 + (idx * 0.12) % 0.5, 3)
            x_end = round(min(x_start + 0.2, 0.95), 3)
            y_end = round(min(y_start + 0.2, 0.95), 3)

            annotation = {
                "region_index": idx + 1,
                "label": region_desc,
                "bounding_box": {
                    "x_start": x_start,
                    "y_start": y_start,
                    "x_end": x_end,
                    "y_end": y_end,
                },
                "coordinate_system": "normalized_0_to_1",
            }

            if matched_finding:
                annotation["finding_id"] = matched_finding.finding_id
                annotation["finding_description"] = matched_finding.description
                annotation["severity"] = matched_finding.severity

            annotations.append(annotation)

        result = {
            "image_id": image_id,
            "total_regions": len(annotations),
            "annotations": annotations,
        }

        self.db.analysis_log.append({
            "action": "annotate_regions",
            "image_id": image_id,
            "region_count": len(region_list),
        })

        return result

    @is_tool(ToolType.READ)
    def calculate_image_metrics(self, image_id: str) -> dict:
        """Calculate quantitative image metrics including density, intensity distribution, contrast ratios, and signal-to-noise ratio.

        Args:
            image_id: The unique image identifier

        Returns:
            Quantitative image metrics: density, intensity distribution, contrast ratios, SNR
        """
        image = self.db.images.get(image_id)

        # Modality-specific realistic metric ranges
        modality_metrics = {
            "xray": {
                "mean_intensity": 128.4,
                "std_intensity": 42.7,
                "min_intensity": 0,
                "max_intensity": 255,
                "contrast_ratio": 3.2,
                "snr_db": 28.5,
                "density_category": "mixed",
            },
            "ct": {
                "mean_intensity": 45.2,
                "std_intensity": 85.3,
                "min_intensity": -1024,
                "max_intensity": 3071,
                "contrast_ratio": 4.8,
                "snr_db": 32.1,
                "density_category": "soft_tissue",
            },
            "mri": {
                "mean_intensity": 512.6,
                "std_intensity": 198.4,
                "min_intensity": 0,
                "max_intensity": 4095,
                "contrast_ratio": 5.1,
                "snr_db": 25.3,
                "density_category": "mixed",
            },
            "ultrasound": {
                "mean_intensity": 95.8,
                "std_intensity": 55.2,
                "min_intensity": 0,
                "max_intensity": 255,
                "contrast_ratio": 2.4,
                "snr_db": 18.7,
                "density_category": "echogenic",
            },
            "mammography": {
                "mean_intensity": 145.3,
                "std_intensity": 38.9,
                "min_intensity": 0,
                "max_intensity": 255,
                "contrast_ratio": 2.9,
                "snr_db": 30.2,
                "density_category": "heterogeneously_dense",
            },
        }

        if image:
            metrics = modality_metrics.get(image.modality, modality_metrics["xray"])
            result = {
                "image_id": image_id,
                "modality": image.modality,
                "body_part": image.body_part,
                "intensity_distribution": {
                    "mean": metrics["mean_intensity"],
                    "std_dev": metrics["std_intensity"],
                    "min": metrics["min_intensity"],
                    "max": metrics["max_intensity"],
                },
                "contrast_ratio": metrics["contrast_ratio"],
                "signal_to_noise_ratio_db": metrics["snr_db"],
                "density_category": metrics["density_category"],
                "measurement_method": "simulated_modality_typical",
            }
        else:
            # Return generic simulated metrics for unknown images
            result = {
                "image_id": image_id,
                "modality": "unknown",
                "body_part": "unknown",
                "intensity_distribution": {
                    "mean": 128.0,
                    "std_dev": 45.0,
                    "min": 0,
                    "max": 255,
                },
                "contrast_ratio": 3.0,
                "signal_to_noise_ratio_db": 25.0,
                "density_category": "mixed",
                "measurement_method": "simulated_default",
            }

        self.db.analysis_log.append({
            "action": "calculate_image_metrics",
            "image_id": image_id,
        })

        return result

    @is_tool(ToolType.READ)
    def get_differential_visual(self, image_id: str) -> dict:
        """Generate a visual differential diagnosis based on image findings.

        Returns a ranked list of diagnostic possibilities with likelihood estimates.

        Args:
            image_id: The unique image identifier

        Returns:
            Ranked differential diagnosis list with likelihood and supporting features
        """
        report = None
        image = self.db.images.get(image_id)
        for r in self.db.reports.values():
            if r.image_id == image_id:
                report = r
                break

        if report and report.findings:
            # Build differential based on actual findings
            primary_impression = report.impression
            finding_descriptions = [f.description for f in report.findings]
            severities = [f.severity for f in report.findings]
            max_severity = max(
                severities,
                key=lambda s: ["normal", "mild", "moderate", "severe", "critical"].index(s),
            )

            differentials = [
                {
                    "rank": 1,
                    "diagnosis": primary_impression,
                    "likelihood": "high",
                    "likelihood_pct": 65,
                    "supporting_features": finding_descriptions[:3],
                    "contra_features": [],
                },
                {
                    "rank": 2,
                    "diagnosis": f"Alternative etiology for {report.findings[0].location or 'observed region'} findings",
                    "likelihood": "moderate",
                    "likelihood_pct": 20,
                    "supporting_features": finding_descriptions[:2],
                    "contra_features": ["Primary impression better explains constellation of findings"],
                },
                {
                    "rank": 3,
                    "diagnosis": "Incidental or benign variant",
                    "likelihood": "low",
                    "likelihood_pct": 10,
                    "supporting_features": ["Some findings could represent normal variant"],
                    "contra_features": [f"Overall severity is {max_severity}"],
                },
            ]
        else:
            # Simulated differential for images without reports
            body_part = image.body_part if image else "unknown"
            differentials = [
                {
                    "rank": 1,
                    "diagnosis": f"Primary pathology of {body_part}",
                    "likelihood": "moderate",
                    "likelihood_pct": 40,
                    "supporting_features": ["Image findings pending detailed analysis"],
                    "contra_features": [],
                },
                {
                    "rank": 2,
                    "diagnosis": f"Secondary/inflammatory process of {body_part}",
                    "likelihood": "moderate",
                    "likelihood_pct": 30,
                    "supporting_features": ["Cannot exclude without further analysis"],
                    "contra_features": [],
                },
                {
                    "rank": 3,
                    "diagnosis": "Normal or benign variant",
                    "likelihood": "low",
                    "likelihood_pct": 20,
                    "supporting_features": ["Needs correlation with clinical context"],
                    "contra_features": [],
                },
            ]

        result = {
            "image_id": image_id,
            "modality": image.modality if image else "unknown",
            "differential_count": len(differentials),
            "differentials": differentials,
            "recommendation": "Correlate with clinical history and consider additional imaging if needed.",
        }

        self.db.analysis_log.append({
            "action": "get_differential_visual",
            "image_id": image_id,
        })

        return result

    @is_tool(ToolType.READ)
    def assess_image_quality(self, image_id: str) -> dict:
        """Assess the quality of a medical image including positioning, exposure, motion artifacts, and completeness.

        Flags whether a repeat study is recommended.

        Args:
            image_id: The unique image identifier

        Returns:
            Quality assessment including positioning, exposure, artifacts, completeness, and repeat recommendation
        """
        image = self.db.images.get(image_id)

        if image:
            # Modality-specific quality criteria
            modality_quality = {
                "xray": {
                    "positioning": "adequate",
                    "positioning_details": "Patient centered, no significant rotation",
                    "exposure": "optimal",
                    "exposure_details": "Adequate penetration, vertebral bodies faintly visible through cardiac silhouette",
                    "motion_artifacts": "none",
                    "completeness": "complete",
                    "completeness_details": f"Full {image.body_part} included with adequate margins",
                },
                "ct": {
                    "positioning": "adequate",
                    "positioning_details": "Patient centered in gantry",
                    "exposure": "optimal",
                    "exposure_details": "Appropriate mAs and kVp for body habitus",
                    "motion_artifacts": "none",
                    "completeness": "complete",
                    "completeness_details": f"Full {image.body_part} coverage with appropriate slice thickness",
                },
                "mri": {
                    "positioning": "adequate",
                    "positioning_details": "Proper coil placement and patient positioning",
                    "exposure": "optimal",
                    "exposure_details": "Appropriate signal intensity across sequences",
                    "motion_artifacts": "minimal",
                    "completeness": "complete",
                    "completeness_details": "All required sequences acquired",
                },
                "ultrasound": {
                    "positioning": "adequate",
                    "positioning_details": "Standard scanning planes obtained",
                    "exposure": "optimal",
                    "exposure_details": "Appropriate gain and depth settings",
                    "motion_artifacts": "none",
                    "completeness": "complete",
                    "completeness_details": f"{image.body_part} visualized in standard planes",
                },
            }

            quality = modality_quality.get(image.modality, {
                "positioning": "adequate",
                "positioning_details": "Acceptable positioning",
                "exposure": "optimal",
                "exposure_details": "Acceptable exposure",
                "motion_artifacts": "none",
                "completeness": "complete",
                "completeness_details": "Study appears complete",
            })

            overall_score = 0.92  # default good quality
            repeat_recommended = False
            repeat_reason = ""

            if quality["motion_artifacts"] not in ("none", "minimal"):
                overall_score -= 0.2
                repeat_recommended = True
                repeat_reason = "Significant motion artifacts may obscure findings"

            result = {
                "image_id": image_id,
                "modality": image.modality,
                "body_part": image.body_part,
                "view": image.view,
                "quality_assessment": quality,
                "overall_quality_score": round(overall_score, 2),
                "diagnostic_quality": "adequate" if overall_score >= 0.7 else "suboptimal",
                "repeat_recommended": repeat_recommended,
                "repeat_reason": repeat_reason,
            }
        else:
            result = {
                "image_id": image_id,
                "modality": "unknown",
                "body_part": "unknown",
                "view": "unknown",
                "quality_assessment": {
                    "positioning": "cannot_assess",
                    "exposure": "cannot_assess",
                    "motion_artifacts": "cannot_assess",
                    "completeness": "cannot_assess",
                },
                "overall_quality_score": 0.0,
                "diagnostic_quality": "not_assessable",
                "repeat_recommended": False,
                "repeat_reason": "Image not found in database; quality cannot be assessed",
            }

        self.db.analysis_log.append({
            "action": "assess_image_quality",
            "image_id": image_id,
        })

        return result

    @is_tool(ToolType.READ)
    def track_lesion_changes(self, image_id: str, prior_image_id: str) -> dict:
        """Track changes in lesion size and characteristics between current and prior study.

        Returns growth rate, change category, and detailed comparison.

        Args:
            image_id: Current image ID
            prior_image_id: Prior image ID for comparison

        Returns:
            Lesion change tracking with growth rate, change category
            (stable/growing/shrinking/new), and size comparison
        """
        current_report = None
        prior_report = None
        for r in self.db.reports.values():
            if r.image_id == image_id:
                current_report = r
            elif r.image_id == prior_image_id:
                prior_report = r

        current_image = self.db.images.get(image_id)
        prior_image = self.db.images.get(prior_image_id)

        tracked_lesions = []

        if current_report and prior_report:
            # Match findings between current and prior by location
            for current_finding in current_report.findings:
                best_match = None
                best_score = 0.0
                for prior_finding in prior_report.findings:
                    score = _keyword_relevance(
                        current_finding.location + " " + current_finding.description,
                        prior_finding.location + " " + prior_finding.description,
                    )
                    if score > best_score:
                        best_score = score
                        best_match = prior_finding

                severity_scale = {
                    "normal": 1, "mild": 2, "moderate": 3, "severe": 4, "critical": 5,
                }
                current_sev = severity_scale.get(current_finding.severity, 3)

                if best_match and best_score > 0.2:
                    prior_sev = severity_scale.get(best_match.severity, 3)
                    delta = current_sev - prior_sev

                    if delta > 0:
                        change_category = "growing"
                        growth_rate_pct = round(delta * 15.0, 1)
                    elif delta < 0:
                        change_category = "shrinking"
                        growth_rate_pct = round(delta * 15.0, 1)
                    else:
                        change_category = "stable"
                        growth_rate_pct = 0.0

                    tracked_lesions.append({
                        "lesion_description": current_finding.description,
                        "location": current_finding.location,
                        "current_severity": current_finding.severity,
                        "prior_severity": best_match.severity,
                        "change_category": change_category,
                        "growth_rate_pct": growth_rate_pct,
                        "match_confidence": round(best_score, 3),
                    })
                else:
                    tracked_lesions.append({
                        "lesion_description": current_finding.description,
                        "location": current_finding.location,
                        "current_severity": current_finding.severity,
                        "prior_severity": None,
                        "change_category": "new",
                        "growth_rate_pct": None,
                        "match_confidence": 0.0,
                    })
        elif current_report:
            for finding in current_report.findings:
                tracked_lesions.append({
                    "lesion_description": finding.description,
                    "location": finding.location,
                    "current_severity": finding.severity,
                    "prior_severity": None,
                    "change_category": "new",
                    "growth_rate_pct": None,
                    "match_confidence": 0.0,
                })
        else:
            # Simulated data when no reports available
            tracked_lesions.append({
                "lesion_description": "Unspecified lesion",
                "location": "unspecified",
                "current_severity": "moderate",
                "prior_severity": "mild",
                "change_category": "growing",
                "growth_rate_pct": 15.0,
                "match_confidence": 0.0,
            })

        # Summarize changes
        categories = [t["change_category"] for t in tracked_lesions]
        summary = {
            "total_tracked": len(tracked_lesions),
            "stable": categories.count("stable"),
            "growing": categories.count("growing"),
            "shrinking": categories.count("shrinking"),
            "new": categories.count("new"),
        }

        result = {
            "image_id": image_id,
            "prior_image_id": prior_image_id,
            "current_modality": current_image.modality if current_image else "unknown",
            "tracked_lesions": tracked_lesions,
            "summary": summary,
            "overall_assessment": (
                "Progression detected" if summary["growing"] > 0
                else "Stable disease" if summary["stable"] > 0
                else "New findings" if summary["new"] > 0
                else "Improvement noted"
            ),
        }

        self.db.analysis_log.append({
            "action": "track_lesion_changes",
            "image_id": image_id,
            "prior_image_id": prior_image_id,
        })

        return result

    # ==========================================
    # Category 6: Reasoning & Answer
    # ==========================================

    @is_tool(ToolType.GENERIC)
    def think(self, thought: str) -> str:
        """Internal reasoning tool. Use this to reason through visual findings before making a diagnosis.

        Args:
            thought: Your diagnostic reasoning about the image findings

        Returns:
            Empty string (thinking is logged but produces no output)
        """
        return ""

    @is_tool(ToolType.GENERIC)
    def submit_answer(self, answer: str, reasoning: str = "") -> str:
        """Submit your answer to the visual medical question.

        Args:
            answer: Your answer to the question
            reasoning: Your reasoning for the answer

        Returns:
            Confirmation of the submitted answer
        """
        return f"Answer '{answer}' submitted. Reasoning: {reasoning}"

    @is_tool(ToolType.WRITE)
    def record_visual_diagnosis(
        self, image_id: str, diagnosis: str, confidence: str = "moderate", reasoning: str = ""
    ) -> dict:
        """Record a visual diagnosis for an image.

        Args:
            image_id: The image being diagnosed
            diagnosis: The diagnosis based on visual findings
            confidence: Confidence level ('low', 'moderate', 'high')
            reasoning: Visual reasoning supporting the diagnosis

        Returns:
            Confirmation of the diagnosis being recorded
        """
        if image_id not in self.db.images:
            raise ValueError(f"Image '{image_id}' not found.")

        from datetime import datetime
        log_entry = {
            "image_id": image_id,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
        }
        self.db.analysis_log.append(log_entry)

        return {
            "status": "recorded",
            "image_id": image_id,
            "diagnosis": diagnosis,
            "confidence": confidence,
        }

    # ==========================================
    # Assertion helpers (for evaluation)
    # ==========================================

    def assert_correct_answer(self, question_id: str, submitted_answer: str) -> bool:
        """Check if the submitted answer is correct for a visual question."""
        if question_id not in self.db.questions:
            return False
        question = self.db.questions[question_id]
        return submitted_answer.strip().lower() == question.answer.strip().lower()

    def assert_image_analyzed(self, image_id: str) -> bool:
        """Check if an image was analyzed during the session."""
        return any(
            log.get("image_id") == image_id and log.get("action") == "analyze_medical_image"
            for log in self.db.analysis_log
        )
