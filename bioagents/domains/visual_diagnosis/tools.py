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
    # Category 5: Reasoning & Answer
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
