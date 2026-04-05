"""
CrewAI tool: SHAPExplainerTool
Generates a human-readable SHAP-based explanation of why the model
predicted the primary disease for a given patient.
"""

import json
import logging
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.models.disease_classifier import get_classifier

logger = logging.getLogger(__name__)


class SHAPToolInput(BaseModel):
    patient_data: dict[str, Any] = Field(
        ...,
        description=(
            "Same patient_data dict used in DiagnosisTool. "
            "Returns the top-10 SHAP feature contributions that drove "
            "the primary prediction."
        ),
    )


class SHAPExplainerTool(BaseTool):
    name: str = "SHAPExplainerTool"
    description: str = (
        "Explains *why* the model predicted a specific disease by computing "
        "SHAP (SHapley Additive exPlanations) values. "
        "Returns the top contributing features with their SHAP values and "
        "the patient's actual value for each feature. "
        "Positive SHAP values push toward the predicted disease; "
        "negative values push away from it."
    )
    args_schema: type[BaseModel] = SHAPToolInput

    def _run(self, patient_data: dict[str, Any]) -> str:
        logger.info("SHAPExplainerTool called")
        try:
            clf = get_classifier()
            explanation = clf.explain(patient_data)
        except Exception as e:
            logger.exception("SHAP explanation failed")
            return json.dumps({"error": "SHAP_FAILED", "detail": str(e)})

        # Build a readable narrative alongside the raw data
        lines = [
            f"Primary diagnosis: {explanation['primary_diagnosis']}",
            f"Base probability (population average): {explanation['base_value']:.4f}",
            "",
            "Top feature contributions (+ pushes toward diagnosis, - pushes away):",
        ]
        for i, fc in enumerate(explanation["feature_contributions"], 1):
            direction = "▲" if fc["shap_value"] > 0 else "▼"
            lines.append(
                f"  {i:2}. {fc['feature']:25s} "
                f"SHAP={fc['shap_value']:+.4f} {direction}  "
                f"(patient value: {fc['patient_value']})"
            )

        explanation["narrative"] = "\n".join(lines)
        return json.dumps(explanation, indent=2)
