"""
CrewAI tool: SafetyCheckTool
Applies the output guardrail to a prediction result and returns a
structured safety assessment including risk level and escalation flags.
"""

import json
import logging
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.guardrails.medical_guardrails import (
    InputGuardrail,
    OutputGuardrail,
    PatientInput,
)

logger = logging.getLogger(__name__)

_input_guardrail = InputGuardrail()
_output_guardrail = OutputGuardrail()


class SafetyCheckInput(BaseModel):
    prediction: dict[str, Any] = Field(
        ...,
        description=(
            "The raw prediction dict returned by DiagnosisTool "
            "(keys: primary_diagnosis, confidence, top_predictions)."
        ),
    )
    patient_data: dict[str, Any] = Field(
        ...,
        description="The same patient_data dict used in DiagnosisTool.",
    )


class SafetyCheckTool(BaseTool):
    name: str = "SafetyCheckTool"
    description: str = (
        "Applies clinical safety guardrails to a diagnosis prediction. "
        "Evaluates confidence thresholds, high-risk disease flags, patient "
        "age risk, and vital sign alerts. "
        "Returns a risk_level (low/moderate/high/critical), "
        "requires_escalation flag, escalation_reasons, and a mandatory "
        "medical disclaimer."
    )
    args_schema: type[BaseModel] = SafetyCheckInput

    def _run(self, prediction: dict[str, Any], patient_data: dict[str, Any]) -> str:
        logger.info("SafetyCheckTool called for diagnosis: %s", prediction.get("primary_diagnosis"))
        try:
            # Re-validate patient to get typed object for the output guardrail
            patient: PatientInput = _input_guardrail.validate(patient_data)
            output = _output_guardrail.validate(prediction, patient)
        except Exception as e:
            logger.exception("Safety check failed")
            return json.dumps({"error": "SAFETY_CHECK_FAILED", "detail": str(e)})

        result = output.model_dump()
        result["risk_level"] = output.risk_level.value  # serialize enum
        return json.dumps(result, indent=2)
