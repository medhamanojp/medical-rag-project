"""
CrewAI tool: DiagnosisTool
Runs the disease classifier on validated patient data and returns the
top-3 predicted diseases with probabilities.
"""

import json
import logging
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.guardrails.medical_guardrails import (
    InputGuardrail,
    MedicalGuardrailError,
)
from src.models.disease_classifier import get_classifier

logger = logging.getLogger(__name__)

_input_guardrail = InputGuardrail()


class DiagnosisToolInput(BaseModel):
    patient_data: dict[str, Any] = Field(
        ...,
        description=(
            "Dict containing patient vitals and symptoms. "
            "Required keys: age, temperature, heart_rate, bp_systolic, "
            "bp_diastolic, oxygen_saturation. "
            "Optional binary (0/1) symptom keys: fever, cough, "
            "shortness_of_breath, fatigue, chest_pain, headache, "
            "body_ache, loss_of_taste_smell, sore_throat, runny_nose, "
            "nausea, dizziness, frequent_urination, blurred_vision, sweating."
        ),
    )


class DiagnosisTool(BaseTool):
    name: str = "DiagnosisTool"
    description: str = (
        "Classifies the most likely disease(s) based on a patient's vitals "
        "and symptoms. Accepts a JSON-serialisable patient_data dict. "
        "Returns top-3 predictions with probabilities. "
        "Will raise an error if emergency vital thresholds are detected."
    )
    args_schema: type[BaseModel] = DiagnosisToolInput

    def _run(self, patient_data: dict[str, Any]) -> str:
        logger.info("DiagnosisTool called")
        try:
            validated = _input_guardrail.validate(patient_data)
        except MedicalGuardrailError as e:
            return json.dumps(
                {
                    "error": "EMERGENCY_GUARDRAIL",
                    "rule": e.rule,
                    "message": e.message,
                    "action_required": (
                        "STOP — escalate to emergency care immediately. "
                        "Do not proceed with automated diagnosis."
                    ),
                }
            )
        except Exception as e:
            return json.dumps({"error": "INPUT_VALIDATION_FAILED", "detail": str(e)})

        clf = get_classifier()
        result = clf.predict(validated.to_feature_dict())
        logger.info(
            "Primary diagnosis: %s (%.0f%% confidence)",
            result["primary_diagnosis"],
            result["confidence"] * 100,
        )
        return json.dumps(result, indent=2)
