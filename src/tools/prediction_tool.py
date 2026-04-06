"""
DiagnosisTool — CrewAI tool that runs the RandomForest classifier.
Called by the DiagnosticianAgent to get top-3 disease predictions.
"""

import json
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from src.models.disease_classifier import get_classifier


class DiagnosisInput(BaseModel):
    symptoms: dict[str, int] = Field(
        ...,
        description="Dict of symptom_name -> 1 (present) or 0 (absent)."
    )


class DiagnosisTool(BaseTool):
    name: str = "DiagnosisTool"
    description: str = (
        "Runs the disease classifier on a patient's symptoms. "
        "Input: a dict of symptom flags (0/1). "
        "Output: top-3 disease predictions with probabilities."
    )
    args_schema: type[BaseModel] = DiagnosisInput

    def _run(self, symptoms: dict[str, int]) -> str:
        clf    = get_classifier()
        result = clf.predict(symptoms)
        return json.dumps(result, indent=2)
