"""
SHAPExplainerTool — CrewAI tool that computes SHAP feature attributions.
Called by the ExplainerAgent to explain why the model made a prediction.
"""

import json
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from src.models.disease_classifier import get_classifier


class SHAPInput(BaseModel):
    symptoms: dict[str, int] = Field(
        ...,
        description="Same symptom dict used for prediction."
    )


class SHAPExplainerTool(BaseTool):
    name: str = "SHAPExplainerTool"
    description: str = (
        "Computes SHAP feature attributions for the primary disease prediction. "
        "Input: a dict of symptom flags (0/1). "
        "Output: top-10 symptoms ranked by their contribution to the diagnosis, "
        "with SHAP values showing which symptoms pushed the prediction and by how much."
    )
    args_schema: type[BaseModel] = SHAPInput

    def _run(self, symptoms: dict[str, int]) -> str:
        clf    = get_classifier()
        result = clf.explain(symptoms)
        return json.dumps(result, indent=2)
