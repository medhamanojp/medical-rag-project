"""
SafetyCheckTool — CrewAI tool that runs the output guardrail.
Called by the SafetyOfficerAgent to assess risk and escalation need.
"""

import json
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from src.guardrails.medical_guardrails import validate_output


class SafetyInput(BaseModel):
    prediction: dict = Field(
        ...,
        description="The prediction dict returned by DiagnosisTool."
    )
    age: int | None = Field(
        default=None,
        description="Patient age (optional). Used to elevate risk for elderly patients."
    )


class SafetyCheckTool(BaseTool):
    name: str = "SafetyCheckTool"
    description: str = (
        "Runs safety guardrails on a disease prediction. "
        "Checks confidence level, disease risk, and patient age. "
        "Returns risk level (low/moderate/high/critical), escalation flag, "
        "and a mandatory clinical disclaimer."
    )
    args_schema: type[BaseModel] = SafetyInput

    def _run(self, prediction: dict, age: int | None = None) -> str:
        result = validate_output(prediction, age=age)
        return json.dumps(result, indent=2)
