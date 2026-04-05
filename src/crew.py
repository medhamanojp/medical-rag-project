"""
MedicalDiagnosisCrew — assembles the three-agent CrewAI pipeline.

Pipeline (sequential):
  1. DiagnosticianAgent  → runs DiagnosisTool → top-3 predictions
  2. ExplainerAgent      → runs SHAPExplainerTool → clinical narrative
  3. SafetyOfficerAgent  → runs SafetyCheckTool → risk level + escalation

The final task output is a complete, safety-endorsed clinical report
ready to be reviewed by the treating physician.
"""

import json
import logging
import os
from typing import Any

from crewai import Crew, Process, Task
from langchain_anthropic import ChatAnthropic

from src.agents.diagnosis_agent import create_diagnostician
from src.agents.explainability_agent import create_explainer
from src.agents.safety_agent import create_safety_officer

logger = logging.getLogger(__name__)


def _build_llm() -> ChatAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or export it in your shell."
        )
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=api_key,
        temperature=0.1,   # low temperature → more deterministic clinical reasoning
        max_tokens=4096,
    )


class MedicalDiagnosisCrew:
    """
    Orchestrates the full diagnosis pipeline for a single patient case.

    Usage:
        crew = MedicalDiagnosisCrew()
        report = crew.run(patient_data)
    """

    def __init__(self, rag_context: str = ""):
        self.llm = _build_llm()
        self.rag_context = rag_context
        self.diagnostician = create_diagnostician(self.llm)
        self.explainer = create_explainer(self.llm)
        self.safety_officer = create_safety_officer(self.llm)

    def run(self, patient_data: dict[str, Any]) -> str:
        """
        Run the full pipeline for a patient case.

        Args:
            patient_data: flat dict with vitals + binary symptom flags.

        Returns:
            A comprehensive clinical report string.
        """
        patient_json = json.dumps(patient_data, indent=2)
        logger.info("Starting MedicalDiagnosisCrew for patient data: %s", patient_json)

        # ------------------------------------------------------------------
        # Task 1: Diagnosis
        # ------------------------------------------------------------------
        rag_section = (
            f"\n\nAdditional medical evidence retrieved from MedQuAD:\n{self.rag_context}"
            if self.rag_context else
            "\n\nYou may also use the MedicalRAGTool to retrieve supporting evidence."
        )

        task_diagnose = Task(
            description=(
                f"A patient presents with the following data:\n\n{patient_json}\n\n"
                "Use the DiagnosisTool with the exact patient_data dict above to "
                "get the top-3 disease predictions. Then present the results as a "
                "clear differential diagnosis list, noting each disease and its "
                "probability. If an emergency guardrail fires, report it immediately "
                "and do not proceed further."
                f"{rag_section}"
            ),
            expected_output=(
                "A ranked differential diagnosis list with the top-3 most probable "
                "diseases and their probabilities. Flag any emergency guardrail errors."
            ),
            agent=self.diagnostician,
        )

        # ------------------------------------------------------------------
        # Task 2: Explanation (depends on Task 1 output)
        # ------------------------------------------------------------------
        task_explain = Task(
            description=(
                f"Given the diagnosis from the previous step for the patient data:\n\n"
                f"{patient_json}\n\n"
                "Use the SHAPExplainerTool with the exact same patient_data dict to "
                "retrieve the SHAP feature contributions for the primary diagnosis. "
                "Then write a clear clinical narrative (3-5 sentences) explaining:\n"
                "  - Which patient findings most strongly drove the prediction\n"
                "  - Whether those findings are clinically consistent with the diagnosis\n"
                "  - Any surprising or counter-intuitive contributions that warrant scrutiny"
            ),
            expected_output=(
                "A clinical explanation narrative identifying the key drivers of the "
                "primary diagnosis with SHAP values interpreted in plain medical language."
            ),
            agent=self.explainer,
            context=[task_diagnose],
        )

        # ------------------------------------------------------------------
        # Task 3: Safety Check (depends on Tasks 1 & 2)
        # ------------------------------------------------------------------
        task_safety = Task(
            description=(
                f"Given the diagnosis and explanation from the previous steps, "
                f"perform a safety assessment for the patient:\n\n{patient_json}\n\n"
                "Use the SafetyCheckTool with:\n"
                "  - prediction: the JSON prediction result from Task 1 "
                "(must include primary_diagnosis, confidence, top_predictions)\n"
                "  - patient_data: the exact patient_data dict above\n\n"
                "Then produce the final clinical report in the following structure:\n\n"
                "## Clinical Decision Support Report\n"
                "### 1. Differential Diagnosis\n"
                "### 2. Model Reasoning (SHAP)\n"
                "### 3. Safety Assessment\n"
                "   - Risk Level\n"
                "   - Escalation Required: Yes/No\n"
                "   - Escalation Reasons (if any)\n"
                "### 4. Recommended Next Steps\n"
                "### 5. Disclaimer\n"
            ),
            expected_output=(
                "A complete, structured clinical decision support report with "
                "risk level, escalation recommendation, next steps, and the "
                "mandatory medical disclaimer."
            ),
            agent=self.safety_officer,
            context=[task_diagnose, task_explain],
        )

        # ------------------------------------------------------------------
        # Assemble and run the crew
        # ------------------------------------------------------------------
        crew = Crew(
            agents=[self.diagnostician, self.explainer, self.safety_officer],
            tasks=[task_diagnose, task_explain, task_safety],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        return str(result)
