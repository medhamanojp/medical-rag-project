"""
ExplainabilityAgent — SHAP-based reasoning agent.

Role: Medical AI Explainability Specialist.
Goal: Translate SHAP feature contributions into plain clinical language
      so the doctor understands *why* the model produced its diagnosis.
Tools: SHAPExplainerTool
"""

from crewai import Agent

from src.tools.shap_tool import SHAPExplainerTool
from src.tools.rag_tool import MedicalRAGTool


def create_explainer(llm) -> Agent:
    return Agent(
        role="Medical AI Explainability Specialist",
        goal=(
            "Use the SHAPExplainerTool to retrieve the feature contributions "
            "behind the primary diagnosis, then use MedicalRAGTool to find "
            "corroborating clinical evidence. Translate SHAP values into "
            "a clear, evidence-backed clinical narrative the treating "
            "physician can understand and audit."
        ),
        backstory=(
            "You are a clinical informatics expert who bridges machine learning "
            "and bedside medicine. You have deep knowledge of both SHAP "
            "explainability methods and clinical physiology. You always explain "
            "which patient findings most strongly influenced the model's decision, "
            "cite supporting medical evidence from the knowledge base, and flag "
            "any surprising or counter-intuitive SHAP contributions for human review."
        ),
        tools=[SHAPExplainerTool(), MedicalRAGTool()],
        llm=llm,
        verbose=True,
        max_iter=5,
    )
