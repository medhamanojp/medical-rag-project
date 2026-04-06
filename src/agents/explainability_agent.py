"""
ExplainerAgent — acts as a clinical AI informatics specialist.
Translates raw SHAP values into a human-readable clinical narrative.
"""

from crewai import Agent
from src.tools.shap_tool import SHAPExplainerTool
from src.tools.rag_tool import MedicalRAGTool


def build_explainer_agent(llm) -> Agent:
    return Agent(
        role="Clinical AI Explainability Specialist",
        goal=(
            "Explain in plain clinical language exactly which symptoms "
            "drove the diagnosis and why, using SHAP feature attributions. "
            "Make the reasoning transparent and understandable to both "
            "doctors and patients."
        ),
        backstory=(
            "You specialise in clinical informatics and AI explainability. "
            "Your job is to bridge the gap between machine learning outputs "
            "and clinical understanding. You translate SHAP values into "
            "clear narratives: which symptoms were most important, which "
            "ruled things out, and what the model was most uncertain about. "
            "You never use jargon without explanation."
        ),
        tools=[SHAPExplainerTool(), MedicalRAGTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
