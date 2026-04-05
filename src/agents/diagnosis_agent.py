"""
DiagnosticianAgent — primary disease classification agent.

Role: Acts as a specialist physician analysing patient data.
Goal: Produce an accurate ranked list of candidate diagnoses.
Tools: DiagnosisTool
"""

from crewai import Agent

from src.tools.prediction_tool import DiagnosisTool
from src.tools.rag_tool import MedicalRAGTool


def create_diagnostician(llm) -> Agent:
    return Agent(
        role="Clinical Diagnostician",
        goal=(
            "Analyse the patient's vitals and symptoms using the DiagnosisTool "
            "to get ML predictions, then use the MedicalRAGTool to retrieve "
            "supporting evidence from the medical knowledge base. "
            "Produce a clear, evidence-grounded differential diagnosis."
        ),
        backstory=(
            "You are an experienced internal medicine physician with 20 years "
            "of clinical practice. You are methodical, evidence-based, and always "
            "frame your findings in terms a fellow clinician can act upon. "
            "You cross-reference ML predictions with established medical literature "
            "and never speculate beyond what the data supports."
        ),
        tools=[DiagnosisTool(), MedicalRAGTool()],
        llm=llm,
        verbose=True,
        max_iter=5,
    )
