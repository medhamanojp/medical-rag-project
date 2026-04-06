"""
DiagnosticianAgent — acts as a clinical physician.
Runs the ML classifier and produces a differential diagnosis.
"""

from crewai import Agent
from src.tools.prediction_tool import DiagnosisTool


def build_diagnosis_agent(llm) -> Agent:
    return Agent(
        role="Clinical Diagnostician",
        goal=(
            "Analyse the patient's reported symptoms and produce a clear "
            "differential diagnosis with the top-3 most likely diseases "
            "and their probabilities."
        ),
        backstory=(
            "You are an experienced general practitioner with 20 years of "
            "clinical experience. You use evidence-based tools to assess "
            "patient symptoms methodically and produce a ranked differential "
            "diagnosis. You are thorough, precise, and always prioritise "
            "patient safety over speed."
        ),
        tools=[DiagnosisTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
