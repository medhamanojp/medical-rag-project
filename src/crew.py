"""
Crew assembly — wires the three agents into a sequential pipeline.

Flow:
  1. DiagnosticianAgent  →  runs ML classifier, produces differential diagnosis
  2. ExplainerAgent      →  explains SHAP attributions in plain language
  3. SafetyOfficerAgent  →  checks safety, sets risk level, appends disclaimer

Each task feeds its output into the context of the next task.
"""

from crewai import Crew, Task, Process
from langchain_anthropic import ChatAnthropic

from src.agents.diagnosis_agent import build_diagnosis_agent
from src.agents.explainability_agent import build_explainer_agent
from src.agents.safety_agent import build_safety_agent
from src.guardrails.medical_guardrails import validate_input


def build_llm(api_key: str) -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=api_key,
        temperature=0.2,   # low temperature = more consistent clinical reasoning
    )


def run_pipeline(symptoms: dict[str, int], age: int | None, api_key: str) -> dict:
    """
    Run the full diagnostic pipeline for a patient.

    Args:
        symptoms:  dict of symptom_name -> 0 or 1
        age:       patient age (optional)
        api_key:   Anthropic API key

    Returns:
        {
          "status":  "ok" | "emergency" | "error",
          "report":  str   (full markdown clinical report),
          "error":   str | None,
        }
    """

    # ------------------------------------------------------------------
    # 1. Input guardrail — check before anything else
    # ------------------------------------------------------------------
    guard = validate_input(symptoms, age=age)

    if guard["emergency"]:
        return {
            "status": "emergency",
            "report": (
                f"## EMERGENCY\n\n{guard['emergency_reason']}\n\n"
                "**Call emergency services immediately. Do not wait.**"
            ),
            "error": guard["emergency_reason"],
        }

    if not guard["valid"]:
        return {
            "status": "error",
            "report": None,
            "error":  guard["error"],
        }

    # ------------------------------------------------------------------
    # 2. Build agents and LLM
    # ------------------------------------------------------------------
    llm = build_llm(api_key)

    diagnosis_agent = build_diagnosis_agent(llm)
    explainer_agent = build_explainer_agent(llm)
    safety_agent    = build_safety_agent(llm)

    # Format symptoms for the task descriptions
    active_symptoms = [s for s, v in symptoms.items() if v == 1]
    symptom_str     = ", ".join(active_symptoms)
    age_str         = f"Age: {age}" if age else "Age: not provided"

    # ------------------------------------------------------------------
    # 3. Define tasks
    # ------------------------------------------------------------------
    task_diagnose = Task(
        description=(
            f"Patient presents with the following symptoms: {symptom_str}. "
            f"{age_str}. "
            "Use the DiagnosisTool with this exact symptoms dict: "
            f"{symptoms}. "
            "Produce a clear differential diagnosis listing the top-3 "
            "most likely diseases with their probabilities."
        ),
        expected_output=(
            "A differential diagnosis with top-3 diseases, their probabilities, "
            "and a brief clinical interpretation of the findings."
        ),
        agent=diagnosis_agent,
    )

    task_explain = Task(
        description=(
            "Using the diagnosis from the previous task, call the SHAPExplainerTool "
            f"with this symptoms dict: {symptoms}. "
            "Translate the SHAP feature attributions into a plain-language clinical "
            "narrative explaining which symptoms most strongly drove the primary "
            "diagnosis and which symptoms were less significant."
        ),
        expected_output=(
            "A clear clinical narrative explaining the top contributing symptoms "
            "to the primary diagnosis, written so both doctors and patients can "
            "understand it."
        ),
        agent=explainer_agent,
        context=[task_diagnose],
    )

    task_safety = Task(
        description=(
            "Review the diagnosis and explanation from the previous tasks. "
            "Call the SafetyCheckTool with the prediction from the diagnosis task "
            f"and age={age}. "
            "Write the final clinical report including: risk level, whether "
            "escalation to a human clinician is required, the reasons why, "
            "and the mandatory clinical disclaimer."
        ),
        expected_output=(
            "A final clinical safety report with: risk level, escalation decision "
            "with justification, any safety flags, and the clinical disclaimer."
        ),
        agent=safety_agent,
        context=[task_diagnose, task_explain],
    )

    # ------------------------------------------------------------------
    # 4. Assemble and run crew
    # ------------------------------------------------------------------
    crew = Crew(
        agents=[diagnosis_agent, explainer_agent, safety_agent],
        tasks=[task_diagnose, task_explain, task_safety],
        process=Process.sequential,   # each task waits for the previous one
        verbose=True,
    )

    result = crew.kickoff()

    return {
        "status": "ok",
        "report": str(result),
        "error":  None,
    }
