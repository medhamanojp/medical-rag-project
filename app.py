"""
Streamlit UI for the Medical Diagnosis Bot.

Run with:
    streamlit run app.py
"""

import json
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Medical Diagnosis Bot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 Medical Diagnosis Bot")
st.caption(
    "Clinical Decision Support · Powered by CrewAI + Claude · "
    "Explainability via SHAP · RAG from MedQuAD"
)
st.warning(
    "**Disclaimer:** This is a clinical decision *support* tool only. "
    "It does not replace a licensed physician. All outputs must be reviewed "
    "by a qualified clinician before any treatment decision.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# Sidebar — API key + mode
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Your Claude API key. Not stored anywhere.",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    st.subheader("Demo Patients")
    demo_cases = {
        "None (enter manually)": None,
        "Pneumonia (65yo, low SpO2)": {
            "age": 65, "temperature": 103.2, "heart_rate": 108,
            "bp_systolic": 118, "bp_diastolic": 76, "oxygen_saturation": 91,
            "fever": 1, "cough": 1, "shortness_of_breath": 1,
            "chest_pain": 1, "fatigue": 1, "sweating": 1,
        },
        "COVID-19 (40yo)": {
            "age": 40, "temperature": 100.9, "heart_rate": 92,
            "bp_systolic": 122, "bp_diastolic": 80, "oxygen_saturation": 94,
            "fever": 1, "cough": 1, "loss_of_taste_smell": 1,
            "fatigue": 1, "body_ache": 1,
        },
        "Diabetes (52yo)": {
            "age": 52, "temperature": 98.7, "heart_rate": 78,
            "bp_systolic": 138, "bp_diastolic": 88, "oxygen_saturation": 97,
            "frequent_urination": 1, "blurred_vision": 1,
            "fatigue": 1, "nausea": 1,
        },
        "Hypertension (58yo)": {
            "age": 58, "temperature": 98.6, "heart_rate": 88,
            "bp_systolic": 195, "bp_diastolic": 118, "oxygen_saturation": 96,
            "headache": 1, "dizziness": 1, "chest_pain": 1, "blurred_vision": 1,
        },
        "EMERGENCY — Critical SpO2 (72yo)": {
            "age": 72, "temperature": 104.5, "heart_rate": 130,
            "bp_systolic": 88, "bp_diastolic": 55, "oxygen_saturation": 82,
            "fever": 1, "cough": 1, "shortness_of_breath": 1,
            "chest_pain": 1, "fatigue": 1,
        },
    }
    selected_demo = st.selectbox("Load a demo patient", options=list(demo_cases.keys()))
    demo_data = demo_cases[selected_demo]

    st.divider()
    use_rag = st.toggle(
        "Enable RAG (MedQuAD)",
        value=True,
        help="Augments the report with evidence retrieved from the MedQuAD "
             "medical Q&A dataset. Requires initial index build (~30s).",
    )

# ---------------------------------------------------------------------------
# Patient input form
# ---------------------------------------------------------------------------

st.subheader("Patient Data Entry")

SYMPTOM_COLS = [
    "fever", "cough", "shortness_of_breath", "fatigue", "chest_pain",
    "headache", "body_ache", "loss_of_taste_smell", "sore_throat",
    "runny_nose", "nausea", "dizziness", "frequent_urination",
    "blurred_vision", "sweating",
]

col_vitals, col_symptoms = st.columns([1, 1])

with col_vitals:
    st.markdown("**Vitals**")
    age = st.number_input(
        "Age (years)", min_value=0, max_value=120, value=int(demo_data["age"]) if demo_data else 45
    )
    temperature = st.number_input(
        "Temperature (°F)", min_value=90.0, max_value=110.0,
        value=float(demo_data["temperature"]) if demo_data else 98.6, step=0.1, format="%.1f"
    )
    heart_rate = st.number_input(
        "Heart Rate (bpm)", min_value=20, max_value=300,
        value=int(demo_data["heart_rate"]) if demo_data else 72
    )
    bp_systolic = st.number_input(
        "BP Systolic (mmHg)", min_value=50, max_value=260,
        value=int(demo_data["bp_systolic"]) if demo_data else 120
    )
    bp_diastolic = st.number_input(
        "BP Diastolic (mmHg)", min_value=30, max_value=160,
        value=int(demo_data["bp_diastolic"]) if demo_data else 80
    )
    oxygen_saturation = st.number_input(
        "SpO2 (%)", min_value=50, max_value=100,
        value=int(demo_data["oxygen_saturation"]) if demo_data else 98
    )

with col_symptoms:
    st.markdown("**Symptoms** (check all that apply)")
    symptom_values: dict[str, int] = {}
    for symptom in SYMPTOM_COLS:
        label = symptom.replace("_", " ").title()
        default = bool(demo_data.get(symptom, 0)) if demo_data else False
        symptom_values[symptom] = int(st.checkbox(label, value=default))

# ---------------------------------------------------------------------------
# Build patient dict
# ---------------------------------------------------------------------------

patient_data = {
    "age": age,
    "temperature": temperature,
    "heart_rate": heart_rate,
    "bp_systolic": bp_systolic,
    "bp_diastolic": bp_diastolic,
    "oxygen_saturation": oxygen_saturation,
    **symptom_values,
}

with st.expander("Raw patient JSON"):
    st.json(patient_data)

# ---------------------------------------------------------------------------
# Run diagnosis
# ---------------------------------------------------------------------------

st.divider()
run_button = st.button("Run Diagnosis", type="primary", use_container_width=True)

if run_button:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    # Step 1: Input guardrail (fast, before spawning the crew)
    from src.guardrails.medical_guardrails import InputGuardrail, MedicalGuardrailError
    input_guard = InputGuardrail()

    try:
        input_guard.validate(patient_data)
    except MedicalGuardrailError as e:
        st.error(f"🚨 EMERGENCY GUARDRAIL FIRED — {e.message}")
        st.error(
            "**Action required:** Escalate to emergency care immediately. "
            "Do NOT rely on automated diagnosis in this situation."
        )
        st.stop()
    except Exception as e:
        st.error(f"Input validation error: {e}")
        st.stop()

    # Step 2: Quick ML preview (no LLM cost — instant feedback)
    with st.spinner("Running ML classifier..."):
        from src.models.disease_classifier import get_classifier
        clf = get_classifier()
        prediction = clf.predict(patient_data)
        explanation = clf.explain(patient_data)

    # Show quick ML results immediately
    st.subheader("Model Predictions (pre-LLM)")
    pred_cols = st.columns(3)
    for i, pred in enumerate(prediction["top_predictions"]):
        with pred_cols[i]:
            pct = pred["probability"] * 100
            color = "🟢" if i == 0 else ("🟡" if i == 1 else "🔴")
            st.metric(
                label=f"{color} #{i+1}: {pred['disease'].title()}",
                value=f"{pct:.1f}%",
            )

    with st.expander("SHAP Feature Contributions"):
        st.caption(
            f"Top features driving the **{explanation['primary_diagnosis'].title()}** prediction. "
            "Positive SHAP → pushes toward diagnosis. Negative → pushes away."
        )
        for fc in explanation["feature_contributions"]:
            bar_color = "🔵" if fc["shap_value"] > 0 else "🔸"
            st.write(
                f"{bar_color} **{fc['feature'].replace('_', ' ').title()}** "
                f"— SHAP: `{fc['shap_value']:+.4f}` "
                f"(patient value: `{fc['patient_value']}`)"
            )

    # Step 3: RAG retrieval (optional)
    rag_context = ""
    if use_rag:
        with st.spinner("Retrieving evidence from MedQuAD knowledge base..."):
            try:
                from src.rag.rag_pipeline import get_rag_pipeline
                rag = get_rag_pipeline()
                query = (
                    f"{prediction['primary_diagnosis']} symptoms treatment "
                    f"age {patient_data['age']}"
                )
                docs = rag.retrieve(query, k=3)
                rag_context = "\n\n".join(
                    f"[Source {i+1}] Q: {d['question']}\nA: {d['answer']}"
                    for i, d in enumerate(docs)
                )
                with st.expander("RAG Evidence Retrieved (MedQuAD)"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Source {i}:** {d['question']}")
                        st.caption(d["answer"][:400] + ("..." if len(d["answer"]) > 400 else ""))
                        st.divider()
            except Exception as e:
                st.warning(f"RAG retrieval skipped: {e}")

    # Step 4: Full CrewAI report
    with st.spinner("Running CrewAI agents (Diagnostician → Explainer → Safety Officer)..."):
        try:
            from src.crew import MedicalDiagnosisCrew
            crew = MedicalDiagnosisCrew(rag_context=rag_context)
            report = crew.run(patient_data)
        except Exception as e:
            st.error(f"Crew failed: {e}")
            st.stop()

    st.divider()
    st.subheader("Clinical Decision Support Report")
    st.markdown(report)

    # Download button
    st.download_button(
        label="Download Report",
        data=report,
        file_name="clinical_report.md",
        mime="text/markdown",
    )
