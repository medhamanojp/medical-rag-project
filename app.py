"""
Streamlit web UI for the Medical Diagnosis Bot.

Reads ANTHROPIC_API_KEY from .env automatically.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from src.crew import run_pipeline
from src.models.disease_classifier import get_classifier

# Load .env file (ANTHROPIC_API_KEY)
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Medical Diagnosis Bot",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Medical Diagnosis Bot")
st.caption(
    "Clinical Decision Support — "
    "RandomForest + SHAP + CrewAI agents + MedQuAD RAG"
)

if not API_KEY:
    st.error(
        "ANTHROPIC_API_KEY not found. "
        "Create a .env file with ANTHROPIC_API_KEY=sk-ant-... and restart the app."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — info only, no key input
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Pipeline")
    st.markdown(
        """
        **1. Input Guardrail**
        Validates symptoms. Halts on emergencies.

        **2. DiagnosticianAgent**
        RandomForest → top-3 diseases.
        MedQuAD RAG for evidence.

        **3. ExplainerAgent**
        SHAP values → plain language explanation.

        **4. SafetyOfficerAgent**
        Risk level + escalation decision + disclaimer.
        """
    )
    st.divider()
    st.header("Data Sources")
    st.markdown(
        """
        - **Diseases**: `kamruzzaman-asif/Diseases_Dataset`
        - **RAG**: MedQuAD (NIH/NLM ~16k Q&A)
        """
    )
    st.divider()
    st.warning(
        "For decision support only. Does not replace a licensed physician."
    )

# ---------------------------------------------------------------------------
# Load classifier (once, cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading disease model...")
def load_model():
    return get_classifier()

try:
    clf = load_model()
    all_symptoms = clf.feature_names
except Exception as e:
    st.error(
        f"Model not found. Train it first:\n\n"
        f"```\npython -c \"from src.models.disease_classifier import train_and_save; train_and_save()\"\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Patient input
# ---------------------------------------------------------------------------

st.header("Patient Symptoms")

age = st.number_input("Patient Age (optional)", min_value=0, max_value=120, value=0)
age = int(age) if age > 0 else None

selected_symptoms = st.multiselect(
    "Select all symptoms that are present",
    options=all_symptoms,
    placeholder="Start typing to search symptoms...",
)

symptom_flags = {s: (1 if s in selected_symptoms else 0) for s in all_symptoms}

# ---------------------------------------------------------------------------
# Quick ML preview
# ---------------------------------------------------------------------------

if selected_symptoms:
    st.divider()
    st.subheader("Quick ML Preview")
    st.caption("Instant prediction from RandomForest — no agents, no API cost.")

    try:
        prediction = clf.predict(symptom_flags)
        top3 = prediction["top_predictions"]

        c1, c2, c3 = st.columns(3)
        for col, pred in zip([c1, c2, c3], top3):
            with col:
                st.metric(
                    label=pred["disease"].replace("_", " ").title(),
                    value=f"{pred['probability']:.1%}",
                )

        explanation = clf.explain(symptom_flags)
        contribs    = explanation["feature_contributions"]

        import pandas as pd
        shap_df = pd.DataFrame(contribs).head(10)
        shap_df["feature"] = shap_df["feature"].str.replace("_", " ").str.title()
        shap_df = shap_df.sort_values("shap_value")

        st.bar_chart(shap_df.set_index("feature")["shap_value"])
        st.caption(
            "SHAP values: positive = pushed toward this diagnosis, "
            "negative = pushed away."
        )

    except Exception as e:
        st.warning(f"Preview error: {e}")

# ---------------------------------------------------------------------------
# Full agent pipeline
# ---------------------------------------------------------------------------

st.divider()

if not selected_symptoms:
    st.info("Select at least one symptom above to run a diagnosis.")
else:
    if st.button("Run Full Diagnosis", type="primary", use_container_width=True):
        with st.spinner("Running agents: Diagnose → Explain → Safety check..."):
            try:
                result = run_pipeline(
                    symptoms=symptom_flags,
                    age=age,
                    api_key=API_KEY,
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        if result["status"] == "emergency":
            st.error("EMERGENCY DETECTED", icon="🚨")
            st.markdown(result["report"])

        elif result["status"] == "error":
            st.warning(f"Validation failed: {result['error']}")

        else:
            st.success("Diagnosis complete.")
            st.markdown("## Clinical Report")
            st.markdown(result["report"])
            st.download_button(
                label="Download Report (Markdown)",
                data=result["report"],
                file_name="clinical_report.md",
                mime="text/markdown",
            )
