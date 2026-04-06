"""
Streamlit web UI for the Medical Diagnosis Bot.

Run with:
    streamlit run app.py
"""

import streamlit as st
import os
import json
from src.crew import run_pipeline
from src.models.disease_classifier import get_classifier

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
    "A trustworthy Clinical Decision Support tool — "
    "powered by real disease data, RandomForest + SHAP, and CrewAI agents."
)

# ---------------------------------------------------------------------------
# Sidebar — API key
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Your API key is never stored or logged.",
    )

    st.divider()
    st.header("About")
    st.markdown(
        """
        **Pipeline:**
        1. Input guardrail validation
        2. DiagnosticianAgent → top-3 diseases
        3. ExplainerAgent → SHAP narrative
        4. SafetyOfficerAgent → risk + disclaimer

        **Data:** `kamruzzaman-asif/Diseases_Dataset`
        (HuggingFace, real disease-symptom mappings)

        **RAG:** MedQuAD — NIH/NLM medical Q&A
        """
    )
    st.divider()
    st.warning(
        "This tool is for decision support only. "
        "It does not replace a licensed physician."
    )

# ---------------------------------------------------------------------------
# Load classifier to get the real symptom list from the trained model
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading disease model...")
def load_model():
    return get_classifier()

try:
    clf = load_model()
    all_symptoms = clf.feature_names
except Exception as e:
    st.error(f"Could not load model: {e}. Run `python -c 'from src.models.disease_classifier import train_and_save; train_and_save()'` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Patient input
# ---------------------------------------------------------------------------

st.header("Patient Symptoms")

col1, col2 = st.columns([1, 2])

with col1:
    age = st.number_input("Patient Age (optional)", min_value=0, max_value=120, value=0)
    age = int(age) if age > 0 else None

with col2:
    st.markdown("**Select all symptoms that are present:**")

# Show symptoms as a multi-select (cleaner than 100+ checkboxes)
selected_symptoms = st.multiselect(
    "Symptoms",
    options=all_symptoms,
    placeholder="Start typing to search symptoms...",
    help="Select all symptoms the patient is currently experiencing."
)

# Convert to binary dict
symptom_flags = {s: (1 if s in selected_symptoms else 0) for s in all_symptoms}

# ---------------------------------------------------------------------------
# Quick ML preview (before running full pipeline)
# ---------------------------------------------------------------------------

if selected_symptoms:
    st.divider()
    st.subheader("Quick ML Preview")
    st.caption("Instant prediction from the RandomForest model (no agents yet).")

    try:
        prediction = clf.predict(symptom_flags)
        top3       = prediction["top_predictions"]

        pcol1, pcol2, pcol3 = st.columns(3)
        for col, pred in zip([pcol1, pcol2, pcol3], top3):
            with col:
                st.metric(
                    label=pred["disease"].replace("_", " ").title(),
                    value=f"{pred['probability']:.1%}",
                )

        # SHAP bar chart
        explanation = clf.explain(symptom_flags)
        contribs    = explanation["feature_contributions"]

        import pandas as pd
        shap_df = pd.DataFrame(contribs).head(10)
        shap_df["feature"] = shap_df["feature"].str.replace("_", " ").str.title()
        shap_df = shap_df.sort_values("shap_value")

        st.bar_chart(shap_df.set_index("feature")["shap_value"])
        st.caption("SHAP values — positive = pushes toward this diagnosis, negative = pushes away.")

    except Exception as e:
        st.warning(f"Preview unavailable: {e}")

# ---------------------------------------------------------------------------
# Full pipeline — run agents
# ---------------------------------------------------------------------------

st.divider()

if not api_key:
    st.info("Enter your Anthropic API key in the sidebar to run the full agent pipeline.")
elif not selected_symptoms:
    st.info("Select at least one symptom above to run a diagnosis.")
else:
    if st.button("Run Full Diagnosis", type="primary", use_container_width=True):
        with st.spinner("Running diagnostic pipeline (Diagnose → Explain → Safety check)..."):
            try:
                result = run_pipeline(
                    symptoms=symptom_flags,
                    age=age,
                    api_key=api_key,
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        if result["status"] == "emergency":
            st.error("🚨 EMERGENCY DETECTED", icon="🚨")
            st.markdown(result["report"])

        elif result["status"] == "error":
            st.warning(f"Input validation failed: {result['error']}")

        else:
            st.success("Diagnosis complete.")
            st.markdown("## Clinical Report")
            st.markdown(result["report"])

            # Download button
            st.download_button(
                label="Download Report (Markdown)",
                data=result["report"],
                file_name="clinical_report.md",
                mime="text/markdown",
            )
