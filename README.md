# Medical Diagnosis Bot

A trustworthy **Clinical Decision Support (CDS)** framework that combines real disease data, machine learning, explainable AI, and multi-agent reasoning to assist clinicians — not replace them.

---

## What it does

A patient's symptoms go in. A structured clinical report comes out.

```
Patient Symptoms
      │
      ▼
┌─────────────────────┐
│   Input Guardrail   │  Validates input. Halts on emergency symptom combinations.
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ DiagnosticianAgent  │  Runs RandomForest classifier → top-3 disease predictions.
│  (CrewAI + Claude)  │  Looks up supporting evidence from MedQuAD (RAG).
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ExplainerAgent    │  Runs SHAP → explains which symptoms drove the diagnosis.
│  (CrewAI + Claude)  │  Translates SHAP values into plain clinical language.
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ SafetyOfficerAgent  │  Checks risk level. Flags escalation if needed.
│  (CrewAI + Claude)  │  Appends mandatory clinical disclaimer.
└──────────┬──────────┘
           │
           ▼
    Clinical Report (Markdown)
```

---

## Architecture

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| DiagnosticianAgent | Clinical Physician | DiagnosisTool, MedicalRAGTool |
| ExplainerAgent | AI Informatics Specialist | SHAPExplainerTool, MedicalRAGTool |
| SafetyOfficerAgent | Patient Safety Officer | SafetyCheckTool |

### ML Model

- **Algorithm**: RandomForestClassifier (200 trees, depth 12)
- **Training data**: `kamruzzaman-asif/Diseases_Dataset` (HuggingFace) — real disease-symptom mappings
- **Features**: Binary symptom flags — discovered dynamically from the dataset (no hardcoded list)
- **Explainability**: SHAP TreeExplainer (exact, not approximate)
- **Output**: Top-3 disease predictions with confidence scores

### Guardrails

**Input** (before inference):
- Pydantic v2 validation — all symptom values must be 0 or 1
- At least one symptom must be present
- Emergency combinations halt the pipeline immediately:
  - Chest pain + shortness of breath
  - Loss of consciousness
  - Difficulty breathing + high fever
  - Sudden severe headache + stiff neck (meningitis sign)

**Output** (after prediction):
- Low confidence (< 40%) → escalation required
- High-risk diseases (pneumonia, meningitis, sepsis, etc.) → escalation flag
- Patient age ≥ 70 → elevated risk flag
- Clinical disclaimer always appended

### RAG Pipeline

- **Knowledge base**: MedQuAD — ~16,000 medical Q&A pairs from NIH/NLM
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store**: FAISS (IndexFlatIP, cosine similarity)
- **Cache**: Built once, stored in `.rag_cache/`, reused on all subsequent runs

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/medhamanojp/medical-rag-project.git
cd medical-rag-project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 5. Train the model (downloads dataset on first run)
python -c "from src.models.disease_classifier import train_and_save; train_and_save()"
```

---

## Usage

### Web UI (Streamlit)

```bash
streamlit run app.py
```

- Select symptoms from a searchable dropdown
- Instant ML preview with SHAP bar chart
- Full agent pipeline with one click
- Download the clinical report as Markdown

### Python API

```python
from src.crew import run_pipeline

result = run_pipeline(
    symptoms={"headache": 1, "nausea": 1, "sensitivity_to_light": 1},
    age=34,
    api_key="sk-ant-...",
)

print(result["report"])
```

---

## Project Structure

```
medical-rag-project/
├── app.py                          # Streamlit web UI
├── requirements.txt
│
├── src/
│   ├── crew.py                     # Pipeline orchestration
│   ├── agents/
│   │   ├── diagnosis_agent.py      # DiagnosticianAgent
│   │   ├── explainability_agent.py # ExplainerAgent
│   │   └── safety_agent.py         # SafetyOfficerAgent
│   ├── tools/
│   │   ├── prediction_tool.py      # DiagnosisTool
│   │   ├── shap_tool.py            # SHAPExplainerTool
│   │   ├── safety_tool.py          # SafetyCheckTool
│   │   └── rag_tool.py             # MedicalRAGTool
│   ├── models/
│   │   └── disease_classifier.py  # RandomForest + SHAP
│   ├── guardrails/
│   │   └── medical_guardrails.py  # Input + output guardrails
│   └── rag/
│       └── rag_pipeline.py        # FAISS + MedQuAD indexing
│
└── data/                          # Sample patient files
```

---

## Dataset

**Disease-Symptom data**: [`kamruzzaman-asif/Diseases_Dataset`](https://huggingface.co/datasets/kamruzzaman-asif/Diseases_Dataset)
- Real disease-symptom mappings combining multiple public medical sources
- Disease classes and symptom features are loaded directly from the dataset — nothing hardcoded

**Medical knowledge base**: MedQuAD
- ~16,000 Q&A pairs sourced from NIH/NLM medical resources
- Used by agents to retrieve supporting evidence for their reasoning

---

## Disclaimer

This system is a **clinical decision support tool only**. It does **not** replace the judgement of a licensed physician. All outputs must be reviewed by a qualified clinician before any treatment decision is made.
