# Medical Diagnosis Bot
## Trustworthy Clinical Decision Support
### with Real Data · Explainability · Safety Guardrails

---

## Slide 1 — The Problem

AI in healthcare has one core challenge:

> **How do you make a system that is accurate AND explainable AND safe?**

- A model that just predicts is a black box — clinicians won't trust it
- An LLM that just talks has no calibrated probabilities — it can hallucinate
- A system with no safety checks can confidently say something dangerous

**Solution**: Combine all three — ML for prediction, SHAP for explanation, LLM agents for clinical reasoning, guardrails for safety.

---

## Slide 2 — System Overview

```
Patient Symptoms
      │
      ▼
┌─────────────────────┐
│   Input Guardrail   │  Validates input. Halts on emergency combinations.
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ DiagnosticianAgent  │  RandomForest → top-3 diseases + probabilities
│  (CrewAI + Claude)  │  MedQuAD RAG → supporting medical evidence
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ExplainerAgent    │  SHAP values → plain language clinical narrative
│  (CrewAI + Claude)  │  "Headache was the strongest indicator because..."
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ SafetyOfficerAgent  │  Risk level · Escalation flag · Disclaimer
│  (CrewAI + Claude)  │  Conservative by design — escalates when uncertain
└──────────┬──────────┘
           │
           ▼
    Clinical Report (Markdown)
```

---

## Slide 3 — Why Three Separate Agents?

Each agent has a **single focused job**. This separation matters:

| Agent | Role | What Claude is told to be |
|-------|------|--------------------------|
| DiagnosticianAgent | Diagnose | Experienced GP, 20 years practice |
| ExplainerAgent | Explain | Clinical AI informatics specialist |
| SafetyOfficerAgent | Safety check | Conservative patient safety officer |

**Each agent only sees what it needs:**
- Diagnostician sees symptoms → produces diagnosis
- Explainer sees diagnosis + symptoms → produces narrative
- Safety Officer sees diagnosis + narrative → produces safety report

This prevents any single prompt from doing too much, which leads to worse outputs.

---

## Slide 4 — The Real Dataset: DDXPlus

**`mila-iqia/ddxplus`** — published at NeurIPS 2022

> Fansi Tchango, Rishab Goel, Zhi Wen, Julien Martel, Joumana Ghosn
> *"DDXPlus: A New English Clinical Cases Dataset For Automatic Medical Diagnosis"*
> NeurIPS 2022 — Mila / McGill University

| Property | Value |
|----------|-------|
| Patient cases | 1.3 million |
| Disease classes | 49 |
| Evidence codes | 223 symptoms/antecedents |
| Evidence types | Binary, categorical, multi-choice |
| License | Public research use |

```python
# Evidence column looks like this per patient:
"['E_1', 'E_10', 'E_45_@_V_3', 'E_67']"
#   ↑ binary       ↑ categorical (evidence 45, value 3)

# At training time, every unique token becomes a binary feature:
all_tokens → ["E_1", "E_10", "E_45_@_V_3", ...] → 223+ columns
# Mapped to readable names via release_evidences.json:
# E_1 → "chest_pain", E_10 → "cough", etc.
```

**Why this matters for your presentation**: This is not a random Kaggle upload. It is a peer-reviewed, published, large-scale clinical dataset from a top ML research institution.

---

## Slide 5 — Why RandomForest (not just LLM)?

An LLM cannot reliably say:

> "74.2% probability of Migraine vs 18.1% Tension Headache"

It would guess. RandomForest gives **calibrated probabilities** from real data.

| | RandomForest | LLM alone |
|--|-------------|-----------|
| Disease probabilities | Real, calibrated | Made up |
| SHAP explainability | Yes (exact) | No |
| Grounded in dataset | Yes | No |
| Deterministic | Yes | No |
| Hallucination risk | None (it's maths) | Present |

**Division of labour:**
- RandomForest answers **what** (with numbers)
- SHAP answers **why** (with feature weights)
- LLM agents answer **what does this mean clinically** (with language)

---

## Slide 6 — SHAP Explainability

SHAP = SHapley Additive exPlanations

For every prediction, SHAP tells us exactly how much each symptom contributed:

```
Primary diagnosis: Migraine (74.2%)

Top contributing symptoms:
  headache          +0.31   ← strongly pushed toward Migraine
  nausea            +0.18   ← supported the diagnosis
  sensitivity_light +0.14   ← supported the diagnosis
  fever             -0.09   ← pushed slightly away (more flu-like)
  cough             -0.12   ← pushed away (not migraine-typical)
```

The ExplainerAgent takes these numbers and writes:
> *"The primary diagnosis of Migraine is strongly supported by the combination of headache and nausea. The absence of respiratory symptoms such as cough makes respiratory illness less likely..."*

---

## Slide 7 — Guardrails

### Input Guardrail (before anything runs)

```python
# Pydantic v2 model validates:
# - All values are 0 or 1
# - At least one symptom is active
# - Emergency combinations trigger immediate halt:
if chest_pain AND shortness_of_breath → EMERGENCY
if loss_of_consciousness             → EMERGENCY
if difficulty_breathing AND fever    → EMERGENCY
if severe_headache AND stiff_neck    → EMERGENCY (meningitis)
```

### Output Guardrail (after prediction)

```python
if confidence < 40%          → escalate (model is uncertain)
if disease in HIGH_RISK_LIST → escalate (pneumonia, sepsis, meningitis...)
if age >= 70                 → elevated risk flag
always                       → clinical disclaimer appended
```

---

## Slide 8 — RAG Pipeline

Agents don't just use Claude's built-in knowledge — they look things up.

**Knowledge base**: MedQuAD (~16,000 medical Q&A pairs from NIH/NLM)

```
Agent asks: "what causes migraines"
      │
      ▼
SentenceTransformer encodes the query → vector
      │
      ▼
FAISS searches .rag_cache/faiss.index → finds top-3 most similar Q&A pairs
      │
      ▼
Returns real NIH answers to the agent as context
      │
      ▼
Agent uses this to ground its response in real medical literature
```

**Why this matters**: Reduces hallucination. Agent says things backed by NIH, not invented.

**Performance**: Index is built once (~1 min), then cached. Every subsequent query takes milliseconds.

---

## Slide 9 — Model Lifecycle

A common question: **does the model retrain every time?**

**No. Only once.**

```
First run:
  train_and_save()
    ├── Download dataset from HuggingFace    (~30 sec)
    ├── Parse symptoms into binary matrix    (~5 sec)
    ├── Train RandomForest (200 trees)       (~10 sec)
    ├── Save disease_classifier.pkl ─────── to disk
    ├── Save label_encoder.pkl      ─────── to disk
    └── Save feature_names.pkl      ─────── to disk

Every run after:
  DiseaseClassifier.__init__()
    ├── .pkl exists? → load from disk       (~0.2 sec)
    └── Model ready

In Streamlit:
  @st.cache_resource → model loaded once per server session
  100 users refresh the page → model loaded exactly once
```

---

## Slide 10 — Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Dataset | DDXPlus `mila-iqia/ddxplus` (NeurIPS 2022) | 1.3M patient cases, 49 diseases |
| ML Model | scikit-learn RandomForestClassifier | Calibrated probability predictions |
| Explainability | SHAP TreeExplainer | Exact feature attribution |
| Agent Framework | CrewAI | Sequential multi-agent orchestration |
| LLM | Claude Sonnet 4.6 (Anthropic) | Powers all 3 agents |
| RAG Embeddings | sentence-transformers (MiniLM-L6-v2) | Semantic search |
| Vector Store | FAISS | Fast similarity search over MedQuAD |
| Knowledge Base | MedQuAD (NIH/NLM) | 16k real medical Q&A pairs |
| Data Validation | Pydantic v2 | Input/output guardrails |
| Web UI | Streamlit | Browser interface |
| Config | python-dotenv | API key from .env file |

---

## Slide 11 — How to Run

```bash
# 1. Clone and install
git clone https://github.com/medhamanojp/medical-rag-project.git
cd medical-rag-project
pip install -r requirements.txt

# 2. Add API key
cp .env.example .env
# Edit .env → ANTHROPIC_API_KEY=sk-ant-...

# 3. Train the model (once only)
python -c "from src.models.disease_classifier import train_and_save; train_and_save()"

# 4. Run the app
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Slide 12 — What You See in the App

1. **Symptom selector** — searchable dropdown populated from the real dataset
2. **Age input** — optional, used by safety guardrail for elderly risk flagging
3. **Quick ML Preview** — instant top-3 prediction + SHAP bar chart (free, no API call)
4. **Run Full Diagnosis** — triggers all 3 CrewAI agents (~20-40 seconds)
5. **Clinical Report** — full markdown: diagnosis · explanation · risk · disclaimer
6. **Download** — save report as `.md` file

---

## Slide 13 — Design Principles

1. **Real data over synthetic** — model trained on actual disease-symptom mappings
2. **Fail safe** — guardrails halt on emergencies; output guardrail escalates when uncertain
3. **Explainability first** — every prediction has a SHAP-backed reason
4. **No black boxes** — RandomForest is interpretable; SHAP makes it transparent
5. **Human in the loop** — the system recommends, a clinician decides
6. **Separation of concerns** — diagnosis, explanation, safety in distinct agents
7. **Grounded in evidence** — RAG pulls real NIH content, not LLM imagination

---

## Slide 14 — Disclaimer

> This system is a **clinical decision support tool only**.
> It does **not** replace the judgement of a licensed physician.
> All outputs must be reviewed by a qualified clinician before any treatment decision is made.

---

*Built with CrewAI · Anthropic Claude · scikit-learn · SHAP · FAISS · Streamlit*
