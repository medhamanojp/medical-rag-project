# Medical Diagnosis Bot

Towards a trustworthy Clinical Decision Support framework with Explainability and Safety Guardrails.

## Architecture

A three-agent **CrewAI** pipeline backed by **Claude** (Anthropic), with a **Random Forest** classifier and **SHAP** explanations.

```
Patient Data
     │
     ▼
┌─────────────────────┐
│  InputGuardrail     │  ← Pydantic validation + emergency threshold checks
│  (Pydantic v2)      │    Fires immediately on life-threatening vitals
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ DiagnosticianAgent  │  ← Tool: DiagnosisTool
│ (CrewAI + Claude)   │    RandomForest → top-3 disease predictions
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  ExplainerAgent     │  ← Tool: SHAPExplainerTool
│ (CrewAI + Claude)   │    SHAP TreeExplainer → clinical narrative
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ SafetyOfficerAgent  │  ← Tool: SafetyCheckTool
│ (CrewAI + Claude)   │    OutputGuardrail → risk level + escalation
└─────────┬───────────┘
          │
          ▼
  Clinical Report (Markdown)
```

### Agents

| Agent | Role | Tool |
|---|---|---|
| DiagnosticianAgent | Runs ML classifier, produces differential diagnosis | DiagnosisTool |
| ExplainerAgent | Generates SHAP narrative, explains clinical reasoning | SHAPExplainerTool |
| SafetyOfficerAgent | Applies output guardrails, flags escalations | SafetyCheckTool |

### Guardrails

**Input guardrails** (before the model sees data):
- Pydantic field-level validation (ranges, types)
- BP relationship check (diastolic < systolic)
- Emergency thresholds: SpO2 < 90%, temp > 105°F, heart rate < 30 or > 180 bpm, systolic BP > 200 or < 70 mmHg — these immediately halt the pipeline and escalate

**Output guardrails** (after prediction):
- Low confidence (< 50%) → escalation required
- High-risk diseases (pneumonia, covid19) → escalation flag
- SpO2 < 94% → respiratory alert
- Age ≥ 70 → elevated-risk flag
- Mandatory clinical disclaimer always appended

### ML Model

- **Algorithm**: RandomForestClassifier (200 trees, depth 12)
- **Training data**: Synthetic balanced dataset, 400 patients × 6 classes
- **Classes**: healthy, flu, pneumonia, covid19, diabetes, hypertension
- **Features**: 6 vitals + 15 binary symptom flags
- **Explainability**: SHAP TreeExplainer (exact, not approximate)

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd medical-rag-project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 5. Train the classifier (first run only — also auto-trains on first use)
python main.py --train
```

## Usage

```bash
# Run a built-in demo patient (pneumonia, diabetes, covid19, hypertension_crisis, emergency)
python main.py --demo pneumonia

# Run with a custom patient JSON file
python main.py --patient data/sample_patient.json

# Retrain the ML model
python main.py --train
```

### Patient JSON format

```json
{
  "age": 58,
  "temperature": 101.4,
  "heart_rate": 95,
  "bp_systolic": 135,
  "bp_diastolic": 86,
  "oxygen_saturation": 95,
  "fever": 1,
  "cough": 1,
  "shortness_of_breath": 1,
  "fatigue": 1,
  "loss_of_taste_smell": 1,
  "body_ache": 1
}
```

All symptom fields default to `0` if omitted. All vitals are required.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## Disclaimer

This system is a clinical decision support tool only. It does **not** replace the judgement of a licensed physician. All recommendations must be reviewed by a qualified clinician before any treatment decision is made.

## Project Structure

```
medical-rag-project/
├── main.py                          # Entry point
├── requirements.txt
├── .env.example
├── data/
│   └── sample_patient.json
├── src/
│   ├── crew.py                      # CrewAI crew assembly
│   ├── agents/
│   │   ├── diagnosis_agent.py       # DiagnosticianAgent
│   │   ├── explainability_agent.py  # ExplainerAgent
│   │   └── safety_agent.py          # SafetyOfficerAgent
│   ├── tools/
│   │   ├── prediction_tool.py       # DiagnosisTool (CrewAI)
│   │   ├── shap_tool.py             # SHAPExplainerTool (CrewAI)
│   │   └── safety_tool.py           # SafetyCheckTool (CrewAI)
│   ├── models/
│   │   └── disease_classifier.py   # RandomForest + SHAP
│   └── guardrails/
│       └── medical_guardrails.py   # Input + Output guardrails
└── tests/
    ├── test_guardrails.py
    └── test_classifier.py
```
