"""
Disease classifier using DDXPlus — a large-scale medical diagnosis dataset
from a published NeurIPS 2022 paper by Fansi Tchango et al. (Mila / McGill).

Paper: "DDXPlus: A New English Clinical Cases Dataset For Automatic
        Medical Diagnosis" — https://arxiv.org/abs/2205.09487

HuggingFace: mila-iqia/ddxplus

Dataset structure:
  PATHOLOGY   — disease name (49 conditions)
  EVIDENCES   — string repr of a list of evidence codes, e.g.
                  "['E_1', 'E_2', 'E_10_@_V_3']"
                Binary evidence:     'E_1'         → symptom present
                Categorical evidence: 'E_3_@_V_45' → evidence E_3 = value V_45
  AGE         — patient age (integer)
  SEX         — 'M' or 'F'

Metadata files (loaded from HuggingFace hub):
  release_evidences.json  — maps evidence codes to human-readable names
  release_conditions.json — maps condition codes to human-readable names

The feature matrix is built by:
  1. Parsing EVIDENCES strings into lists
  2. Treating each unique evidence token as a binary column (1=present, 0=absent)
  3. Adding AGE and SEX as numeric features
  4. Discovering all features dynamically — nothing is hardcoded

Training uses a 50k random sample of the 1.3M row train split for speed.
Increase SAMPLE_SIZE for higher accuracy at the cost of training time.
"""

import ast
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

MODEL_PATH    = Path(__file__).parent / "disease_classifier.pkl"
ENCODER_PATH  = Path(__file__).parent / "label_encoder.pkl"
FEATURES_PATH = Path(__file__).parent / "feature_names.pkl"

SAMPLE_SIZE = 50_000   # rows to sample from the 1.3M dataset


# ---------------------------------------------------------------------------
# DDXPlus loader
# ---------------------------------------------------------------------------

def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    """
    Load DDXPlus from HuggingFace, parse evidences into a binary feature
    matrix, and return (DataFrame with features + 'disease' column, feature_names).

    Evidence tokens become binary columns:
      - Binary evidence 'E_1'       → column 'E_1'          = 1 if present
      - Categorical 'E_3_@_V_45'   → column 'E_3_@_V_45'   = 1 if present
      - AGE and SEX_M added as numeric features

    Evidence codes are mapped to readable names where possible using
    release_evidences.json from the dataset repo.
    """
    from datasets import load_dataset as hf_load
    from huggingface_hub import hf_hub_download
    import json

    print("Loading DDXPlus dataset (mila-iqia/ddxplus)...")
    print(f"  Sampling {SAMPLE_SIZE:,} rows from train split...")

    ds = hf_load("mila-iqia/ddxplus", split="train")
    df = ds.to_pandas()

    # Sample for tractable training time
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    print(f"  {len(df):,} rows loaded.")
    print(f"  Disease classes: {df['PATHOLOGY'].nunique()}")

    # ------------------------------------------------------------------
    # Load human-readable evidence names from metadata
    # ------------------------------------------------------------------
    try:
        ev_path = hf_hub_download(
            repo_id="mila-iqia/ddxplus",
            filename="release_evidences.json",
            repo_type="dataset",
        )
        with open(ev_path) as f:
            evidence_meta = json.load(f)
        # Build code → readable name mapping
        code_to_name = {
            code: meta.get("name", code).lower().replace(" ", "_")
            for code, meta in evidence_meta.items()
        }
        print(f"  Evidence metadata loaded ({len(code_to_name)} codes).")
    except Exception as e:
        print(f"  Could not load evidence metadata ({e}). Using raw codes.")
        code_to_name = {}

    # ------------------------------------------------------------------
    # Parse EVIDENCES column
    # ------------------------------------------------------------------
    def parse_evidences(val) -> list[str]:
        """Parse evidence string/list into a list of token strings."""
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return []
        return []

    print("  Parsing evidence lists...")
    df["ev_list"] = df["EVIDENCES"].apply(parse_evidences)

    # ------------------------------------------------------------------
    # Build vocabulary of all unique evidence tokens
    # ------------------------------------------------------------------
    all_tokens = sorted({
        tok
        for evs in df["ev_list"]
        for tok in evs
    })
    print(f"  Unique evidence tokens: {len(all_tokens)}")

    # Map tokens to readable names where available
    def token_to_feature(tok: str) -> str:
        base = tok.split("_@_")[0]   # base code before categorical suffix
        if base in code_to_name:
            name = code_to_name[base]
            suffix = tok[len(base):]  # e.g. '_@_V_3'
            return (name + suffix).lower().replace(" ", "_")
        return tok.lower()

    token_to_col = {tok: token_to_feature(tok) for tok in all_tokens}
    feature_names = list(token_to_col.values())

    # ------------------------------------------------------------------
    # Build binary feature matrix
    # ------------------------------------------------------------------
    print("  Building binary feature matrix...")
    rows = []
    for ev_list in df["ev_list"]:
        ev_set = set(ev_list)
        rows.append([1 if tok in ev_set else 0 for tok in all_tokens])

    feat_df = pd.DataFrame(rows, columns=feature_names)

    # Add age and sex
    feat_df["age"]   = pd.to_numeric(df["AGE"],  errors="coerce").fillna(40).astype(int)
    feat_df["sex_m"] = (df["SEX"].str.upper() == "M").astype(int)
    feature_names    = feature_names + ["age", "sex_m"]

    feat_df["disease"] = df["PATHOLOGY"].str.strip()

    print(f"  Feature matrix: {feat_df.shape[0]:,} rows × {len(feature_names)} features")
    print(f"  Disease classes ({feat_df['disease'].nunique()}):")
    print(feat_df["disease"].value_counts().to_string())

    return feat_df, feature_names


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save() -> tuple[RandomForestClassifier, LabelEncoder, list[str]]:
    """
    Train the RandomForest on DDXPlus data and save model artifacts to disk.

    Saved files:
      disease_classifier.pkl  — trained RandomForestClassifier
      label_encoder.pkl       — LabelEncoder (disease name ↔ integer)
      feature_names.pkl       — ordered list of feature column names
    """
    df, feature_names = load_dataset()

    le = LabelEncoder()
    y  = le.fit_transform(df["disease"])
    X  = df[feature_names]

    print(f"\nTraining RandomForest on {len(X):,} samples, "
          f"{len(feature_names)} features, {len(le.classes_)} classes...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("\n=== Evaluation ===")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(clf,           MODEL_PATH)
    joblib.dump(le,            ENCODER_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    return clf, le, feature_names


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class DiseaseClassifier:
    """
    Loads the trained model from disk and provides predict() and explain().
    Auto-trains on first use if .pkl files are not found.
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            print("Model not found — training now (this takes ~2 minutes)...")
            train_and_save()

        self.model         = joblib.load(MODEL_PATH)
        self.le            = joblib.load(ENCODER_PATH)
        self.feature_names = joblib.load(FEATURES_PATH)
        self.explainer     = shap.TreeExplainer(self.model)

    def predict(self, symptom_flags: dict) -> dict:
        """
        Returns top-3 disease predictions with probabilities.

        Args:
            symptom_flags: dict of feature_name → 0 or 1.
                           Unknown features are ignored; missing ones default to 0.
        """
        X     = self._to_dataframe(symptom_flags)
        probs = self.model.predict_proba(X)[0]

        ranked = sorted(
            zip(self.le.classes_, probs), key=lambda x: x[1], reverse=True
        )
        top3 = [
            {"disease": d, "probability": round(float(p), 4)}
            for d, p in ranked[:3]
        ]
        return {
            "top_predictions":   top3,
            "primary_diagnosis": top3[0]["disease"],
            "confidence":        top3[0]["probability"],
        }

    def explain(self, symptom_flags: dict) -> dict:
        """
        Compute SHAP values for the primary prediction.
        Returns top-10 features by absolute SHAP value.
        """
        X               = self._to_dataframe(symptom_flags)
        shap_values     = self.explainer.shap_values(X)
        probs           = self.model.predict_proba(X)[0]
        primary_idx     = int(np.argmax(probs))
        primary_disease = self.le.classes_[primary_idx]

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[0, :, primary_idx]
        else:
            sv = shap_values[primary_idx][0]

        base_val = float(
            self.explainer.expected_value[primary_idx]
            if hasattr(self.explainer.expected_value, "__len__")
            else self.explainer.expected_value
        )

        contributions = [
            {
                "feature":    feat,
                "shap_value": round(float(v), 5),
                "present":    int(symptom_flags.get(feat, 0)),
            }
            for feat, v in zip(self.feature_names, sv)
        ]
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "primary_diagnosis":     primary_disease,
            "feature_contributions": contributions[:10],
            "base_value":            round(base_val, 5),
        }

    def _to_dataframe(self, symptom_flags: dict) -> pd.DataFrame:
        row = {f: symptom_flags.get(f, 0) for f in self.feature_names}
        return pd.DataFrame([row], columns=self.feature_names)


# Singleton — loaded once per process
_instance: DiseaseClassifier | None = None


def get_classifier() -> DiseaseClassifier:
    global _instance
    if _instance is None:
        _instance = DiseaseClassifier()
    return _instance


if __name__ == "__main__":
    train_and_save()
