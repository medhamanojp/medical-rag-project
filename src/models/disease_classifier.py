"""
Disease classifier using real data from kamruzzaman-asif/Diseases_Dataset
(HuggingFace). Disease classes and symptom features are discovered directly
from the dataset — nothing is hardcoded.

Pipeline:
  1. Load dataset  →  Disease + Symptoms columns
  2. Parse symptoms into a binary feature matrix (one column per symptom)
  3. Train RandomForestClassifier
  4. Save model + label encoder + feature list to disk

SHAP TreeExplainer provides per-prediction feature attributions.
"""

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
FEATURES_PATH = Path(__file__).parent / "feature_names.pkl"   # saved after training


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """
    Load kamruzzaman-asif/Diseases_Dataset from HuggingFace and convert it
    into a binary feature matrix suitable for RandomForest training.

    The dataset has two columns:
      Disease  – disease name string
      Symptoms – symptoms as a comma-separated string or list

    Returns a DataFrame with:
      - one binary column per unique symptom (0 / 1)
      - a 'disease' label column
    """
    from datasets import load_dataset as hf_load

    print("Loading kamruzzaman-asif/Diseases_Dataset from HuggingFace...")
    raw = hf_load("kamruzzaman-asif/Diseases_Dataset", split="train")
    df  = raw.to_pandas()

    print(f"  Raw rows: {len(df)}")
    print(f"  Columns : {df.columns.tolist()}")

    # Normalise column names (dataset may use different casings)
    df.columns = [c.strip().lower() for c in df.columns]

    # Find the disease and symptom columns by name
    disease_col  = _find_col(df, ["disease", "label", "diagnosis"])
    symptom_col  = _find_col(df, ["symptoms", "symptom", "features"])

    df = df[[disease_col, symptom_col]].copy()
    df.columns = ["disease", "symptoms"]

    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)

    # Parse symptom strings into lists
    df["symptom_list"] = df["symptoms"].apply(_parse_symptoms)

    # Build vocabulary of all unique symptoms
    all_symptoms = sorted({
        sym
        for syms in df["symptom_list"]
        for sym in syms
    })
    print(f"  Unique symptoms found: {len(all_symptoms)}")
    print(f"  Disease classes      : {df['disease'].nunique()}")

    # One-hot encode symptoms
    for sym in all_symptoms:
        df[sym] = df["symptom_list"].apply(lambda lst: 1 if sym in lst else 0)

    df = df.drop(columns=["symptoms", "symptom_list"])
    return df, all_symptoms


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {df.columns.tolist()}")


def _parse_symptoms(value) -> list[str]:
    """
    Parse a symptom entry into a clean list of lowercase symptom strings.
    Handles comma-separated strings, pipe-separated strings, and lists.
    """
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        # Try comma first, then pipe
        sep = "," if "," in value else "|"
        items = value.split(sep)
    else:
        return []

    return [s.strip().lower().replace(" ", "_") for s in items if s.strip()]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save() -> tuple[RandomForestClassifier, LabelEncoder, list[str]]:
    """
    Train the RandomForest classifier on real disease-symptom data and
    save the model, label encoder, and feature list to disk.
    """
    df, feature_names = load_dataset()

    le = LabelEncoder()
    y  = le.fit_transform(df["disease"])
    X  = df[feature_names]

    print(f"\nTraining on {len(X)} samples, {len(feature_names)} features, "
          f"{len(le.classes_)} classes...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("\n=== Classifier Evaluation ===")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(clf,          MODEL_PATH)
    joblib.dump(le,           ENCODER_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    print(f"Model saved → {MODEL_PATH}")

    return clf, le, feature_names


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class DiseaseClassifier:
    """
    Wraps the trained RandomForest model with SHAP explainability.
    Loads feature names from disk so inference matches training exactly.
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            print("Model not found — training now...")
            train_and_save()

        self.model         = joblib.load(MODEL_PATH)
        self.le            = joblib.load(ENCODER_PATH)
        self.feature_names = joblib.load(FEATURES_PATH)
        self.explainer     = shap.TreeExplainer(self.model)

    def predict(self, symptom_flags: dict) -> dict:
        """
        Returns top-3 disease predictions with probabilities.

        Args:
            symptom_flags: dict mapping symptom names (strings) to 1 or 0.
                           Unknown symptoms are ignored; missing ones default to 0.

        Returns:
            {
              "top_predictions": [{"disease": str, "probability": float}, ...],
              "primary_diagnosis": str,
              "confidence": float,
            }
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

        Returns:
            {
              "primary_diagnosis": str,
              "feature_contributions": [
                  {"feature": str, "shap_value": float, "present": int},
                  ...  (top 10 by |shap_value|)
              ],
              "base_value": float,
            }
        """
        X                = self._to_dataframe(symptom_flags)
        shap_values      = self.explainer.shap_values(X)
        probs            = self.model.predict_proba(X)[0]
        primary_idx      = int(np.argmax(probs))
        primary_disease  = self.le.classes_[primary_idx]

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
            "primary_diagnosis":    primary_disease,
            "feature_contributions": contributions[:10],
            "base_value":           round(base_val, 5),
        }

    def _to_dataframe(self, symptom_flags: dict) -> pd.DataFrame:
        row = {f: symptom_flags.get(f, 0) for f in self.feature_names}
        return pd.DataFrame([row], columns=self.feature_names)


# Singleton
_instance: DiseaseClassifier | None = None


def get_classifier() -> DiseaseClassifier:
    global _instance
    if _instance is None:
        _instance = DiseaseClassifier()
    return _instance


if __name__ == "__main__":
    train_and_save()
