"""
Disease classifier trained on synthetic clinical data.

Features: vitals (age, temperature, heart rate, blood pressure, SpO2)
          + 15 binary symptom flags
Target:   6 disease classes (pneumonia, diabetes, hypertension, flu,
          covid19, healthy)

SHAP TreeExplainer is used to generate per-prediction feature attributions
that the ExplainabilityAgent surfaces to the doctor.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISEASE_CLASSES = [
    "healthy",
    "flu",
    "pneumonia",
    "covid19",
    "diabetes",
    "hypertension",
]

SYMPTOM_FEATURES = [
    "fever",
    "cough",
    "shortness_of_breath",
    "fatigue",
    "chest_pain",
    "headache",
    "body_ache",
    "loss_of_taste_smell",
    "sore_throat",
    "runny_nose",
    "nausea",
    "dizziness",
    "frequent_urination",
    "blurred_vision",
    "sweating",
]

VITAL_FEATURES = [
    "age",
    "temperature",       # Fahrenheit
    "heart_rate",        # bpm
    "bp_systolic",       # mmHg
    "bp_diastolic",      # mmHg
    "oxygen_saturation", # SpO2 %
]

ALL_FEATURES = VITAL_FEATURES + SYMPTOM_FEATURES

MODEL_PATH = Path(__file__).parent / "disease_classifier.pkl"
ENCODER_PATH = Path(__file__).parent / "label_encoder.pkl"

# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _make_patient(disease: str) -> dict:
    """Generate one synthetic patient record for the given disease."""
    base: dict = {f: 0 for f in SYMPTOM_FEATURES}

    if disease == "healthy":
        return {
            "age": int(_RNG.integers(18, 80)),
            "temperature": round(float(_RNG.normal(98.4, 0.4)), 1),
            "heart_rate": int(_RNG.integers(60, 80)),
            "bp_systolic": int(_RNG.integers(110, 130)),
            "bp_diastolic": int(_RNG.integers(70, 85)),
            "oxygen_saturation": int(_RNG.integers(97, 100)),
            **base,
        }

    if disease == "flu":
        base.update(
            fever=1, cough=int(_RNG.integers(0, 2)),
            fatigue=1, headache=1, body_ache=1,
            runny_nose=int(_RNG.integers(0, 2)),
            sore_throat=int(_RNG.integers(0, 2)),
            sweating=int(_RNG.integers(0, 2)),
        )
        return {
            "age": int(_RNG.integers(18, 80)),
            "temperature": round(float(_RNG.normal(101.5, 0.8)), 1),
            "heart_rate": int(_RNG.integers(80, 110)),
            "bp_systolic": int(_RNG.integers(110, 130)),
            "bp_diastolic": int(_RNG.integers(70, 85)),
            "oxygen_saturation": int(_RNG.integers(95, 99)),
            **base,
        }

    if disease == "pneumonia":
        base.update(
            fever=1, cough=1, shortness_of_breath=1,
            chest_pain=int(_RNG.integers(0, 2)),
            fatigue=1, sweating=int(_RNG.integers(0, 2)),
        )
        return {
            "age": int(_RNG.integers(30, 85)),
            "temperature": round(float(_RNG.normal(102.5, 0.9)), 1),
            "heart_rate": int(_RNG.integers(90, 120)),
            "bp_systolic": int(_RNG.integers(100, 130)),
            "bp_diastolic": int(_RNG.integers(65, 85)),
            "oxygen_saturation": int(_RNG.integers(88, 94)),
            **base,
        }

    if disease == "covid19":
        base.update(
            fever=1, cough=1,
            shortness_of_breath=int(_RNG.integers(0, 2)),
            loss_of_taste_smell=1, fatigue=1,
            body_ache=int(_RNG.integers(0, 2)),
            headache=int(_RNG.integers(0, 2)),
        )
        return {
            "age": int(_RNG.integers(18, 85)),
            "temperature": round(float(_RNG.normal(100.8, 1.0)), 1),
            "heart_rate": int(_RNG.integers(75, 110)),
            "bp_systolic": int(_RNG.integers(105, 135)),
            "bp_diastolic": int(_RNG.integers(65, 88)),
            "oxygen_saturation": int(_RNG.integers(90, 97)),
            **base,
        }

    if disease == "diabetes":
        base.update(
            frequent_urination=1, blurred_vision=1, fatigue=1,
            nausea=int(_RNG.integers(0, 2)),
            dizziness=int(_RNG.integers(0, 2)),
        )
        return {
            "age": int(_RNG.integers(35, 80)),
            "temperature": round(float(_RNG.normal(98.6, 0.4)), 1),
            "heart_rate": int(_RNG.integers(65, 90)),
            "bp_systolic": int(_RNG.integers(120, 145)),
            "bp_diastolic": int(_RNG.integers(78, 95)),
            "oxygen_saturation": int(_RNG.integers(95, 99)),
            **base,
        }

    if disease == "hypertension":
        base.update(
            headache=1, dizziness=1,
            chest_pain=int(_RNG.integers(0, 2)),
            blurred_vision=int(_RNG.integers(0, 2)),
            nausea=int(_RNG.integers(0, 2)),
        )
        return {
            "age": int(_RNG.integers(40, 85)),
            "temperature": round(float(_RNG.normal(98.6, 0.4)), 1),
            "heart_rate": int(_RNG.integers(70, 100)),
            "bp_systolic": int(_RNG.integers(145, 180)),
            "bp_diastolic": int(_RNG.integers(92, 115)),
            "oxygen_saturation": int(_RNG.integers(95, 99)),
            **base,
        }

    raise ValueError(f"Unknown disease: {disease}")


def generate_dataset(n_per_class: int = 300) -> pd.DataFrame:
    """Generate a balanced synthetic clinical dataset."""
    records = []
    for disease in DISEASE_CLASSES:
        for _ in range(n_per_class):
            row = _make_patient(disease)
            row["disease"] = disease
            records.append(row)
    df = pd.DataFrame(records)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save() -> tuple[RandomForestClassifier, LabelEncoder]:
    """Train the classifier and persist it to disk."""
    df = generate_dataset(n_per_class=400)
    le = LabelEncoder()
    y = le.fit_transform(df["disease"])
    X = df[ALL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("=== Classifier Evaluation ===")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return clf, le


# ---------------------------------------------------------------------------
# Inference interface
# ---------------------------------------------------------------------------

class DiseaseClassifier:
    """Wrapper around the trained RF model + SHAP explainer."""

    def __init__(self):
        if not MODEL_PATH.exists():
            print("Model not found — training now...")
            train_and_save()

        self.model: RandomForestClassifier = joblib.load(MODEL_PATH)
        self.le: LabelEncoder = joblib.load(ENCODER_PATH)
        # TreeExplainer is fast and exact for tree-based models
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, patient_data: dict) -> dict:
        """
        Returns top-3 disease predictions with probabilities.

        Args:
            patient_data: dict with keys matching ALL_FEATURES

        Returns:
            {
              "top_predictions": [{"disease": str, "probability": float}, ...],
              "primary_diagnosis": str,
              "confidence": float,
            }
        """
        X = self._to_dataframe(patient_data)
        probs = self.model.predict_proba(X)[0]
        classes = self.le.classes_

        ranked = sorted(
            zip(classes, probs), key=lambda x: x[1], reverse=True
        )
        top3 = [
            {"disease": d, "probability": round(float(p), 4)}
            for d, p in ranked[:3]
        ]
        return {
            "top_predictions": top3,
            "primary_diagnosis": top3[0]["disease"],
            "confidence": top3[0]["probability"],
        }

    def explain(self, patient_data: dict) -> dict:
        """
        Compute SHAP values for the primary prediction.

        Returns:
            {
              "primary_diagnosis": str,
              "feature_contributions": [
                  {"feature": str, "shap_value": float, "patient_value": any},
                  ...  (sorted by |shap_value| desc, top 10)
              ],
              "base_value": float,
            }
        """
        X = self._to_dataframe(patient_data)
        # shap_values shape varies by SHAP version:
        #   old (<0.46): list of (n_samples, n_features), one per class
        #   new (>=0.46): ndarray of (n_samples, n_features, n_classes)
        shap_values = self.explainer.shap_values(X)

        probs = self.model.predict_proba(X)[0]
        primary_class_idx = int(np.argmax(probs))
        primary_disease = self.le.classes_[primary_class_idx]

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # New format: (n_samples, n_features, n_classes)
            sv_for_class = shap_values[0, :, primary_class_idx]
        else:
            # Old format: list of (n_samples, n_features)
            sv_for_class = shap_values[primary_class_idx][0]

        base_val = float(
            self.explainer.expected_value[primary_class_idx]
            if hasattr(self.explainer.expected_value, "__len__")
            else self.explainer.expected_value
        )

        contributions = [
            {
                "feature": feat,
                "shap_value": round(float(sv), 5),
                "patient_value": patient_data.get(feat, 0),
            }
            for feat, sv in zip(ALL_FEATURES, sv_for_class)
        ]
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "primary_diagnosis": primary_disease,
            "feature_contributions": contributions[:10],
            "base_value": round(base_val, 5),
        }

    def _to_dataframe(self, patient_data: dict) -> pd.DataFrame:
        row = {f: patient_data.get(f, 0) for f in ALL_FEATURES}
        return pd.DataFrame([row], columns=ALL_FEATURES)


# Singleton — loaded once, reused across all tool calls
_classifier_instance: DiseaseClassifier | None = None


def get_classifier() -> DiseaseClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = DiseaseClassifier()
    return _classifier_instance


if __name__ == "__main__":
    train_and_save()
