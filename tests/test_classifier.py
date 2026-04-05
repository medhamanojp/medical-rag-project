"""
Unit tests for DiseaseClassifier (no LLM required).

Tests cover:
  - predict() returns expected structure
  - explain() returns SHAP values for top-10 features
  - Model is deterministic (same input → same output)
  - Each synthetic patient class is predicted correctly (smoke test)
"""

import pytest

from src.models.disease_classifier import (
    DISEASE_CLASSES,
    DiseaseClassifier,
    _make_patient,
    get_classifier,
)

# Shared classifier instance (trained once for the whole test session)
@pytest.fixture(scope="session")
def clf() -> DiseaseClassifier:
    return get_classifier()


BASE_PNEUMONIA = {
    "age": 65,
    "temperature": 103.2,
    "heart_rate": 108,
    "bp_systolic": 118,
    "bp_diastolic": 76,
    "oxygen_saturation": 91,
    "fever": 1,
    "cough": 1,
    "shortness_of_breath": 1,
    "chest_pain": 1,
    "fatigue": 1,
    "sweating": 1,
    "headache": 0,
    "body_ache": 0,
    "loss_of_taste_smell": 0,
    "sore_throat": 0,
    "runny_nose": 0,
    "nausea": 0,
    "dizziness": 0,
    "frequent_urination": 0,
    "blurred_vision": 0,
}


class TestPredictOutput:
    def test_returns_required_keys(self, clf):
        result = clf.predict(BASE_PNEUMONIA)
        assert "top_predictions" in result
        assert "primary_diagnosis" in result
        assert "confidence" in result

    def test_top3_length(self, clf):
        result = clf.predict(BASE_PNEUMONIA)
        assert len(result["top_predictions"]) == 3

    def test_probabilities_sum_to_one(self, clf):
        result = clf.predict(BASE_PNEUMONIA)
        # top-3 should be <= 1.0 (they're a subset of all probs)
        total = sum(p["probability"] for p in result["top_predictions"])
        assert total <= 1.001

    def test_primary_diagnosis_is_valid_class(self, clf):
        result = clf.predict(BASE_PNEUMONIA)
        assert result["primary_diagnosis"] in DISEASE_CLASSES

    def test_confidence_between_0_and_1(self, clf):
        result = clf.predict(BASE_PNEUMONIA)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_deterministic(self, clf):
        r1 = clf.predict(BASE_PNEUMONIA)
        r2 = clf.predict(BASE_PNEUMONIA)
        assert r1["primary_diagnosis"] == r2["primary_diagnosis"]
        assert r1["confidence"] == r2["confidence"]


class TestExplainOutput:
    def test_returns_required_keys(self, clf):
        result = clf.explain(BASE_PNEUMONIA)
        assert "primary_diagnosis" in result
        assert "feature_contributions" in result
        assert "base_value" in result

    def test_top10_contributions(self, clf):
        result = clf.explain(BASE_PNEUMONIA)
        assert len(result["feature_contributions"]) <= 10

    def test_contribution_keys(self, clf):
        result = clf.explain(BASE_PNEUMONIA)
        for fc in result["feature_contributions"]:
            assert "feature" in fc
            assert "shap_value" in fc
            assert "patient_value" in fc

    def test_shap_values_are_floats(self, clf):
        result = clf.explain(BASE_PNEUMONIA)
        for fc in result["feature_contributions"]:
            assert isinstance(fc["shap_value"], float)


class TestSyntheticPatientsSmoke:
    """
    Loose smoke test: each synthetic patient class should have its true
    disease in the top-2 predictions.
    """

    @pytest.mark.parametrize("disease", DISEASE_CLASSES)
    def test_true_class_in_top2(self, clf, disease):
        patient = _make_patient(disease)
        result = clf.predict(patient)
        top2_diseases = [p["disease"] for p in result["top_predictions"][:2]]
        assert disease in top2_diseases, (
            f"Expected '{disease}' in top-2 but got {top2_diseases} "
            f"(patient={patient})"
        )
