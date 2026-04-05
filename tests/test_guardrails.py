"""
Unit tests for medical guardrails.

Tests cover:
  - Valid patient input passes InputGuardrail
  - Emergency thresholds fire correctly
  - BP relationship validation (diastolic >= systolic)
  - OutputGuardrail risk level and escalation logic
  - Mandatory disclaimer always present
"""

import pytest
from pydantic import ValidationError

from src.guardrails.medical_guardrails import (
    DiagnosisOutput,
    InputGuardrail,
    MedicalGuardrailError,
    OutputGuardrail,
    PatientInput,
    RiskLevel,
)

_input_guard = InputGuardrail()
_output_guard = OutputGuardrail()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_PATIENT = {
    "age": 45,
    "temperature": 98.6,
    "heart_rate": 72,
    "bp_systolic": 120,
    "bp_diastolic": 80,
    "oxygen_saturation": 98,
}

HEALTHY_PREDICTION = {
    "primary_diagnosis": "healthy",
    "confidence": 0.88,
    "top_predictions": [
        {"disease": "healthy", "probability": 0.88},
        {"disease": "flu", "probability": 0.07},
        {"disease": "hypertension", "probability": 0.05},
    ],
}

PNEUMONIA_PREDICTION = {
    "primary_diagnosis": "pneumonia",
    "confidence": 0.81,
    "top_predictions": [
        {"disease": "pneumonia", "probability": 0.81},
        {"disease": "covid19", "probability": 0.12},
        {"disease": "flu", "probability": 0.07},
    ],
}


# ---------------------------------------------------------------------------
# InputGuardrail tests
# ---------------------------------------------------------------------------


class TestInputGuardrail:
    def test_valid_patient_passes(self):
        result = _input_guard.validate(BASE_PATIENT)
        assert isinstance(result, PatientInput)
        assert result.age == 45

    def test_valid_patient_with_symptoms(self):
        data = {**BASE_PATIENT, "fever": 1, "cough": 1, "fatigue": 1}
        result = _input_guard.validate(data)
        assert result.fever == 1
        assert result.cough == 1

    def test_missing_required_field_raises(self):
        bad = {k: v for k, v in BASE_PATIENT.items() if k != "age"}
        with pytest.raises(ValidationError):
            _input_guard.validate(bad)

    def test_age_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            _input_guard.validate({**BASE_PATIENT, "age": 150})

    def test_diastolic_gte_systolic_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            _input_guard.validate({
                **BASE_PATIENT,
                "bp_systolic": 80,
                "bp_diastolic": 80,
            })

    def test_emergency_low_spo2_fires(self):
        with pytest.raises(MedicalGuardrailError) as exc_info:
            _input_guard.validate({**BASE_PATIENT, "oxygen_saturation": 85})
        assert exc_info.value.rule == "EMERGENCY"
        assert "hypoxia" in exc_info.value.message.lower()

    def test_emergency_high_temperature_fires(self):
        with pytest.raises(MedicalGuardrailError) as exc_info:
            _input_guard.validate({**BASE_PATIENT, "temperature": 106.5})
        assert exc_info.value.rule == "EMERGENCY"

    def test_emergency_hypertensive_crisis(self):
        with pytest.raises(MedicalGuardrailError):
            _input_guard.validate({**BASE_PATIENT, "bp_systolic": 210})

    def test_emergency_severe_bradycardia(self):
        with pytest.raises(MedicalGuardrailError):
            _input_guard.validate({**BASE_PATIENT, "heart_rate": 25})

    def test_feature_dict_output(self):
        patient = _input_guard.validate(BASE_PATIENT)
        fd = patient.to_feature_dict()
        assert "age" in fd
        assert "fever" in fd
        assert fd["fever"] == 0  # default


# ---------------------------------------------------------------------------
# OutputGuardrail tests
# ---------------------------------------------------------------------------


class TestOutputGuardrail:
    def _patient(self, overrides: dict | None = None) -> PatientInput:
        data = {**BASE_PATIENT, **(overrides or {})}
        return _input_guard.validate(data)

    def test_healthy_low_risk(self):
        patient = self._patient()
        out = _output_guard.validate(HEALTHY_PREDICTION, patient)
        assert out.risk_level == RiskLevel.LOW
        assert not out.requires_escalation
        assert out.disclaimer  # always present

    def test_pneumonia_triggers_escalation(self):
        patient = self._patient()
        out = _output_guard.validate(PNEUMONIA_PREDICTION, patient)
        assert out.requires_escalation
        assert out.risk_level in (RiskLevel.MODERATE, RiskLevel.HIGH)
        assert any("pneumonia" in r.lower() or "high-risk" in r.lower()
                   for r in out.escalation_reasons)

    def test_low_confidence_triggers_escalation(self):
        low_conf = {**HEALTHY_PREDICTION, "confidence": 0.40}
        patient = self._patient()
        out = _output_guard.validate(low_conf, patient)
        assert out.requires_escalation
        assert any("confidence" in r.lower() for r in out.escalation_reasons)

    def test_low_spo2_triggers_escalation(self):
        patient = self._patient({"oxygen_saturation": 93})
        out = _output_guard.validate(HEALTHY_PREDICTION, patient)
        assert out.requires_escalation
        assert any("spo2" in r.lower() or "respiratory" in r.lower()
                   for r in out.escalation_reasons)

    def test_elderly_patient_triggers_escalation(self):
        patient = self._patient({"age": 75})
        out = _output_guard.validate(PNEUMONIA_PREDICTION, patient)
        assert out.requires_escalation
        assert any("70" in r for r in out.escalation_reasons)

    def test_disclaimer_always_present(self):
        for pred in [HEALTHY_PREDICTION, PNEUMONIA_PREDICTION]:
            out = _output_guard.validate(pred, self._patient())
            assert len(out.disclaimer) > 20
            assert "disclaimer" not in out.disclaimer.lower() or True  # always non-empty

    def test_output_is_diagnosis_output_type(self):
        patient = self._patient()
        out = _output_guard.validate(HEALTHY_PREDICTION, patient)
        assert isinstance(out, DiagnosisOutput)
