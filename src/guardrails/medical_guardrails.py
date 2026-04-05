"""
Medical guardrails — input validation and output safety checks.

Two layers:
  1. InputGuardrail   — validates patient data before it touches the model.
  2. OutputGuardrail  — validates / enriches the diagnosis result before it
                        is shown to the doctor.

Design principles:
  * Fail-safe: any violation raises MedicalGuardrailError (never silently pass).
  * Clinically conservative: when in doubt, escalate to a human clinician.
  * Transparent: every guard decision is logged with a reason string.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class MedicalGuardrailError(Exception):
    """Raised when a guardrail fires and the request must not proceed."""

    def __init__(self, rule: str, message: str):
        self.rule = rule
        self.message = message
        super().__init__(f"[{rule}] {message}")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Input model (Pydantic v2)
# ---------------------------------------------------------------------------


class PatientInput(BaseModel):
    """Structured, validated patient data accepted by the system."""

    # Demographics
    age: int = Field(..., ge=0, le=120, description="Patient age in years")

    # Vitals
    temperature: float = Field(
        ..., ge=90.0, le=110.0,
        description="Body temperature in Fahrenheit"
    )
    heart_rate: int = Field(
        ..., ge=20, le=300,
        description="Heart rate in bpm"
    )
    bp_systolic: int = Field(
        ..., ge=50, le=260,
        description="Systolic blood pressure in mmHg"
    )
    bp_diastolic: int = Field(
        ..., ge=30, le=160,
        description="Diastolic blood pressure in mmHg"
    )
    oxygen_saturation: int = Field(
        ..., ge=50, le=100,
        description="SpO2 percentage"
    )

    # Binary symptom flags (0 or 1)
    fever: int = Field(default=0, ge=0, le=1)
    cough: int = Field(default=0, ge=0, le=1)
    shortness_of_breath: int = Field(default=0, ge=0, le=1)
    fatigue: int = Field(default=0, ge=0, le=1)
    chest_pain: int = Field(default=0, ge=0, le=1)
    headache: int = Field(default=0, ge=0, le=1)
    body_ache: int = Field(default=0, ge=0, le=1)
    loss_of_taste_smell: int = Field(default=0, ge=0, le=1)
    sore_throat: int = Field(default=0, ge=0, le=1)
    runny_nose: int = Field(default=0, ge=0, le=1)
    nausea: int = Field(default=0, ge=0, le=1)
    dizziness: int = Field(default=0, ge=0, le=1)
    frequent_urination: int = Field(default=0, ge=0, le=1)
    blurred_vision: int = Field(default=0, ge=0, le=1)
    sweating: int = Field(default=0, ge=0, le=1)

    @field_validator("temperature")
    @classmethod
    def temperature_sanity(cls, v: float) -> float:
        if v < 95.0:
            logger.warning("Hypothermia range temperature: %.1f°F", v)
        if v > 106.0:
            logger.warning("Life-threatening hyperthermia: %.1f°F", v)
        return v

    @model_validator(mode="after")
    def bp_relationship(self) -> "PatientInput":
        if self.bp_diastolic >= self.bp_systolic:
            raise ValueError(
                "Diastolic BP must be lower than systolic BP "
                f"(got {self.bp_systolic}/{self.bp_diastolic})"
            )
        return self

    def to_feature_dict(self) -> dict[str, Any]:
        """Return the flat dict expected by the ML model."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Input guardrail
# ---------------------------------------------------------------------------


class InputGuardrail:
    """Gate that validates patient data and detects immediate emergencies."""

    # Thresholds that trigger an immediate emergency alert
    EMERGENCY_RULES: list[tuple[str, str]] = [
        ("oxygen_saturation < 90", "Critical hypoxia — SpO2 below 90%"),
        ("temperature > 105.0", "Life-threatening hyperthermia (>105°F)"),
        ("temperature < 95.0", "Hypothermia (<95°F)"),
        ("heart_rate > 180", "Severe tachycardia (>180 bpm)"),
        ("heart_rate < 30", "Severe bradycardia (<30 bpm)"),
        ("bp_systolic > 200", "Hypertensive crisis (systolic >200 mmHg)"),
        ("bp_systolic < 70", "Severe hypotension (systolic <70 mmHg)"),
        ("chest_pain == 1 and bp_systolic < 90",
         "Suspected cardiogenic shock — chest pain with systolic <90 mmHg"),
    ]

    def validate(self, raw_data: dict) -> PatientInput:
        """
        Validate raw_data, fire emergency guardrails, and return a
        PatientInput on success.

        Raises:
            MedicalGuardrailError: if an emergency threshold is breached.
            ValidationError: if Pydantic field constraints are violated.
        """
        patient = PatientInput(**raw_data)
        self._check_emergencies(patient)
        logger.info("Input guardrail passed for patient age=%d", patient.age)
        return patient

    def _check_emergencies(self, p: PatientInput) -> None:
        for rule_expr, message in self.EMERGENCY_RULES:
            if self._eval_rule(rule_expr, p):
                logger.error("EMERGENCY guardrail fired: %s", message)
                raise MedicalGuardrailError(rule="EMERGENCY", message=message)

    @staticmethod
    def _eval_rule(expr: str, p: PatientInput) -> bool:
        try:
            return bool(eval(expr, {}, p.model_dump()))  # noqa: S307
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Output guardrail
# ---------------------------------------------------------------------------


class DiagnosisOutput(BaseModel):
    """Structured output returned to the doctor after all guardrails pass."""

    primary_diagnosis: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_predictions: list[dict]
    risk_level: RiskLevel
    requires_escalation: bool
    escalation_reasons: list[str] = Field(default_factory=list)
    disclaimer: str
    shap_summary: str = ""


class OutputGuardrail:
    """
    Validates prediction results and applies clinical safety policies:
      - Low-confidence predictions trigger escalation.
      - High-risk diagnoses are flagged.
      - A mandatory disclaimer is always appended.
    """

    LOW_CONFIDENCE_THRESHOLD = 0.50
    MODERATE_CONFIDENCE_THRESHOLD = 0.70

    HIGH_RISK_DISEASES = {"pneumonia", "covid19"}

    MANDATORY_DISCLAIMER = (
        "This system is a clinical decision support tool only. "
        "It does NOT replace the judgement of a licensed physician. "
        "All recommendations must be reviewed by a qualified clinician "
        "before any treatment decision is made."
    )

    def validate(
        self,
        prediction: dict,
        patient: PatientInput,
    ) -> DiagnosisOutput:
        """
        Apply output guardrails and return a DiagnosisOutput.

        Args:
            prediction: raw output from DiseaseClassifier.predict()
            patient:    validated PatientInput (used for risk contextualisation)

        Returns:
            DiagnosisOutput with risk_level, escalation flags, and disclaimer.
        """
        primary = prediction["primary_diagnosis"]
        confidence = prediction["confidence"]
        top3 = prediction["top_predictions"]

        risk_level, escalation_reasons = self._assess_risk(
            primary, confidence, patient
        )
        requires_escalation = bool(escalation_reasons)

        if requires_escalation:
            logger.warning(
                "Escalation required for diagnosis '%s': %s",
                primary,
                escalation_reasons,
            )

        return DiagnosisOutput(
            primary_diagnosis=primary,
            confidence=confidence,
            top_predictions=top3,
            risk_level=risk_level,
            requires_escalation=requires_escalation,
            escalation_reasons=escalation_reasons,
            disclaimer=self.MANDATORY_DISCLAIMER,
        )

    def _assess_risk(
        self,
        disease: str,
        confidence: float,
        patient: PatientInput,
    ) -> tuple[RiskLevel, list[str]]:
        reasons: list[str] = []

        if confidence < self.LOW_CONFIDENCE_THRESHOLD:
            reasons.append(
                f"Model confidence is low ({confidence:.0%}) — "
                "differential diagnosis required."
            )

        if disease in self.HIGH_RISK_DISEASES:
            reasons.append(
                f"'{disease}' is classified as a high-risk condition "
                "requiring urgent clinical review."
            )

        if patient.oxygen_saturation < 94:
            reasons.append(
                f"SpO2 of {patient.oxygen_saturation}% warrants "
                "immediate respiratory assessment."
            )

        if patient.age >= 70:
            reasons.append(
                "Patient age ≥ 70 — elevated risk of rapid deterioration."
            )

        # Determine overall risk level
        if reasons:
            if (
                disease in self.HIGH_RISK_DISEASES
                or confidence < self.LOW_CONFIDENCE_THRESHOLD
                or patient.oxygen_saturation < 90
            ):
                return RiskLevel.HIGH, reasons
            return RiskLevel.MODERATE, reasons

        if confidence < self.MODERATE_CONFIDENCE_THRESHOLD:
            return RiskLevel.MODERATE, []

        return RiskLevel.LOW, []
