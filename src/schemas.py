"""
schemas.py
----------
Pydantic models for the FastAPI request and response bodies.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class PatientInput(BaseModel):
    """
    Raw clinical values entered by the user.
    Required fields are those available to virtually all Korean health
    check-up participants. Optional fields improve prediction accuracy
    when lab results are available.
    """

    # --- Required: basic biometrics ---
    age: int = Field(..., ge=20, le=90, description="Age in years (20–90)")
    sex_code: int = Field(..., description="Sex: 1 = male, 2 = female")
    height_cm: float = Field(..., ge=100, le=230, description="Height in centimetres")
    weight_kg: float = Field(..., ge=20, le=250, description="Weight in kilograms")
    waist_cm: float = Field(..., ge=30, le=250, description="Waist circumference in centimetres")

    # --- Required: blood pressure ---
    systolic_bp: int = Field(..., ge=60, le=300, description="Systolic blood pressure (mmHg)")
    diastolic_bp: int = Field(..., ge=30, le=200, description="Diastolic blood pressure (mmHg)")

    # --- Required: urine ---
    urine_protein: int = Field(
        ..., ge=1, le=6,
        description=(
            "Urine protein grade (1 = negative, 2 = trace, 3 = +1, "
            "4 = +2, 5 = +3, 6 = +4)"
        ),
    )

    # --- Required: lifestyle ---
    smoking_status: int = Field(
        ..., description="1 = never smoked, 2 = former smoker, 3 = current smoker"
    )
    alcohol_consumption: int = Field(..., description="0 = no, 1 = yes")

    # --- Optional: non-lipid lab values ---
    hemoglobin: Optional[float] = Field(None, ge=3, le=25, description="Hemoglobin (g/dL)")
    serum_creatinine: Optional[float] = Field(
        None, ge=0.1, le=30, description="Serum creatinine (mg/dL)"
    )
    ast: Optional[float] = Field(
        None, ge=1, le=5000, description="AST / serum GOT (U/L)"
    )
    alt: Optional[float] = Field(
        None, ge=1, le=5000, description="ALT / serum GPT (U/L)"
    )
    ggt: Optional[float] = Field(
        None, ge=1, le=5000, description="GGT / gamma-GTP (U/L)"
    )

    # --- Optional: lipid panel ---
    total_cholesterol: Optional[float] = Field(
        None, ge=50, le=1000, description="Total cholesterol (mg/dL)"
    )
    triglycerides: Optional[float] = Field(
        None, ge=10, le=5000, description="Triglycerides (mg/dL)"
    )
    hdl_cholesterol: Optional[float] = Field(
        None, ge=5, le=200, description="HDL cholesterol (mg/dL)"
    )
    ldl_cholesterol: Optional[float] = Field(
        None, ge=5, le=700, description="LDL cholesterol (mg/dL)"
    )

    @field_validator("sex_code")
    @classmethod
    def sex_must_be_valid(cls, v: int) -> int:
        if v not in {1, 2}:
            raise ValueError("sex_code must be 1 (male) or 2 (female)")
        return v

    @field_validator("smoking_status")
    @classmethod
    def smoking_must_be_valid(cls, v: int) -> int:
        if v not in {1, 2, 3}:
            raise ValueError("smoking_status must be 1, 2, or 3")
        return v

    @field_validator("alcohol_consumption")
    @classmethod
    def alcohol_must_be_valid(cls, v: int) -> int:
        if v not in {0, 1}:
            raise ValueError("alcohol_consumption must be 0 or 1")
        return v


class PredictionResponse(BaseModel):
    """Full prediction result returned by POST /predict."""
    probability: float = Field(description="Calibrated risk probability (0–1)")
    screen_positive: bool = Field(description="True if probability exceeds the screening threshold")
    risk_tier: str = Field(description="Risk tier: Low | Moderate | High | Very High")
    risk_tier_color: str = Field(description="Hex colour for the risk tier badge")
    recommendation: str = Field(description="Plain-language clinical recommendation")
    waterfall_chart: dict[str, Any] = Field(description="Plotly figure JSON for the SHAP waterfall")


class GlobalImportanceResponse(BaseModel):
    """Global feature importance chart returned by GET /global-importance."""
    chart: dict[str, Any] = Field(description="Plotly figure JSON for the importance bar chart")
