"""
Request/response schemas. Validation ranges mirror the cleaning rules in
prepare_cardio.py's config, so the API rejects the same physiologically
implausible inputs the training pipeline would have dropped.
"""
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class PredictionRequest(BaseModel):
    consent_given: bool = Field(..., description="Must be true to receive and store a prediction")
    nickname: Optional[str] = Field(None, max_length=50, description="Optional, never required")

    age: float = Field(..., ge=18, le=100, description="Age in years")
    gender: int = Field(..., ge=1, le=2, description="1=female, 2=male (per source dataset encoding)")
    height: float = Field(..., ge=120, le=220, description="Height in cm")
    weight: float = Field(..., ge=30, le=200, description="Weight in kg")
    ap_hi: float = Field(..., ge=80, le=250, description="Systolic blood pressure (mmHg)")
    ap_lo: float = Field(..., ge=40, le=200, description="Diastolic blood pressure (mmHg)")
    cholesterol: int = Field(..., ge=1, le=3, description="1=normal, 2=above normal, 3=well above normal")
    gluc: int = Field(..., ge=1, le=3, description="1=normal, 2=above normal, 3=well above normal")
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1, description="Alcohol intake: 0=no, 1=yes")
    active: int = Field(..., ge=0, le=1, description="Physically active: 0=no, 1=yes")

    @model_validator(mode="after")
    def check_bp_consistency(self):
        if self.ap_hi < self.ap_lo:
            raise ValueError("ap_hi (systolic) must be >= ap_lo (diastolic) — check your inputs")
        return self


class ContributionItem(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases" | "decreases"


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_class: str
    model_version: str
    threshold_used: float
    top_contributions: list[ContributionItem]
    disclaimer: str = (
        "This is a portfolio/educational project, not a medical device. "
        "It does not diagnose disease and should never replace professional "
        "medical advice. If you have health concerns, please see a doctor."
    )


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
