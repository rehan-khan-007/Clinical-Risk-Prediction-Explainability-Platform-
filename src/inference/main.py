"""
Phase 5: FastAPI inference service.

Loads the persisted model artifact once at startup (not per-request),
serves predictions with SHAP-based local explanations, and — only with
explicit consent — logs each prediction to the database.

Run:
    uvicorn src.inference.main:app --reload --port 8000
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.inference.database import Prediction, init_db, get_db
from src.inference.schemas import (
    PredictionRequest, PredictionResponse, ContributionItem, HealthResponse,
)

MODEL_DIR = Path("models")

app = FastAPI(
    title="Clinical Risk Prediction API",
    description="Portfolio/educational project — not a medical device. See /predict response disclaimer.",
    version="1.0.0",
)

import os

# Allow the frontend (local dev + deployed) to call this API. Set
# ALLOWED_ORIGINS on Render after the Vercel URL exists (comma-separated),
# e.g. "https://clinical-risk-platform.vercel.app,http://localhost:5173".
# Defaults to "*" for local dev before that's configured.
_origins = os.environ.get("ALLOWED_ORIGINS", "*")
allow_origins = _origins.split(",") if _origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_state = {"model": None, "scaler": None, "metadata": None, "explainer": None}


@app.on_event("startup")
def load_model():
    _state["model"] = joblib.load(MODEL_DIR / "model.joblib")
    _state["scaler"] = joblib.load(MODEL_DIR / "scaler.joblib")
    with open(MODEL_DIR / "model_metadata.json") as f:
        _state["metadata"] = json.load(f)
    _state["explainer"] = shap.TreeExplainer(_state["model"])
    init_db()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_version=_state["metadata"]["version"] if _state["metadata"] else "unloaded",
        model_loaded=_state["model"] is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metadata = _state["metadata"]
    feature_columns = metadata["feature_columns"]
    threshold = metadata["threshold"]

    raw_row = {col: getattr(request, col) for col in feature_columns}
    X_raw = pd.DataFrame([raw_row])[feature_columns]

    X_scaled = X_raw.copy()
    numeric_cols = X_scaled.select_dtypes(include="number").columns
    X_scaled[numeric_cols] = _state["scaler"].transform(X_raw[numeric_cols])

    proba = float(_state["model"].predict_proba(X_scaled)[0, 1])
    risk_class = "high" if proba >= threshold else "low"

    shap_values = _state["explainer"].shap_values(X_scaled)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    elif shap_values.ndim == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    contributions = sorted(
        zip(feature_columns, sv), key=lambda x: abs(x[1]), reverse=True
    )[:5]
    top_contributions = [
        ContributionItem(
            feature=feat, shap_value=float(val),
            direction="increases" if val > 0 else "decreases",
        )
        for feat, val in contributions
    ]

    if request.consent_given:
        db_row = Prediction(
            consent_given=True,
            nickname=request.nickname,
            inputs=raw_row,
            model_version=metadata["version"],
            risk_probability=proba,
            risk_class=risk_class,
            threshold_used=threshold,
            top_contributions=[c.model_dump() for c in top_contributions],
        )
        db.add(db_row)
        db.commit()

    return PredictionResponse(
        risk_probability=proba,
        risk_class=risk_class,
        model_version=metadata["version"],
        threshold_used=threshold,
        top_contributions=top_contributions,
    )


@app.get("/history/count")
def history_count(db: Session = Depends(get_db)):
    """Aggregate count only — never exposes individual stored predictions
    publicly, since those are personal inputs even if anonymous."""
    count = db.query(Prediction).count()
    return {"total_consented_predictions": count}
