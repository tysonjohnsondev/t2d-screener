"""
api/main.py
-----------
FastAPI application for the T2D Risk Screener.

Endpoints:
  GET  /                   → Serve frontend HTML
  GET  /static/{path}      → Serve CSS / JS
  POST /predict            → Run model inference, return PredictionResponse
  GET  /global-importance  → Return pre-computed global SHAP chart
  GET  /metadata           → Return model metadata and test metrics
  GET  /health             → Liveness check

Run locally:
    uvicorn api.main:app --reload --port 8000

The model bundle is loaded once at startup via the lifespan context manager
and injected into request handlers through app.state.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.model import (
    ModelBundle,
    build_global_importance_chart,
    load_model_bundle,
    predict,
)
from src.preprocessing import prepare_single_prediction
from src.schemas import GlobalImportanceResponse, PatientInput, PredictionResponse

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MODELS_DIR   = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts at startup; release on shutdown."""
    print("Loading model bundle …")
    app.state.bundle: ModelBundle = load_model_bundle(MODELS_DIR)
    print(
        f"Model ready — threshold={app.state.bundle.threshold:.4f}  "
        f"tiers={app.state.bundle.tier_labels}"
    )
    yield
    print("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="T2D Risk Screener",
    description=(
        "Screening tool for undiagnosed Type 2 Diabetes risk, trained on the "
        "South Korean National Health Insurance Service (NHIS) 2024 health examination dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Return the main single-page application."""
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not built.")
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: Request, patient: PatientInput):
    """
    Accept patient biometric and lab data; return calibrated risk probability,
    risk tier, clinical recommendation, and a SHAP waterfall chart.
    """
    bundle: ModelBundle = request.app.state.bundle

    try:
        X, _ = prepare_single_prediction(
            age=patient.age,
            sex_code=patient.sex_code,
            height_cm=patient.height_cm,
            weight_kg=patient.weight_kg,
            waist_cm=patient.waist_cm,
            systolic_bp=patient.systolic_bp,
            diastolic_bp=patient.diastolic_bp,
            urine_protein=patient.urine_protein,
            smoking_status=patient.smoking_status,
            alcohol_consumption=patient.alcohol_consumption,
            hemoglobin=patient.hemoglobin,
            serum_creatinine=patient.serum_creatinine,
            ast=patient.ast,
            alt=patient.alt,
            ggt=patient.ggt,
            total_cholesterol=patient.total_cholesterol,
            triglycerides=patient.triglycerides,
            hdl_cholesterol=patient.hdl_cholesterol,
            ldl_cholesterol=patient.ldl_cholesterol,
            nominal_features=bundle.nominal_features,
            model_features=bundle.model_features,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Input preprocessing failed: {exc}") from exc

    try:
        result = predict(bundle, X)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    return PredictionResponse(**result)


@app.get("/metadata")
async def metadata_endpoint(request: Request):
    """
    Return model metadata and held-out test performance metrics.
    Used by the About page to populate the metrics table.
    """
    bundle: ModelBundle = request.app.state.bundle
    return {
        "threshold":          round(bundle.threshold, 6),
        "risk_tier_labels":   bundle.tier_labels,
        "risk_tier_edges":    [round(e, 4) for e in bundle.tier_edges],
        "n_features":         len(bundle.model_features),
        "test_roc_auc":       bundle.test_roc_auc,
        "test_avg_precision": bundle.test_avg_precision,
        "test_brier":         bundle.test_brier,
    }


@app.get("/global-importance", response_model=GlobalImportanceResponse)
async def global_importance_endpoint(request: Request):
    """
    Return the pre-computed global feature importance chart (mean |SHAP|).
    Suitable for embedding in an 'About the model' section.
    """
    bundle: ModelBundle = request.app.state.bundle

    if not bundle.global_shap:
        raise HTTPException(
            status_code=503,
            detail="Global SHAP data not available. Run train.py first.",
        )

    chart = build_global_importance_chart(bundle.global_shap)
    return GlobalImportanceResponse(chart=chart)
