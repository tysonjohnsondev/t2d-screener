"""
model.py
--------
Model loading, inference, and SHAP explanation generation.

All state is loaded once at startup and held in a ModelBundle dataclass.
The bundle is constructed by load_model_bundle() and injected into the
FastAPI app via lifespan dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

from src.config import (
    FEATURE_DISPLAY_NAMES,
    MODELS_DIR,
    RISK_TIER_COLORS,
    RISK_TIER_LABELS,
    RISK_TIER_RECOMMENDATIONS,
)


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    model: CatBoostClassifier
    calibrator: LogisticRegression
    threshold: float
    tier_edges: list[float]
    tier_labels: list[str]
    model_features: list[str]
    nominal_features: list[str]
    cat_feature_indices: list[int]
    global_shap: dict[str, float] = field(default_factory=dict)
    test_roc_auc: float | None = None
    test_avg_precision: float | None = None
    test_brier: float | None = None


def load_model_bundle(models_dir: Path = MODELS_DIR) -> ModelBundle:
    """Load all model artefacts from disk and return a ModelBundle."""
    model = CatBoostClassifier()
    model.load_model(str(models_dir / "catboost_model.cbm"))

    calibrator = joblib.load(models_dir / "calibrator.pkl")

    meta = json.loads((models_dir / "metadata.json").read_text())

    global_shap: dict[str, float] = {}
    shap_path = models_dir / "global_shap.json"
    if shap_path.exists():
        global_shap = json.loads(shap_path.read_text())

    return ModelBundle(
        model=model,
        calibrator=calibrator,
        threshold=meta["threshold"],
        tier_edges=meta["risk_tier_edges"],
        tier_labels=meta.get("risk_tier_labels", list(RISK_TIER_LABELS)),
        model_features=meta["model_features"],
        nominal_features=meta["nominal_features"],
        cat_feature_indices=meta["cat_feature_indices"],
        global_shap=global_shap,
        test_roc_auc=meta.get("test_roc_auc"),
        test_avg_precision=meta.get("test_avg_precision"),
        test_brier=meta.get("test_brier"),
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _prob_to_logit(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    return np.log(prob / (1 - prob))


def calibrate(calibrator: LogisticRegression, raw_prob: np.ndarray) -> np.ndarray:
    logits = _prob_to_logit(raw_prob).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


# ---------------------------------------------------------------------------
# Risk tier assignment
# ---------------------------------------------------------------------------

def assign_risk_tier(prob: float, edges: list[float], labels: list[str]) -> str:
    """Map a calibrated probability to its risk tier label."""
    for i, label in enumerate(labels):
        if prob <= edges[i + 1]:
            return label
    return labels[-1]


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    cat_feature_indices: list[int],
) -> tuple[np.ndarray, float]:
    """
    Compute per-feature SHAP values for a single-row DataFrame using
    CatBoost's native ShapValues implementation.

    Returns (shap_values, base_value) both in log-odds space.
    The base value is the average model output across the training data.
    """
    pool = Pool(X, cat_features=cat_feature_indices)
    shap_matrix = model.get_feature_importance(pool, type="ShapValues")
    # shap_matrix shape: (n_rows, n_features + 1); last col is bias / base value
    shap_vals = shap_matrix[0, :-1]
    base_value = float(shap_matrix[0, -1])
    return shap_vals, base_value


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------

def build_waterfall_chart(
    shap_values: np.ndarray,
    feature_names: list[str],
    base_value: float,
    n_features: int = 12,
    missing_mask: dict[str, bool] | None = None,
) -> dict:
    """
    Build a Plotly horizontal waterfall chart showing how each feature
    contributed to pushing the risk score above or below the baseline.

    Parameters
    ----------
    missing_mask : optional dict mapping feature name → True if the user
        did not provide that value (i.e. it was NaN in the input).  When
        True, "(not provided)" is appended to the feature label so the
        reader understands why the bar appears despite no value being entered.

    Returns a Plotly figure serialised as a dict (JSON-safe).
    """
    if missing_mask is None:
        missing_mask = {}

    # Select top-N features by absolute SHAP value
    abs_vals = np.abs(shap_values)
    top_idx  = np.argsort(abs_vals)[::-1][:n_features]

    # Sort ascending for the chart (least important at top, most at bottom)
    top_idx  = top_idx[::-1]

    display_names = []
    for i in top_idx:
        fname = feature_names[i]
        label = FEATURE_DISPLAY_NAMES.get(fname, fname)
        if missing_mask.get(fname, False):
            label = f"{label} (not provided)"
        display_names.append(label)

    values = [float(shap_values[i]) for i in top_idx]
    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in values]
    # Pre-format hover labels in Python to avoid Plotly.js format-string quirks
    hover_labels = [f"{v:+.3f}" for v in values]

    fig = go.Figure()

    # Main bars (no legend entry)
    fig.add_trace(go.Bar(
        x=values,
        y=display_names,
        orientation="h",
        marker_color=colors,
        customdata=hover_labels,
        hovertemplate="%{y}: %{customdata}<extra></extra>",
        showlegend=False,
    ))

    # Invisible dummy traces that drive a clean color legend
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        orientation="h",
        name="Increases risk",
        marker_color="#ef4444",
        showlegend=True,
    ))
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        orientation="h",
        name="Decreases risk",
        marker_color="#3b82f6",
        showlegend=True,
    ))

    fig.add_vline(
        x=0,
        line_width=1,
        line_dash="solid",
        line_color="#6b7280",
    )

    fig.update_layout(
        title={
            "text": "Feature contributions to this prediction",
            "font": {"size": 15, "color": "#111827"},
            "x": 0,
        },
        xaxis={
            "title": "Impact on risk score (log-odds scale)",
            "zeroline": False,
            "title_font": {"size": 12},
        },
        yaxis={"automargin": True},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin={"l": 10, "r": 20, "t": 70, "b": 50},
        height=max(300, 30 * n_features + 100),
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11, "color": "#6b7280"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
        },
        font={"family": "Inter, system-ui, sans-serif", "size": 12, "color": "#374151"},
    )

    return fig.to_dict()


def build_global_importance_chart(global_shap: dict[str, float], n_features: int = 15) -> dict:
    """
    Build a horizontal bar chart of mean |SHAP| values across the test set.
    Used on the About / methodology page.
    """
    top = list(global_shap.items())[:n_features]
    top = top[::-1]  # ascending for chart

    names  = [FEATURE_DISPLAY_NAMES.get(k, k) for k, _ in top]
    values = [v for _, v in top]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color="#6366f1",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "Overall feature importance (mean |SHAP|, test set)",
            "font": {"size": 15, "color": "#111827"},
            "x": 0,
        },
        xaxis={
            "title": "Mean absolute SHAP value",
            "zeroline": False,
        },
        yaxis={"automargin": True},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin={"l": 10, "r": 20, "t": 50, "b": 30},
        height=max(300, 28 * n_features + 80),
        showlegend=False,
        font={"family": "Inter, system-ui, sans-serif", "size": 12, "color": "#374151"},
    )

    return fig.to_dict()


# ---------------------------------------------------------------------------
# Full prediction pipeline
# ---------------------------------------------------------------------------

def predict(bundle: ModelBundle, X: pd.DataFrame) -> dict:
    """
    Run the full inference pipeline for a single patient row.

    Returns a dict with probability, risk tier, screen status, SHAP chart,
    and the clinical recommendation.
    """
    pool = Pool(X, cat_features=bundle.cat_feature_indices)
    raw_prob = bundle.model.predict_proba(pool)[:, 1]
    cal_prob = float(calibrate(bundle.calibrator, raw_prob)[0])

    screen_positive = cal_prob >= bundle.threshold
    tier = assign_risk_tier(cal_prob, bundle.tier_edges, bundle.tier_labels)

    shap_vals, base_value = compute_shap_values(
        bundle.model, X, bundle.cat_feature_indices
    )

    # Build a mask so the chart can label blank optional fields clearly
    missing_mask = {
        col: bool(pd.isna(X[col].iloc[0]))
        for col in X.columns
    }

    waterfall = build_waterfall_chart(
        shap_values=shap_vals,
        feature_names=list(X.columns),
        base_value=base_value,
        missing_mask=missing_mask,
    )

    return {
        "probability":       round(cal_prob, 4),
        "screen_positive":   bool(screen_positive),
        "risk_tier":         tier,
        "risk_tier_color":   RISK_TIER_COLORS[tier],
        "recommendation":    RISK_TIER_RECOMMENDATIONS[tier],
        "waterfall_chart":   waterfall,
    }
