"""
generate_confusion_matrix.py
----------------------------
Regenerate ``figures/catboost_confusion_matrix.png``.

Loads the committed production bundle (CatBoost + Platt calibrator + metadata),
rebuilds the same stratified 60/20/20 split used in ``train.py`` (seed 42) so
the held-out test set matches the one the committed metrics were computed on,
computes calibrated probabilities, thresholds at ``metadata.json["threshold"]``
(which was originally derived from ``TARGET_RECALL`` = 0.85 sensitivity on the
validation set during training), and plots a 2x2 confusion matrix with counts
and row-normalised percentages.

Run from the project root::

    python scripts/generate_confusion_matrix.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Make ``src`` importable when this script is launched from anywhere.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import MODELS_DIR, TARGET_CAT  # noqa: E402
from src.preprocessing import (  # noqa: E402
    create_blindness_features,
    create_features,
    create_missingness_features,
    create_panel_features,
    create_screening_features,
    download_dataset,
    load_raw_data,
    prepare_catboost_inputs,
    tidy_column_names,
    validate_dataframe,
)

SEED: int = 42
FIGURES_DIR: Path = ROOT_DIR / "figures"
OUTPUT_PATH: Path = FIGURES_DIR / "catboost_confusion_matrix.png"


def set_seeds(seed: int = SEED) -> None:
    """Seed ``random`` and ``numpy`` for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def prob_to_logit(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Map probabilities in (0, 1) to logits, with clipping for numerical safety."""
    prob = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    return np.log(prob / (1 - prob))


def apply_calibrator(calibrator: LogisticRegression, raw_prob: np.ndarray) -> np.ndarray:
    """Apply a fitted Platt calibrator (logistic regression on logits) to raw scores."""
    logits = prob_to_logit(raw_prob).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def load_bundle() -> tuple[CatBoostClassifier, LogisticRegression, dict]:
    """Load the committed CatBoost model, Platt calibrator, and metadata."""
    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / "catboost_model.cbm"))
    calibrator = joblib.load(MODELS_DIR / "calibrator.pkl")
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
    return model, calibrator, metadata


def build_test_set(model_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Rebuild the exact held-out test split used at training time (seed 42)."""
    download_dataset()
    df_raw = load_raw_data()
    df = tidy_column_names(df_raw)
    df = create_blindness_features(df)
    df = validate_dataframe(df)
    df = create_features(df)
    df = create_screening_features(df)
    df = create_panel_features(df)

    missing_flag_cols = [
        "hemoglobin", "serum_creatinine", "serum_got_ast",
        "serum_gpt_alt", "gamma_gtp",
        "total_cholesterol", "triglycerides", "hdl_cholesterol", "ldl_cholesterol",
    ]
    df = create_missingness_features(df, missing_flag_cols)

    available = [f for f in model_features if f in df.columns]
    df_model = df[available + [TARGET_CAT]].copy()
    df_model = df_model[df_model[TARGET_CAT].notna()].reset_index(drop=True)

    y = df_model[TARGET_CAT].astype(int)
    X = df_model[available].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    # The inner split is performed for parity with train.py even though we only
    # need the test slice here.
    train_test_split(
        X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train
    )

    # Format to CatBoost-ready dtypes. Only the test frame is returned.
    _, _, X_test_cb, _ = prepare_catboost_inputs(
        X_train[available], X_train[available].iloc[:1], X_test[available]
    )
    return X_test_cb, y_test


def plot_confusion_matrix(
    cm: np.ndarray,
    threshold: float,
    sensitivity: float,
    specificity: float,
    output_path: Path,
) -> None:
    """Save a labelled 2x2 confusion matrix figure to ``output_path``."""
    class_labels = ["No T2D", "Likely T2D"]
    row_totals = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals > 0)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels, fontsize=11)
    ax.set_yticklabels(class_labels, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            pct = row_pct[i, j] * 100
            text_color = "white" if row_pct[i, j] > 0.5 else "#111"
            ax.text(
                j, i,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=13, fontweight="bold", color=text_color,
            )

    title = (
        f"CatBoost confusion matrix (test set)\n"
        f"threshold = {threshold:.4f}  |  "
        f"sensitivity = {sensitivity:.3f}  |  specificity = {specificity:.3f}"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Row-normalised share", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Regenerate the committed CatBoost confusion-matrix figure.

    Loads the committed model bundle, rebuilds the seed-42 test split to match
    training, computes calibrated test probabilities, applies the metadata
    threshold, and writes ``figures/catboost_confusion_matrix.png``.

    The threshold is read from ``models/metadata.json`` (originally chosen as
    the lowest threshold on validation achieving at least
    ``TARGET_RECALL = 0.85`` sensitivity during training).
    """
    set_seeds()

    print("Loading committed model bundle ...")
    model, calibrator, metadata = load_bundle()
    threshold = float(metadata["threshold"])
    model_features = list(metadata["model_features"])
    print(f"  threshold = {threshold:.6f}")

    print("Rebuilding held-out test split (seed=42) ...")
    X_test_cb, y_test = build_test_set(model_features)
    print(f"  test set: n={len(y_test):,}  positives={int(y_test.sum()):,}")

    # Align feature order with the model's expectation.
    X_test_cb = X_test_cb[model_features]

    print("Scoring test set ...")
    raw_prob = model.predict_proba(X_test_cb)[:, 1]
    cal_prob = apply_calibrator(calibrator, raw_prob)
    y_pred = (cal_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    print("\nConfusion matrix (rows=true, cols=pred)")
    print(cm)
    print(f"\nSensitivity (recall)     : {sensitivity:.4f}")
    print(f"Specificity              : {specificity:.4f}")
    print(f"Positive predictive value: {ppv:.4f}")
    print(f"Negative predictive value: {npv:.4f}")

    plot_confusion_matrix(cm, threshold, sensitivity, specificity, OUTPUT_PATH)
    print(f"\nFigure saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
