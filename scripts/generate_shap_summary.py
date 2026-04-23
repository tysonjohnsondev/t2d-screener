"""
generate_shap_summary.py
------------------------
Regenerate ``figures/shap_importance.png`` — the global mean-absolute SHAP
bar chart (top 20 features).

SHAP values are computed via CatBoost's native
``get_feature_importance(Pool, type="ShapValues")`` (exact, fast, CPU-only)
rather than the ``shap`` Python library. Up to ``MAX_SAMPLE`` test rows are
used.

Run from the project root::

    python scripts/generate_shap_summary.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    FEATURE_DISPLAY_NAMES,
    MODELS_DIR,
    TARGET_CAT,
)
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
MAX_SAMPLE: int = 10_000
TOP_N: int = 20
FIGURES_DIR: Path = ROOT_DIR / "figures"
OUTPUT_PATH: Path = FIGURES_DIR / "shap_importance.png"


def set_seeds(seed: int = SEED) -> None:
    """Seed ``random`` and ``numpy`` for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_model_and_metadata() -> tuple[CatBoostClassifier, dict]:
    """Load the committed CatBoost model and its metadata."""
    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / "catboost_model.cbm"))
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
    return model, metadata


def build_test_frame(model_features: list[str]) -> pd.DataFrame:
    """Rebuild the same held-out test features the committed model was evaluated on."""
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

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    train_test_split(
        X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train
    )
    _, _, X_test_cb, _ = prepare_catboost_inputs(
        X_train[available], X_train[available].iloc[:1], X_test[available]
    )
    return X_test_cb[available]


def compute_mean_abs_shap(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    cat_indices: list[int],
    sample_size: int,
) -> pd.Series:
    """Return mean absolute SHAP per feature over ``min(sample_size, len(X))`` rows."""
    n = min(sample_size, len(X))
    idx = np.random.choice(len(X), n, replace=False)
    X_sample = X.iloc[idx]
    pool = Pool(X_sample, cat_features=cat_indices)
    shap_matrix = model.get_feature_importance(pool, type="ShapValues")
    shap_vals = shap_matrix[:, :-1]  # last column is the bias term
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)


def plot_shap_summary(series: pd.Series, top_n: int, output_path: Path) -> None:
    """Save a horizontal bar chart of the top_n features by mean absolute SHAP."""
    top = series.head(top_n).iloc[::-1]  # reversed so largest ends up on top
    display_labels = [FEATURE_DISPLAY_NAMES.get(feat, feat) for feat in top.index]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.barh(display_labels, top.values, color="#1a5276", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|  (log-odds units)", fontsize=12)
    ax.set_title(
        "Global feature importance (mean |SHAP|)",
        fontsize=14, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="x", alpha=0.3)

    for i, v in enumerate(top.values):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=9, color="#111")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Regenerate the global SHAP importance figure.

    Loads the committed CatBoost model, rebuilds the seed-42 test split,
    samples up to ``MAX_SAMPLE`` rows, computes exact SHAP values via
    CatBoost's native API, and plots the top ``TOP_N`` features by mean
    absolute SHAP.
    """
    set_seeds()

    print("Loading committed CatBoost model ...")
    model, metadata = load_model_and_metadata()
    model_features: list[str] = list(metadata["model_features"])
    cat_indices: list[int] = list(metadata["cat_feature_indices"])

    print("Rebuilding held-out test features (seed=42) ...")
    X_test_cb = build_test_frame(model_features)
    X_test_cb = X_test_cb[model_features]
    print(f"  test frame: {X_test_cb.shape}")

    print(f"Computing SHAP values on up to {MAX_SAMPLE:,} rows ...")
    mean_abs = compute_mean_abs_shap(model, X_test_cb, cat_indices, MAX_SAMPLE)

    print(f"\nTop {TOP_N} features by mean |SHAP|:")
    for feat, val in mean_abs.head(TOP_N).items():
        label = FEATURE_DISPLAY_NAMES.get(feat, feat)
        print(f"  {label:<40} {val:.4f}")

    plot_shap_summary(mean_abs, TOP_N, OUTPUT_PATH)
    print(f"\nFigure saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
