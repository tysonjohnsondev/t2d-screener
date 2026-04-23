"""
train.py
--------
Trains the CatBoost T2D screening model on the full NHIS dataset,
calibrates probabilities, selects the screening threshold, computes
global SHAP feature importance, and saves all artefacts to models/.

Run from the project root:
    python train.py

Outputs written to models/:
    catboost_model.cbm   — trained CatBoost model
    calibrator.pkl       — Platt calibrator (sklearn LogisticRegression)
    metadata.json        — threshold, risk-tier edges, feature list
    global_shap.json     — mean |SHAP| per feature (for the About page)
"""

import json
import random

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    MODELS_DIR,
    MODEL_FEATURES,
    MODEL_NOMINAL_FEATURES,
    RISK_TIER_LABELS,
    RISK_TIER_QUANTILES,
    TARGET_CAT,
    TARGET_RECALL,
)
import shap

from src.preprocessing import (
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

SEED = 42
ITERATIONS = 1500
LEARNING_RATE = 0.03
DEPTH = 6
L2_LEAF_REG = 5.0
SHAP_SAMPLE_SIZE = 5_000  # rows used to compute global SHAP importance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def prob_to_logit(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    return np.log(prob / (1 - prob))


def fit_calibrator(raw_prob: np.ndarray, y: pd.Series) -> LogisticRegression:
    logits = prob_to_logit(raw_prob).reshape(-1, 1)
    cal = LogisticRegression(random_state=SEED)
    cal.fit(logits, y)
    return cal


def apply_calibrator(cal: LogisticRegression, raw_prob: np.ndarray) -> np.ndarray:
    logits = prob_to_logit(raw_prob).reshape(-1, 1)
    return cal.predict_proba(logits)[:, 1]


def choose_threshold(y_true: pd.Series, y_prob: np.ndarray, target_recall: float) -> float:
    """Return the lowest threshold that achieves target_recall while maximising precision."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions, recalls = precisions[:-1], recalls[:-1]
    candidates = np.where(recalls >= target_recall)[0]
    if len(candidates) == 0:
        idx = int(np.argmax(recalls))
    else:
        idx = int(candidates[np.argmax(precisions[candidates])])
    return float(thresholds[idx])


def compute_risk_tier_edges(y_prob: np.ndarray, quantiles: tuple) -> list[float]:
    """Compute probability thresholds for risk tier boundaries from the test distribution."""
    edges = list(np.quantile(y_prob, quantiles).tolist())
    edges[0] = 0.0
    edges[-1] = 1.0
    # Ensure strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(1.0, edges[i - 1] + 1e-6)
    return edges


def assign_risk_tier(prob: float, edges: list[float], labels: tuple) -> str:
    for i, label in enumerate(labels):
        if prob <= edges[i + 1]:
            return label
    return labels[-1]


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    set_seeds()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and clean data
    print("Loading data …")
    download_dataset()
    df_raw = load_raw_data()
    df = tidy_column_names(df_raw)
    df = create_blindness_features(df)   # flag blind eyes before validation
    df = validate_dataframe(df)          # cast dtypes, null out-of-domain values
    df = create_features(df)             # has_diabetes label + wwi (needs clean numerics)
    df = create_screening_features(df)   # bmi, waist_to_height, log labs, lipid ratios …
    df = create_panel_features(df)       # has_complete_lipid_panel, n_lipids_available

    missing_flag_cols = [
        "hemoglobin", "serum_creatinine", "serum_got_ast",
        "serum_gpt_alt", "gamma_gtp",
        "total_cholesterol", "triglycerides", "hdl_cholesterol", "ldl_cholesterol",
    ]
    df = create_missingness_features(df, missing_flag_cols)

    # 2. Build modelling dataset
    available = [f for f in MODEL_FEATURES if f in df.columns]
    missing_cols = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing_cols:
        print(f"Warning: features missing from DataFrame: {missing_cols}")

    df_model = df[available + [TARGET_CAT]].copy()
    df_model = df_model[df_model[TARGET_CAT].notna()].reset_index(drop=True)

    # 3. Train / validation / test split (stratified)
    y = df_model[TARGET_CAT].astype(int)
    X = df_model[available].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train
    )
    print(
        f"Split: train={len(y_train):,}  valid={len(y_valid):,}  test={len(y_test):,}  "
        f"pos-rate={y.mean():.3%}"
    )

    # 4. Prepare CatBoost inputs
    X_train_cb, X_valid_cb, X_test_cb, cat_indices = prepare_catboost_inputs(
        X_train[available], X_valid[available], X_test[available]
    )

    # 5. Class weights (inverse frequency)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    class_weights = [1.0, neg / max(pos, 1)]

    # 6. Train CatBoost
    print("Training CatBoost …")
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        l2_leaf_reg=L2_LEAF_REG,
        random_seed=SEED,
        verbose=100,
        allow_writing_files=False,
        class_weights=class_weights,
    )
    model.fit(
        X_train_cb, y_train,
        cat_features=cat_indices,
        eval_set=(X_valid_cb, y_valid),
        use_best_model=True,
        early_stopping_rounds=100,
    )

    # 7. Platt calibration on validation set
    print("Calibrating probabilities …")
    raw_valid = model.predict_proba(X_valid_cb)[:, 1]
    raw_test  = model.predict_proba(X_test_cb)[:, 1]

    calibrator = fit_calibrator(raw_valid, y_valid)
    cal_valid  = apply_calibrator(calibrator, raw_valid)
    cal_test   = apply_calibrator(calibrator, raw_test)

    # 8. Threshold selection (target recall on validation set)
    threshold = choose_threshold(y_valid, cal_valid, TARGET_RECALL)
    print(f"Selected threshold: {threshold:.6f}")

    # 9. Evaluate on test set
    roc  = roc_auc_score(y_test, cal_test)
    ap   = average_precision_score(y_test, cal_test)
    brier = brier_score_loss(y_test, cal_test)
    y_pred = (cal_test >= threshold).astype(int)
    recall_at_t = y_test[y_pred == 1].sum() / y_test.sum() if y_test.sum() > 0 else 0
    print(
        f"Test  ROC-AUC={roc:.4f}  AvgPrecision={ap:.4f}  "
        f"Brier={brier:.4f}  Recall@threshold={recall_at_t:.4f}"
    )

    # 10. Risk-tier probability edges (from test distribution)
    tier_edges = compute_risk_tier_edges(cal_test, RISK_TIER_QUANTILES)
    print(f"Risk-tier edges: {[f'{e:.4f}' for e in tier_edges]}")

    # 11. Global SHAP feature importance (from a sample of the test set)
    print("Computing global SHAP importance …")
    sample_idx = np.random.choice(len(X_test_cb), min(SHAP_SAMPLE_SIZE, len(X_test_cb)), replace=False)
    X_shap = X_test_cb.iloc[sample_idx]

    pool_shap = Pool(X_shap, cat_features=cat_indices)
    shap_matrix = model.get_feature_importance(pool_shap, type="ShapValues")
    shap_vals = shap_matrix[:, :-1]  # last column is bias

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    global_shap = {
        feat: float(val)
        for feat, val in zip(available, mean_abs_shap)
    }
    # Sort descending
    global_shap = dict(sorted(global_shap.items(), key=lambda x: x[1], reverse=True))

    # 12. Save artefacts
    model_path = MODELS_DIR / "catboost_model.cbm"
    model.save_model(str(model_path))
    print(f"Model saved → {model_path}")

    cal_path = MODELS_DIR / "calibrator.pkl"
    joblib.dump(calibrator, cal_path)
    print(f"Calibrator saved → {cal_path}")

    meta = {
        "threshold":          threshold,
        "risk_tier_edges":    tier_edges,
        "risk_tier_labels":   list(RISK_TIER_LABELS),
        "model_features":     available,
        "nominal_features":   MODEL_NOMINAL_FEATURES,
        "cat_feature_indices": cat_indices,
        "test_roc_auc":       roc,
        "test_avg_precision": ap,
        "test_brier":         brier,
        "best_iteration":     int(model.get_best_iteration()),
    }
    meta_path = MODELS_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved → {meta_path}")

    shap_path = MODELS_DIR / "global_shap.json"
    shap_path.write_text(json.dumps(global_shap, indent=2))
    print(f"Global SHAP saved → {shap_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
