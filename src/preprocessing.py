"""
preprocessing.py
----------------
Data cleaning, feature engineering, and inference-time input preparation.

Two usage modes:
  1. Training pipeline  — operates on a full DataFrame loaded from the NHIS CSV.
  2. Inference pipeline — converts a single PatientInput (raw user values) into
     the feature DataFrame the CatBoost model expects.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import (
    COLUMN_DOMAINS,
    COLUMN_NAMES_INIT,
    COLUMN_NAMES_TO_DROP,
    COLUMN_TYPES,
    COLUMNS_ALL,
    DATA_ENCODING,
    DATA_FILE,
    FEATURES_MISSING_ALLOWED,
    GH_URL,
    MODEL_FEATURES,
    MODEL_NOMINAL_FEATURES,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def download_dataset(
    url: str = GH_URL,
    data_path: Path = DATA_FILE,
) -> None:
    """Download the NHIS dataset from GitHub if not already present."""
    if data_path.exists():
        print("Dataset already downloaded.")
        return
    data_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url} …")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    data_path.write_bytes(response.content)
    print(f"Saved to {data_path}")


def load_raw_data(data_path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the raw NHIS CSV."""
    return pd.read_csv(data_path, encoding=DATA_ENCODING)


# ---------------------------------------------------------------------------
# Training pipeline helpers
# ---------------------------------------------------------------------------

def tidy_column_names(
    data: pd.DataFrame,
    col_names_init: list[str] = COLUMN_NAMES_INIT,
    cols_to_drop: list[str] = COLUMN_NAMES_TO_DROP,
    cols_in_order: list[str] = COLUMNS_ALL,
) -> pd.DataFrame:
    """Rename columns, drop irrelevant ones, and reorder."""
    df = data.copy()
    df.columns = col_names_init
    df = df.drop(columns=cols_to_drop)
    df = df[cols_in_order]
    return df


def validate_dataframe(
    data: pd.DataFrame,
    col_types: dict = COLUMN_TYPES,
    col_domains: dict = COLUMN_DOMAINS,
    print_removed: bool = False,
) -> pd.DataFrame:
    """Cast columns to correct dtypes and null out out-of-domain values."""
    df = data.copy()
    if print_removed:
        missing_before = df.isna().sum()

    for column, (_, data_type) in col_types.items():
        if data_type == "int":
            df[column] = pd.to_numeric(df[column], errors="coerce").round().astype("Int64")
        elif data_type == "float":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Float64")

        domain = col_domains[column]
        if isinstance(domain, set):
            df.loc[~df[column].isin(domain), column] = pd.NA
        else:
            lo, hi = domain
            df.loc[(df[column] < lo) | (df[column] > hi), column] = pd.NA

    if print_removed:
        print("Invalid entries removed per column:")
        print(df.isna().sum() - missing_before)

    return df


def filter_complete_dataframe(
    data: pd.DataFrame,
    col_names: list[str] = COLUMNS_ALL,
    allowed_missing: dict = FEATURES_MISSING_ALLOWED,
    print_removed: bool = False,
) -> pd.DataFrame:
    """Drop rows with NA values, allowing exceptions for conditional columns."""
    df = data.copy()
    keep = np.ones(len(df), dtype=bool)

    for col in col_names:
        if col in allowed_missing:
            cond_col, cond_val = allowed_missing[col]
            cond = df[cond_col].eq(cond_val).fillna(False) if cond_col in df.columns else False
            keep &= df[col].notna() | cond
        else:
            keep &= df[col].notna()

    out = df.loc[keep].reset_index(drop=True)
    if print_removed:
        print(f"{len(df) - len(out)} incomplete entries removed.")
    return out


# ---------------------------------------------------------------------------
# Feature engineering — shared between training and inference
# ---------------------------------------------------------------------------

def _safe_divide(num: float | None, den: float | None) -> float | None:
    """Scalar safe division returning None on invalid inputs."""
    if num is None or den is None or den == 0 or math.isnan(float(den)):
        return None
    return float(num) / float(den)


def create_blindness_features(data: pd.DataFrame) -> pd.DataFrame:
    """Flag patients with vision == 9.9 as blind and null out that eye's vision."""
    df = data.copy()
    df["blind_left"]  = (df["vision_left"]  == 9.9).astype("Int64")
    df["blind_right"] = (df["vision_right"] == 9.9).astype("Int64")
    df.loc[df["blind_left"]  == 1, "vision_left"]  = pd.NA
    df.loc[df["blind_right"] == 1, "vision_right"] = pd.NA
    return df


def create_diabetes_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create binary diabetes and pre-diabetes labels from fasting plasma glucose."""
    df = data.copy()
    df["has_diabetes"]    = (df["fpg"] >= 126).astype("Int64")
    df["has_prediabetes"] = ((df["fpg"] >= 100) & (df["fpg"] < 126)).astype("Int64")
    return df


def create_wwi_feature(data: pd.DataFrame) -> pd.DataFrame:
    """Weight-adjusted waist index: waist_circumference / sqrt(weight)."""
    df = data.copy()
    df["wwi"] = df["waist_circumference"] / np.sqrt(df["weight"])
    return df


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply diabetes label creation and WWI (used in training pipeline)."""
    df = create_diabetes_features(data)
    df = create_wwi_feature(df)
    return df


def create_screening_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Derive clinically meaningful engineered features.
    Operates on a full DataFrame (training pipeline).
    """
    df = data.copy()

    df["age_years_mid"] = (
        pd.to_numeric(df["age_group_code"], errors="coerce").astype(float) * 5 - 2.5
    )

    height_m  = pd.to_numeric(df["height"], errors="coerce").astype(float) / 100.0
    height_cm = pd.to_numeric(df["height"], errors="coerce").astype(float)
    weight    = pd.to_numeric(df["weight"], errors="coerce").astype(float)
    waist     = pd.to_numeric(df["waist_circumference"], errors="coerce").astype(float)
    systolic  = pd.to_numeric(df["systolic_bp"], errors="coerce").astype(float)
    diastolic = pd.to_numeric(df["diastolic_bp"], errors="coerce").astype(float)
    ast       = pd.to_numeric(df["serum_got_ast"], errors="coerce").astype(float)
    alt       = pd.to_numeric(df["serum_gpt_alt"], errors="coerce").astype(float)
    ggt       = pd.to_numeric(df["gamma_gtp"], errors="coerce").astype(float)
    tot_chol  = pd.to_numeric(df["total_cholesterol"], errors="coerce").astype(float)
    tg        = pd.to_numeric(df["triglycerides"], errors="coerce").astype(float)
    hdl       = pd.to_numeric(df["hdl_cholesterol"], errors="coerce").astype(float)
    ldl       = pd.to_numeric(df["ldl_cholesterol"], errors="coerce").astype(float)

    def _safe_div_series(n: pd.Series, d: pd.Series) -> pd.Series:
        out = n / d
        out[(d <= 0) | d.isna()] = np.nan
        return out

    df["bmi"]                  = _safe_div_series(weight, height_m ** 2)
    df["waist_to_height"]      = _safe_div_series(waist, height_cm)
    df["pulse_pressure"]       = systolic - diastolic
    df["mean_arterial_pressure"] = (systolic + 2 * diastolic) / 3.0
    df["ast_alt_ratio"]        = _safe_div_series(ast, alt)
    df["log_ast"]              = np.log1p(ast)
    df["log_alt"]              = np.log1p(alt)
    df["log_ggt"]              = np.log1p(ggt)
    df["log_triglycerides"]    = np.log1p(tg)
    df["non_hdl_cholesterol"]  = tot_chol - hdl
    df["tg_hdl_ratio"]         = _safe_div_series(tg, hdl)
    df["tc_hdl_ratio"]         = _safe_div_series(tot_chol, hdl)
    df["ldl_hdl_ratio"]        = _safe_div_series(ldl, hdl)

    return df


def create_missingness_features(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add binary missing-value indicator columns for the given feature list."""
    df = data.copy()
    for col in columns:
        df[f"missing_{col}"] = df[col].isna().astype(int)
    return df


def create_panel_features(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise how much of the lipid panel is available per row."""
    df = data.copy()
    lipid_cols = ["total_cholesterol", "triglycerides", "hdl_cholesterol", "ldl_cholesterol"]
    df["has_complete_lipid_panel"] = df[lipid_cols].notna().all(axis=1).astype(int)
    df["n_lipids_available"]       = df[lipid_cols].notna().sum(axis=1).astype(int)
    return df


# ---------------------------------------------------------------------------
# CatBoost input preparation (training)
# ---------------------------------------------------------------------------

def prepare_catboost_inputs(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    nominal_features: list[str] = MODEL_NOMINAL_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int]]:
    """
    Format DataFrames for CatBoost:
      - Numeric columns → float (NaN preserved for native missing handling)
      - Nominal columns → str with 'Missing' sentinel
    Returns (X_train, X_valid, X_test, cat_feature_indices).
    """
    frames = []
    for df in [X_train, X_valid, X_test]:
        df = df.copy()
        for col in df.columns:
            if col in nominal_features:
                df[col] = df[col].astype("string").fillna("Missing")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        frames.append(df)

    cat_indices = [X_train.columns.get_loc(c) for c in nominal_features if c in X_train.columns]
    return frames[0], frames[1], frames[2], cat_indices


# ---------------------------------------------------------------------------
# Inference pipeline — single-row preparation
# ---------------------------------------------------------------------------

def prepare_single_prediction(
    age: int,
    sex_code: int,
    height_cm: float,
    weight_kg: float,
    waist_cm: float,
    systolic_bp: int,
    diastolic_bp: int,
    urine_protein: int,
    smoking_status: int,
    alcohol_consumption: int,
    hemoglobin: float | None = None,
    serum_creatinine: float | None = None,
    ast: float | None = None,
    alt: float | None = None,
    ggt: float | None = None,
    total_cholesterol: float | None = None,
    triglycerides: float | None = None,
    hdl_cholesterol: float | None = None,
    ldl_cholesterol: float | None = None,
    nominal_features: list[str] = MODEL_NOMINAL_FEATURES,
    model_features: list[str] = MODEL_FEATURES,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Convert raw user inputs into a single-row DataFrame ready for CatBoost inference.

    Age is mapped to the midpoint of the corresponding 5-year NHIS age band,
    matching how age_years_mid was computed during training.

    Returns (X, cat_feature_indices).
    """
    # Map age to 5-year band midpoint (NHIS codes 5–18, i.e. 20–24 through 85+)
    age_code = max(5, min(18, int(age) // 5))
    age_years_mid = float(age_code) * 5 - 2.5

    height_m = height_cm / 100.0

    # Derived numeric features
    bmi               = weight_kg / (height_m ** 2)
    waist_to_height   = waist_cm / height_cm
    wwi               = waist_cm / math.sqrt(weight_kg)
    pulse_pressure    = float(systolic_bp - diastolic_bp)
    map_              = (systolic_bp + 2 * diastolic_bp) / 3.0

    log_ast           = math.log1p(ast)            if ast  is not None else None
    log_alt           = math.log1p(alt)            if alt  is not None else None
    log_ggt           = math.log1p(ggt)            if ggt  is not None else None
    ast_alt_ratio     = _safe_divide(ast, alt)

    log_triglycerides = math.log1p(triglycerides)  if triglycerides    is not None else None
    non_hdl           = (total_cholesterol - hdl_cholesterol) \
                        if (total_cholesterol is not None and hdl_cholesterol is not None) \
                        else None
    tg_hdl_ratio      = _safe_divide(triglycerides,    hdl_cholesterol)
    tc_hdl_ratio      = _safe_divide(total_cholesterol, hdl_cholesterol)
    ldl_hdl_ratio     = _safe_divide(ldl_cholesterol,   hdl_cholesterol)

    lipid_vals = [total_cholesterol, triglycerides, hdl_cholesterol, ldl_cholesterol]
    n_lipids   = sum(v is not None for v in lipid_vals)
    has_lipids = int(n_lipids == 4)

    # Missing-value flags
    missing_hemoglobin       = int(hemoglobin       is None)
    missing_serum_creatinine = int(serum_creatinine is None)
    missing_serum_got_ast    = int(ast              is None)
    missing_serum_gpt_alt    = int(alt              is None)
    missing_gamma_gtp        = int(ggt              is None)
    missing_total_cholesterol = int(total_cholesterol is None)
    missing_triglycerides    = int(triglycerides     is None)
    missing_hdl_cholesterol  = int(hdl_cholesterol   is None)
    missing_ldl_cholesterol  = int(ldl_cholesterol   is None)

    raw: dict[str, object] = {
        # Core numeric
        "age_years_mid":            age_years_mid,
        "height":                   float(height_cm),
        "weight":                   float(weight_kg),
        "waist_circumference":      float(waist_cm),
        "bmi":                      bmi,
        "waist_to_height":          waist_to_height,
        "wwi":                      wwi,
        "systolic_bp":              float(systolic_bp),
        "diastolic_bp":             float(diastolic_bp),
        "pulse_pressure":           pulse_pressure,
        "mean_arterial_pressure":   map_,
        "urine_protein":            float(urine_protein),
        # Core nominal
        "sex_code":                 str(sex_code),
        "smoking_status":           str(smoking_status),
        "alcohol_consumption":      str(alcohol_consumption),
        # Non-lipid labs
        "hemoglobin":               hemoglobin,
        "serum_creatinine":         serum_creatinine,
        "log_ast":                  log_ast,
        "log_alt":                  log_alt,
        "log_ggt":                  log_ggt,
        "ast_alt_ratio":            ast_alt_ratio,
        # Non-lipid missing flags
        "missing_hemoglobin":       float(missing_hemoglobin),
        "missing_serum_creatinine": float(missing_serum_creatinine),
        "missing_serum_got_ast":    float(missing_serum_got_ast),
        "missing_serum_gpt_alt":    float(missing_serum_gpt_alt),
        "missing_gamma_gtp":        float(missing_gamma_gtp),
        # Lipid features
        "total_cholesterol":        total_cholesterol,
        "hdl_cholesterol":          hdl_cholesterol,
        "ldl_cholesterol":          ldl_cholesterol,
        "log_triglycerides":        log_triglycerides,
        "non_hdl_cholesterol":      non_hdl,
        "tg_hdl_ratio":             tg_hdl_ratio,
        "tc_hdl_ratio":             tc_hdl_ratio,
        "ldl_hdl_ratio":            ldl_hdl_ratio,
        "has_complete_lipid_panel": float(has_lipids),
        "n_lipids_available":       float(n_lipids),
        # Lipid missing flags
        "missing_total_cholesterol": float(missing_total_cholesterol),
        "missing_triglycerides":     float(missing_triglycerides),
        "missing_hdl_cholesterol":   float(missing_hdl_cholesterol),
        "missing_ldl_cholesterol":   float(missing_ldl_cholesterol),
    }

    # Build DataFrame with features in the exact order the model was trained on
    row: dict[str, object] = {}
    for feat in model_features:
        val = raw.get(feat)
        if feat in nominal_features:
            row[feat] = str(val) if val is not None else "Missing"
        else:
            row[feat] = float(val) if val is not None else float("nan")

    X = pd.DataFrame([row])
    cat_indices = [X.columns.get_loc(c) for c in nominal_features if c in X.columns]
    return X, cat_indices
