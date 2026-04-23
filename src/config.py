"""
config.py
---------
Central configuration for the T2D Risk Screener.
All constants, column definitions, feature lists, and display names live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
DATA_FILE: Path = DATA_DIR / "health_2024.csv"
MODELS_DIR: Path = ROOT_DIR / "models"

DATA_ENCODING: str = "cp949"
GH_URL: str = (
    "https://github.com/tjohns94/cds492-3c-project/raw/refs/heads/main/data/health_2024.CSV"
)

# ---------------------------------------------------------------------------
# Raw dataset schema
# ---------------------------------------------------------------------------
COLUMN_NAMES_INIT: list[str] = [
    "year", "subscriber_id", "city_code",
    "sex_code", "age_group_code", "height",
    "weight", "waist_circumference", "vision_left",
    "vision_right", "hearing_left", "hearing_right",
    "systolic_bp", "diastolic_bp", "fpg",
    "total_cholesterol", "triglycerides", "hdl_cholesterol",
    "ldl_cholesterol", "hemoglobin", "urine_protein",
    "serum_creatinine", "serum_got_ast", "serum_gpt_alt",
    "gamma_gtp", "smoking_status", "alcohol_consumption",
    "oral_exam", "caries_presence", "missing_teeth_presence",
    "tooth_wear_presence", "wisdom_teeth_abnormality", "plaque_presence",
]

COLUMN_NAMES_TO_DROP: list[str] = [
    "year", "subscriber_id", "missing_teeth_presence",
    "tooth_wear_presence", "wisdom_teeth_abnormality",
]

COLUMN_TYPES: dict[str, list] = {
    "sex_code":             ["category", "int"],
    "city_code":            ["category", "int"],
    "hearing_left":         ["category", "int"],
    "hearing_right":        ["category", "int"],
    "alcohol_consumption":  ["category", "int"],
    "oral_exam":            ["category", "int"],
    "caries_presence":      ["category", "int"],
    "plaque_presence":      ["category", "int"],
    "age_group_code":       ["category", "int"],
    "smoking_status":       ["category", "int"],
    "urine_protein":        ["category", "int"],
    "height":               ["numeric", "int"],
    "weight":               ["numeric", "int"],
    "waist_circumference":  ["numeric", "float"],
    "systolic_bp":          ["numeric", "int"],
    "diastolic_bp":         ["numeric", "int"],
    "vision_left":          ["numeric", "float"],
    "vision_right":         ["numeric", "float"],
    "total_cholesterol":    ["numeric", "int"],
    "triglycerides":        ["numeric", "int"],
    "hdl_cholesterol":      ["numeric", "int"],
    "ldl_cholesterol":      ["numeric", "int"],
    "hemoglobin":           ["numeric", "float"],
    "serum_creatinine":     ["numeric", "float"],
    "serum_got_ast":        ["numeric", "int"],
    "serum_gpt_alt":        ["numeric", "int"],
    "gamma_gtp":            ["numeric", "int"],
    "fpg":                  ["numeric", "int"],
}

COLUMN_DOMAINS: dict[str, list | set] = {
    "sex_code":             {1, 2},
    "city_code":            {11, 26, 27, 28, 29, 30, 31, 36, 41, 42, 43, 44, 45, 46, 47},
    "hearing_left":         {1, 2},
    "hearing_right":        {1, 2},
    "alcohol_consumption":  {0, 1},
    "oral_exam":            {0, 1},
    "caries_presence":      {0, 1},
    "plaque_presence":      {0, 1},
    "age_group_code":       set(range(5, 19)),
    "smoking_status":       {1, 2, 3},
    "urine_protein":        set(range(1, 7)),
    "height":               [100, 230],
    "weight":               [20, 250],
    "waist_circumference":  [30, 250],
    "systolic_bp":          [60, 300],
    "diastolic_bp":         [30, 200],
    "vision_left":          [0.1, 2.5],
    "vision_right":         [0.1, 2.5],
    "total_cholesterol":    [50, 1000],
    "triglycerides":        [10, 5000],
    "hdl_cholesterol":      [5, 200],
    "ldl_cholesterol":      [5, 700],
    "hemoglobin":           [3, 25],
    "serum_creatinine":     [0.1, 30],
    "serum_got_ast":        [1, 5000],
    "serum_gpt_alt":        [1, 5000],
    "gamma_gtp":            [1, 5000],
    "fpg":                  [30, 1000],
}

FEATURES_MISSING_ALLOWED: dict[str, list] = {
    "vision_left":      ["blind_left", 1],
    "vision_right":     ["blind_right", 1],
    "caries_presence":  ["oral_exam", 0],
    "plaque_presence":  ["oral_exam", 0],
}

FEATURES_NUMERIC: list[str] = [
    col for col, (col_type, _) in COLUMN_TYPES.items()
    if col_type == "numeric" and col != "fpg"
]
FEATURES_CATEGORICAL: list[str] = [
    col for col, (col_type, _) in COLUMN_TYPES.items()
    if col_type == "category"
]
FEATURES_ALL: list[str] = FEATURES_NUMERIC + FEATURES_CATEGORICAL
TARGET_NUM: str = "fpg"
TARGET_CAT: str = "has_diabetes"
COLUMNS_ALL: list[str] = FEATURES_ALL + [TARGET_NUM]

DIABETES_THRESHOLD: int = 126
PREDIABETES_THRESHOLD: int = 100

# Target recall (sensitivity) used on the validation set to pick the
# operating threshold for positive screens. See train.choose_threshold.
TARGET_RECALL: float = 0.85

# ---------------------------------------------------------------------------
# Model feature sets
# ---------------------------------------------------------------------------
CORE_NUMERIC_FEATURES: list[str] = [
    "age_years_mid", "height", "weight", "waist_circumference",
    "bmi", "waist_to_height", "wwi",
    "systolic_bp", "diastolic_bp", "pulse_pressure", "mean_arterial_pressure",
    "urine_protein",
]

CORE_NOMINAL_FEATURES: list[str] = [
    "sex_code", "smoking_status", "alcohol_consumption",
]

NONLIPID_LAB_FEATURES: list[str] = [
    "hemoglobin", "serum_creatinine",
    "log_ast", "log_alt", "log_ggt", "ast_alt_ratio",
]

NONLIPID_MISSING_FLAGS: list[str] = [
    "missing_hemoglobin", "missing_serum_creatinine",
    "missing_serum_got_ast", "missing_serum_gpt_alt", "missing_gamma_gtp",
]

LIPID_FEATURES: list[str] = [
    "total_cholesterol", "hdl_cholesterol", "ldl_cholesterol",
    "log_triglycerides", "non_hdl_cholesterol",
    "tg_hdl_ratio", "tc_hdl_ratio", "ldl_hdl_ratio",
    "has_complete_lipid_panel", "n_lipids_available",
]

LIPID_MISSING_FLAGS: list[str] = [
    "missing_total_cholesterol", "missing_triglycerides",
    "missing_hdl_cholesterol", "missing_ldl_cholesterol",
]

# The full feature set used by the production model
MODEL_FEATURES: list[str] = (
    CORE_NUMERIC_FEATURES
    + CORE_NOMINAL_FEATURES
    + NONLIPID_LAB_FEATURES
    + NONLIPID_MISSING_FLAGS
    + LIPID_FEATURES
    + LIPID_MISSING_FLAGS
)

MODEL_NOMINAL_FEATURES: list[str] = CORE_NOMINAL_FEATURES  # CatBoost cat_features

# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------
RISK_TIER_LABELS: tuple[str, ...] = ("Low", "Moderate", "High", "Very High")
RISK_TIER_QUANTILES: tuple[float, ...] = (0.0, 0.50, 0.80, 0.95, 1.0)

RISK_TIER_RECOMMENDATIONS: dict[str, str] = {
    "Low": (
        "Your screening result is reassuring. Maintain a healthy lifestyle "
        "and continue your regular health check-ups."
    ),
    "Moderate": (
        "Your result suggests a moderate level of risk factors associated with "
        "undiagnosed type 2 diabetes. Consider discussing your results with a "
        "healthcare provider at your next scheduled visit."
    ),
    "High": (
        "Your result indicates elevated risk factors. We recommend scheduling "
        "a consultation with your healthcare provider. A fasting plasma glucose "
        "or HbA1c test may be appropriate."
    ),
    "Very High": (
        "Your result places you in the highest-risk group for this screening tool. "
        "Please consult a healthcare provider promptly. A fasting plasma glucose "
        "or HbA1c test is strongly recommended to rule out undiagnosed type 2 diabetes."
    ),
}

RISK_TIER_COLORS: dict[str, str] = {
    "Low": "#22c55e",
    "Moderate": "#eab308",
    "High": "#f97316",
    "Very High": "#ef4444",
}

# ---------------------------------------------------------------------------
# Human-readable feature display names (for SHAP charts)
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "age_years_mid":            "Age (years)",
    "height":                   "Height (cm)",
    "weight":                   "Weight (kg)",
    "waist_circumference":      "Waist circumference (cm)",
    "bmi":                      "BMI",
    "waist_to_height":          "Waist-to-height ratio",
    "wwi":                      "Weight-adjusted waist index",
    "systolic_bp":              "Systolic blood pressure (mmHg)",
    "diastolic_bp":             "Diastolic blood pressure (mmHg)",
    "pulse_pressure":           "Pulse pressure (mmHg)",
    "mean_arterial_pressure":   "Mean arterial pressure (mmHg)",
    "urine_protein":            "Urine protein (grade 1–6)",
    "sex_code":                 "Sex",
    "smoking_status":           "Smoking status",
    "alcohol_consumption":      "Alcohol consumption",
    "hemoglobin":               "Hemoglobin (g/dL)",
    "serum_creatinine":         "Serum creatinine (mg/dL)",
    "log_ast":                  "AST — log scale",
    "log_alt":                  "ALT — log scale",
    "log_ggt":                  "GGT — log scale",
    "ast_alt_ratio":            "AST/ALT ratio",
    "missing_hemoglobin":       "Hemoglobin not recorded",
    "missing_serum_creatinine": "Creatinine not recorded",
    "missing_serum_got_ast":    "AST not recorded",
    "missing_serum_gpt_alt":    "ALT not recorded",
    "missing_gamma_gtp":        "GGT not recorded",
    "total_cholesterol":        "Total cholesterol (mg/dL)",
    "hdl_cholesterol":          "HDL cholesterol (mg/dL)",
    "ldl_cholesterol":          "LDL cholesterol (mg/dL)",
    "log_triglycerides":        "Triglycerides — log scale",
    "non_hdl_cholesterol":      "Non-HDL cholesterol (mg/dL)",
    "tg_hdl_ratio":             "Triglycerides / HDL ratio",
    "tc_hdl_ratio":             "Total cholesterol / HDL ratio",
    "ldl_hdl_ratio":            "LDL / HDL ratio",
    "has_complete_lipid_panel": "Complete lipid panel available",
    "n_lipids_available":       "Number of lipid values available",
    "missing_total_cholesterol":"Total cholesterol not recorded",
    "missing_triglycerides":    "Triglycerides not recorded",
    "missing_hdl_cholesterol":  "HDL not recorded",
    "missing_ldl_cholesterol":  "LDL not recorded",
}
