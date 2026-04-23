# Model Card — T2D Risk Screener (CatBoost)

This card documents the committed model bundle in `models/`: `catboost_model.cbm`, `calibrator.pkl`, `metadata.json`, and `global_shap.json`. These artefacts are the single source of truth for all metrics quoted on the portfolio deck and in the FastAPI service.

## Model Overview

- **Task.** Binary classification — probability that a patient has undiagnosed Type 2 Diabetes, defined as fasting plasma glucose (FPG) >= 126 mg/dL at their annual health examination.
- **Algorithm.** CatBoost — gradient-boosted **oblivious** decision trees (symmetric splits across each level), introduced by Prokhorenkova et al., 2018 (*CatBoost: unbiased boosting with categorical features*, NeurIPS 2018). Oblivious trees give fast, regularised inference and exact per-prediction SHAP values in closed form.
- **Output head.** Raw CatBoost sigmoid probability -> Platt scaler (logistic regression on validation-set logits) -> calibrated probability in [0, 1].
- **Input.** 40 engineered features derived from routine health-examination fields available in the Korean NHIS dataset; 37 numeric, 3 categorical (sex, smoking, alcohol).

## Architecture and Hyperparameters

Values are loaded in `train.py` and preserved in `models/metadata.json` after a training run. The committed model was fit with these settings:

| Hyperparameter | Value |
|---|---|
| Loss function | `Logloss` |
| Evaluation metric | `AUC` |
| Iterations (max) | 1500 |
| Early stopping rounds | 100 |
| Best iteration used | 1324 |
| Learning rate | 0.03 |
| Tree depth | 6 |
| L2 leaf regularisation | 5.0 |
| Class weights | `[1.0, neg / pos]` on the training split |
| Categorical features | `sex_code`, `smoking_status`, `alcohol_consumption` |
| Random seed | 42 (numpy, random, CatBoost) |
| Platt calibrator | `sklearn.linear_model.LogisticRegression(random_state=42)` fit on validation logits |

## Training Data

- **Source.** [Korean National Health Insurance Service (NHIS) General Health Examination 2024](https://www.data.go.kr/en/data/15007122/fileData.do) — a publicly released random sample of approximately 1 million adult subscribers who received a health check-up in 2024.
- **Label.** Binary `has_diabetes = 1` if FPG >= 126 mg/dL, else `0`. Roughly **7.9%** of the cohort is positive, a **~12:1 negative:positive** class imbalance.
- **Splits.** Stratified 60 / 20 / 20 train / validation / test. Seed fixed at 42. Positive rate is preserved across splits.
- **Feature engineering.** Derived features include BMI, waist-to-height ratio, weight-adjusted waist index (WWI), pulse pressure, mean arterial pressure, AST/ALT ratio, log-transforms of right-skewed lab values (AST, ALT, GGT, triglycerides), non-HDL cholesterol, and lipid ratios (TG/HDL, TC/HDL, LDL/HDL). Binary missingness flags are added for each optional lab so the model can learn from absence as well as value.

### Data leakage avoided — FPG is the label, never a feature

Fasting plasma glucose is the quantity used to construct `has_diabetes`. It is **excluded from the model feature set** and does not appear in `MODEL_FEATURES` in `src/config.py`. Including FPG as an input would make the classifier trivially circular: it would simply be re-thresholding the label. The stated goal of the screener is to flag patients who would benefit from an FPG or HbA1c test, so training on indirect metabolic indicators (body composition, blood pressure, kidney markers, liver enzymes, lipids) is the entire point.

## Calibration

Raw CatBoost sigmoid outputs are decently ranked (AUC 0.816) but not well calibrated — at low probabilities the model is over-confident relative to the 7.9% base rate. Platt scaling, a one-parameter-plus-intercept logistic regression fit on validation-set logits, corrects this without touching the model's ranking. The reliability-diagram comparison (raw vs. Platt-calibrated) is produced by `notebooks/tutorial.ipynb` and saved to `figures/calibration_pre_post.png`.

## Test Metrics

From `models/metadata.json` (held-out 20% test slice, n ~ 200k):

| Metric | Value |
|---|---|
| ROC-AUC | **0.8164** |
| PR-AUC (Average Precision) | **0.2734** |
| Brier score | **0.0643** |
| Operating threshold | **0.0659** |
| Sensitivity @ threshold (validation) | >= 85% (target) |

The operating threshold was tuned on the validation set to be the lowest threshold achieving at least **85% sensitivity**, then held fixed for test-set evaluation. This reflects the asymmetric cost of a missed case (a patient who never gets an FPG test ordered) versus a false positive (one extra blood draw).

## Model Comparison

Six candidate models were evaluated on the same held-out test split (all plotted on `figures/roc_all_models.png`, reproduced in the exploratory `all_models_roc.ipynb` notebook in the original source repo):

| Model | ROC-AUC |
|---|---|
| **CatBoost (full feature set) — committed model** | **0.817** |
| PyTorch Tabular Net | 0.816 |
| sklearn MLPClassifier | 0.812 |
| Logistic Regression | 0.792 |
| Support Vector Machine (RBF) | 0.792 |
| Decision Tree | 0.776 |

CatBoost and the PyTorch Tabular Net are effectively tied on ROC-AUC. **CatBoost was chosen over the tied deep network because it supports native, fast, exact per-prediction SHAP values**, which is a product requirement for a clinical screening tool: every positive screen must be accompanied by a transparent explanation of which inputs drove the decision. Computing exact SHAP for the Tabular Net would require a separate surrogate (KernelSHAP) that is slower and approximate. CatBoost also runs on CPU at negligible cost, which removes the GPU dependency from deployment.

**Reproducing figures.** The three committed model figures are regenerated by `scripts/generate_shap_summary.py`, `scripts/generate_confusion_matrix.py`, and `scripts/generate_roc_comparison.py` (see the *Regenerating figures* section of `README.md`).

## Intended Use

- Flagging patients whose routine NHIS-style examination values suggest a non-trivial probability of undiagnosed T2D, so a confirmatory FPG or HbA1c test can be ordered.
- Educational and portfolio demonstration of the full end-to-end path: public health dataset -> feature engineering -> calibrated probabilistic classifier -> per-prediction explanation -> browser-accessible screening tool.

## Out of Scope

- **Not a diagnostic tool.** Diagnosis requires FPG, HbA1c, or OGTT per clinical guidelines.
- **Not clinically validated.** No prospective validation has been conducted.
- **Not evaluated on non-Korean populations.** Features and ranges are from Korean NHIS check-ups.

## Limitations

- **Single cohort, single year.** All training data comes from the 2024 Korean NHIS random sample. Temporal drift and population drift are not assessed.
- **Label noise.** FPG >= 126 mg/dL identifies likely T2D at one point in time; it will misclassify some patients with poorly controlled known diabetes as "undiagnosed" (NHIS does not ship treatment history in the public release).
- **Screening-exam features only.** The model can only see what a patient's annual exam measures. Patients who have been skipping their annual exam entirely are out of reach of the tool.
- **Selection bias in optional labs.** Lipid and liver-enzyme panels were ordered at clinician discretion. Patients with missing labs are on average healthier than those tested, so the model treats absence as a mild "healthy" signal. Predictions made with zero laboratory inputs may therefore underestimate risk. The frontend surfaces this caveat when a user submits with all lab fields blank.
- **Coarse age encoding.** NHIS releases age only as 5-year bands; the model uses the band midpoint.

## Author, License, Date

- **Author.** Tyson Johnson — B.S. Computational and Data Sciences, George Mason University (solo project, CDS 403, spring 2026).
- **License.** MIT (see `LICENSE` in the repo root once added).
- **Last updated.** April 2026.
