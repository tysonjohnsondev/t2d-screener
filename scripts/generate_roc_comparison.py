"""
generate_roc_comparison.py
--------------------------
End-to-end 6-model bake-off that produces ``figures/roc_all_models.png``.

Extracted from the exploratory notebook ``all_models_roc.ipynb`` in the
original source repo. Downloads the NHIS dataset, engineers the same
features used in training, creates a stratified 60/20/20 split (seed 42),
trains all models on that shared split, and saves:

  - ``figures/roc_all_models.png``            -- combined ROC chart
  - ``scripts/roc_comparison_data.json``      -- raw ROC curves + AUCs

Models:
  1. CatBoost (Core Only)
  2. CatBoost (Core + Labs)
  3. CatBoost (Full Model)
  4. sklearn MLPClassifier (ANN)
  5. PyTorch Tabular Net        (skipped gracefully if torch is missing)
  6. Logistic Regression
  7. LinearSVC                  (plotted as "SVM")
  8. Decision Tree

Hyperparameters mirror the notebook exactly. Runtime is roughly 5-10 min
on CPU; the PyTorch Tabular Net dominates wall time.

Run from the project root::

    python scripts/generate_roc_comparison.py
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (  # noqa: E402
    CORE_NOMINAL_FEATURES,
    CORE_NUMERIC_FEATURES,
    LIPID_FEATURES,
    LIPID_MISSING_FLAGS,
    NONLIPID_LAB_FEATURES,
    NONLIPID_MISSING_FLAGS,
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
FIGURES_DIR: Path = ROOT_DIR / "figures"
OUTPUT_PNG: Path = FIGURES_DIR / "roc_all_models.png"
OUTPUT_JSON: Path = Path(__file__).resolve().parent / "roc_comparison_data.json"

FEATURE_SETS: dict[str, dict[str, list[str]]] = {
    "core_screening": {
        "features": CORE_NUMERIC_FEATURES + CORE_NOMINAL_FEATURES,
        "nominal_features": CORE_NOMINAL_FEATURES,
    },
    "core_plus_labs": {
        "features": (
            CORE_NUMERIC_FEATURES
            + CORE_NOMINAL_FEATURES
            + NONLIPID_LAB_FEATURES
            + NONLIPID_MISSING_FLAGS
        ),
        "nominal_features": CORE_NOMINAL_FEATURES,
    },
    "core_plus_labs_plus_lipids": {
        "features": (
            CORE_NUMERIC_FEATURES
            + CORE_NOMINAL_FEATURES
            + NONLIPID_LAB_FEATURES
            + NONLIPID_MISSING_FLAGS
            + LIPID_FEATURES
            + LIPID_MISSING_FLAGS
        ),
        "nominal_features": CORE_NOMINAL_FEATURES,
    },
}


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy, and (if available) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Calibration helpers (shared with train.py)
# ---------------------------------------------------------------------------

def prob_to_logit(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Map probabilities in (0, 1) to logits with clipping."""
    prob = np.clip(np.asarray(prob, float), eps, 1 - eps)
    return np.log(prob / (1 - prob))


def fit_calibrator(raw_prob: np.ndarray, y_true: pd.Series) -> LogisticRegression:
    """Fit a Platt calibrator (LR on logits) on validation-set scores."""
    cal = LogisticRegression(random_state=SEED)
    cal.fit(prob_to_logit(raw_prob).reshape(-1, 1), y_true)
    return cal


def apply_calibrator(cal: LogisticRegression, raw_prob: np.ndarray) -> np.ndarray:
    """Apply a fitted Platt calibrator to raw probabilities."""
    return cal.predict_proba(prob_to_logit(raw_prob).reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_splits(
    all_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[int, float], list[float]]:
    """Load data, engineer features, build the shared stratified 60/20/20 split."""
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

    available = [f for f in all_features if f in df.columns]
    df_model = df[available + [TARGET_CAT]].copy()
    df_model = df_model[df_model[TARGET_CAT].notna()].reset_index(drop=True)

    y = df_model[TARGET_CAT].astype(int)
    X = df_model[available].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train
    )

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    cw_dict = {0: 1.0, 1: neg / max(pos, 1)}
    cw_list = [1.0, neg / max(pos, 1)]

    return X_train, X_valid, X_test, y_train, y_valid, y_test, cw_dict, cw_list


# ---------------------------------------------------------------------------
# CatBoost trio
# ---------------------------------------------------------------------------

def train_catboost_variants(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    cw_list: list[float],
    roc_data: dict[str, dict[str, Any]],
) -> None:
    """Train the three CatBoost variants and append their ROC curves to ``roc_data``."""
    cb_sets = {
        "CatBoost (Core Only)":   FEATURE_SETS["core_screening"],
        "CatBoost (Core + Labs)": FEATURE_SETS["core_plus_labs"],
        "CatBoost (Full Model)":  FEATURE_SETS["core_plus_labs_plus_lipids"],
    }
    for name, spec in cb_sets.items():
        feats = [f for f in spec["features"] if f in X_train.columns]
        print(f"\n{'=' * 60}\n{name}  ({len(feats)} features)\n{'=' * 60}")

        Xtr, Xv, Xte, cat_idx = prepare_catboost_inputs(
            X_train[feats], X_valid[feats], X_test[feats],
            nominal_features=spec["nominal_features"],
        )
        m = CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC",
            iterations=1500, learning_rate=0.03,
            depth=6, l2_leaf_reg=5.0,
            random_seed=SEED, verbose=500,
            allow_writing_files=False, class_weights=cw_list,
        )
        m.fit(
            Xtr, y_train, cat_features=cat_idx,
            eval_set=(Xv, y_valid),
            use_best_model=True, early_stopping_rounds=100,
        )

        raw_valid = m.predict_proba(Xv)[:, 1]
        raw_test = m.predict_proba(Xte)[:, 1]
        cal = fit_calibrator(raw_valid, y_valid)
        cal_test = apply_calibrator(cal, raw_test)

        auc = roc_auc_score(y_test, cal_test)
        fpr, tpr, _ = roc_curve(y_test, cal_test)
        roc_data[name] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(float(auc), 3),
        }
        print(f"  => ROC-AUC = {auc:.4f}")


# ---------------------------------------------------------------------------
# ANN (MLPClassifier) path
# ---------------------------------------------------------------------------

def preprocess_ann_inputs(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    nominal_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Median-impute, standard-scale, one-hot-encode for sklearn dense models."""
    X_train, X_valid, X_test = X_train.copy(), X_valid.copy(), X_test.copy()
    nom = [c for c in nominal_features if c in X_train.columns]
    num = [c for c in X_train.columns if c not in nom]

    Xtr_n = X_train[num].apply(pd.to_numeric, errors="coerce").astype(float)
    Xv_n = X_valid[num].apply(pd.to_numeric, errors="coerce").astype(float)
    Xte_n = X_test[num].apply(pd.to_numeric, errors="coerce").astype(float)

    meds = Xtr_n.median().fillna(0.0)
    Xtr_n, Xv_n, Xte_n = Xtr_n.fillna(meds), Xv_n.fillna(meds), Xte_n.fillna(meds)

    scaler = StandardScaler()
    Xtr_n = pd.DataFrame(scaler.fit_transform(Xtr_n), columns=num, index=X_train.index)
    Xv_n = pd.DataFrame(scaler.transform(Xv_n), columns=num, index=X_valid.index)
    Xte_n = pd.DataFrame(scaler.transform(Xte_n), columns=num, index=X_test.index)

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        if not nom:
            return pd.DataFrame(index=frame.index)
        out = frame[nom].copy()
        for c in nom:
            out[c] = out[c].astype("string").fillna("Missing")
        return pd.get_dummies(out, columns=nom, drop_first=False, dtype=float)

    Xtr_c = encode(X_train)
    Xv_c = encode(X_valid).reindex(columns=Xtr_c.columns, fill_value=0.0)
    Xte_c = encode(X_test).reindex(columns=Xtr_c.columns, fill_value=0.0)

    return (
        pd.concat([Xtr_n, Xtr_c], axis=1),
        pd.concat([Xv_n, Xv_c], axis=1),
        pd.concat([Xte_n, Xte_c], axis=1),
    )


def rebalance_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_pos_rate: float = 0.30,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Upsample the positive class so ``target_pos_rate`` is reached."""
    data = X_train.copy()
    data["_t"] = y_train.to_numpy()
    pos, neg = data[data["_t"] == 1], data[data["_t"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return X_train.copy(), y_train.copy()
    if len(pos) / len(data) >= target_pos_rate:
        return X_train.copy(), y_train.copy()

    n_pos = max(
        int(target_pos_rate * len(neg) / (1 - target_pos_rate)),
        len(pos),
    )
    pos_up = resample(pos, replace=True, n_samples=n_pos, random_state=seed)
    balanced = (
        pd.concat([neg, pos_up])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return balanced.drop(columns="_t"), balanced["_t"].astype(int)


def train_ann(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    roc_data: dict[str, dict[str, Any]],
) -> None:
    """Train the sklearn MLPClassifier ANN on the full feature set."""
    spec = FEATURE_SETS["core_plus_labs_plus_lipids"]
    feats = [f for f in spec["features"] if f in X_train.columns]
    nom = spec["nominal_features"]

    print(f"\nTraining sklearn ANN on {len(feats)} features ...")

    Xtr_a, Xv_a, Xte_a = preprocess_ann_inputs(
        X_train[feats], X_valid[feats], X_test[feats], nominal_features=nom,
    )
    Xtr_bal, ytr_bal = rebalance_training_data(
        Xtr_a, y_train, target_pos_rate=0.30, seed=SEED,
    )

    ann = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=0.001, batch_size=2048, learning_rate_init=0.001,
        max_iter=60, early_stopping=True, n_iter_no_change=5,
        random_state=SEED,
    )
    ann.fit(Xtr_bal, ytr_bal)

    raw_valid = ann.predict_proba(Xv_a)[:, 1]
    raw_test = ann.predict_proba(Xte_a)[:, 1]
    cal = fit_calibrator(raw_valid, y_valid)
    cal_test = apply_calibrator(cal, raw_test)

    auc = roc_auc_score(y_test, cal_test)
    fpr, tpr, _ = roc_curve(y_test, cal_test)
    roc_data["ANN (MLPClassifier)"] = {
        "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(float(auc), 3),
    }
    print(f"  => ROC-AUC = {auc:.4f}")


# ---------------------------------------------------------------------------
# PyTorch Tabular Net path (optional)
# ---------------------------------------------------------------------------

def train_torch_tabular(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    roc_data: dict[str, dict[str, Any]],
) -> bool:
    """Train the embedding-based PyTorch Tabular Net. Returns False if torch is missing."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("\nPyTorch not available — skipping Torch Tabular Net.")
        return False

    spec = FEATURE_SETS["core_plus_labs_plus_lipids"]
    feats = [f for f in spec["features"] if f in X_train.columns]
    nom = spec["nominal_features"]

    print(f"\nTraining PyTorch Tabular Net on {len(feats)} features ...")

    cats = [c for c in nom if c in X_train.columns]
    nums = [c for c in feats if c not in cats]

    Xtr_n = X_train[nums].apply(pd.to_numeric, errors="coerce").astype(float)
    Xv_n = X_valid[nums].apply(pd.to_numeric, errors="coerce").astype(float)
    Xte_n = X_test[nums].apply(pd.to_numeric, errors="coerce").astype(float)
    meds = Xtr_n.median().fillna(0.0)
    Xtr_n, Xv_n, Xte_n = Xtr_n.fillna(meds), Xv_n.fillna(meds), Xte_n.fillna(meds)

    scaler = StandardScaler()
    Xtr_n = scaler.fit_transform(Xtr_n).astype(np.float32)
    Xv_n = scaler.transform(Xv_n).astype(np.float32)
    Xte_n = scaler.transform(Xte_n).astype(np.float32)

    mappings: dict[str, dict[str, int]] = {}
    Xtr_cat, Xv_cat, Xte_cat = X_train.copy(), X_valid.copy(), X_test.copy()
    for frame in (Xtr_cat, Xv_cat, Xte_cat):
        for c in cats:
            frame[c] = frame[c].astype("string").fillna("Missing")
    for c in cats:
        vals = Xtr_cat[c].unique().tolist()
        mappings[c] = {v: i + 1 for i, v in enumerate(vals)}

    def enc(frame: pd.DataFrame) -> np.ndarray:
        if not cats:
            return np.zeros((len(frame), 0), dtype=np.int64)
        cols = [
            frame[c].map(mappings[c]).fillna(0).astype(int).to_numpy() for c in cats
        ]
        return np.column_stack(cols).astype(np.int64)

    Xtr_c = enc(Xtr_cat)
    Xv_c = enc(Xv_cat)
    Xte_c = enc(Xte_cat)
    cards = [max(mappings[c].values(), default=0) + 1 for c in cats]

    class TorchTabularDataset(Dataset):
        def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray | None = None):
            self.x_num = torch.tensor(x_num, dtype=torch.float32)
            self.x_cat = torch.tensor(x_cat, dtype=torch.long)
            self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.x_num)

        def __getitem__(self, idx: int):
            if self.y is None:
                return self.x_num[idx], self.x_cat[idx]
            return self.x_num[idx], self.x_cat[idx], self.y[idx]

    def embedding_dim_for_cardinality(c: int) -> int:
        if c <= 2:
            return 1
        return min(32, max(2, int(round(math.sqrt(c)))))

    class TorchTabularNet(nn.Module):
        def __init__(
            self,
            num_numeric: int,
            cat_cards: list[int],
            hidden_dims: tuple[int, ...] = (128, 64),
            dropout: float = 0.20,
        ):
            super().__init__()
            self.embs = nn.ModuleList()
            emb_out = 0
            for card in cat_cards:
                d = embedding_dim_for_cardinality(card)
                self.embs.append(nn.Embedding(card, d))
                emb_out += d
            layers: list[nn.Module] = []
            prev = num_numeric + emb_out
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
            parts: list[torch.Tensor] = [x_num]
            if len(self.embs) > 0:
                parts.append(
                    torch.cat(
                        [e(x_cat[:, i]) for i, e in enumerate(self.embs)], dim=1
                    )
                )
            return self.mlp(torch.cat(parts, dim=1)).squeeze(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ytr_np = y_train.to_numpy().astype(np.float32)
    yv_np = y_valid.to_numpy().astype(np.float32)

    train_ds = TorchTabularDataset(Xtr_n, Xtr_c, ytr_np)
    valid_ds = TorchTabularDataset(Xv_n, Xv_c, yv_np)
    test_ds = TorchTabularDataset(Xte_n, Xte_c)

    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=4096, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

    net = TorchTabularNet(
        num_numeric=Xtr_n.shape[1],
        cat_cards=cards,
        hidden_dims=(128, 64),
        dropout=0.20,
    ).to(device)

    n_pos = max(float(ytr_np.sum()), 1.0)
    n_neg = max(float(len(ytr_np) - ytr_np.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    def predict_prob(loader: DataLoader) -> np.ndarray:
        net.eval()
        out: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                xn = batch[0].to(device)
                xc = batch[1].to(device)
                out.append(torch.sigmoid(net(xn, xc)).cpu().numpy())
        return np.concatenate(out)

    best_state: dict[str, torch.Tensor] | None = None
    best_ap: float = -float("inf")
    patience_ctr: int = 0

    for epoch in range(20):
        net.train()
        eloss = 0.0
        for xn, xc, yb in train_loader:
            xn, xc, yb = xn.to(device), xc.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(net(xn, xc), yb)
            loss.backward()
            optimizer.step()
            eloss += loss.item() * len(yb)
        tl = eloss / len(train_ds)
        vp = predict_prob(valid_loader)
        vap = average_precision_score(y_valid, vp)
        vauc = roc_auc_score(y_valid, vp)
        print(
            f"  Epoch {epoch+1:02d} | train_loss={tl:.4f} "
            f"| valid_ap={vap:.4f} | valid_auc={vauc:.4f}"
        )
        if vap > best_ap + 1e-6:
            best_ap = vap
            best_state = copy.deepcopy(net.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= 4:
            print("  Early stopping.")
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    raw_valid = predict_prob(valid_loader)
    raw_test = predict_prob(test_loader)
    cal = fit_calibrator(raw_valid, y_valid)
    cal_test = apply_calibrator(cal, raw_test)

    auc = roc_auc_score(y_test, cal_test)
    fpr, tpr, _ = roc_curve(y_test, cal_test)
    roc_data["Torch Tabular Net"] = {
        "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(float(auc), 3),
    }
    print(f"  => ROC-AUC = {auc:.4f}")
    return True


# ---------------------------------------------------------------------------
# LR / SVM / DT
# ---------------------------------------------------------------------------

def train_sklearn_baselines(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cw_dict: dict[int, float],
    all_features: list[str],
    roc_data: dict[str, dict[str, Any]],
) -> None:
    """Train Logistic Regression, LinearSVC, and a Decision Tree."""
    medians = X_train[all_features].median()
    X_train_sk = X_train[all_features].fillna(medians)
    X_test_sk = X_test[all_features].fillna(medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sk)
    X_test_scaled = scaler.transform(X_test_sk)

    print("\nTraining Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight=cw_dict)
    lr.fit(X_train_scaled, y_train)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_prob)
    fpr, tpr, _ = roc_curve(y_test, lr_prob)
    roc_data["Logistic Regression"] = {
        "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(float(lr_auc), 3),
    }
    print(f"  => ROC-AUC = {lr_auc:.4f}")

    print("Training SVM (LinearSVC) ...")
    svm = LinearSVC(random_state=SEED, class_weight="balanced", max_iter=10000)
    svm.fit(X_train_scaled, y_train)
    svm_scores = svm.decision_function(X_test_scaled)
    svm_auc = roc_auc_score(y_test, svm_scores)
    fpr, tpr, _ = roc_curve(y_test, svm_scores)
    roc_data["SVM"] = {
        "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(float(svm_auc), 3),
    }
    print(f"  => ROC-AUC = {svm_auc:.4f}")

    print("Training Decision Tree ...")
    dt = DecisionTreeClassifier(max_depth=10, random_state=SEED, class_weight=cw_dict)
    dt.fit(X_train_sk, y_train)
    dt_prob = dt.predict_proba(X_test_sk.values)[:, 1]
    dt_auc = roc_auc_score(y_test, dt_prob)
    fpr, tpr, _ = roc_curve(y_test, dt_prob)
    roc_data["Decision Tree"] = {
        "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(float(dt_auc), 3),
    }
    print(f"  => ROC-AUC = {dt_auc:.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_combined_roc(
    roc_data: dict[str, dict[str, Any]],
    output_path: Path,
    torch_available: bool,
) -> None:
    """Save a combined ROC figure matching the source notebook's styling."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {
        "CatBoost (Full Model)":  "#1a5276",
        "CatBoost (Core + Labs)": "#2e86c1",
        "CatBoost (Core Only)":   "#85c1e9",
        "ANN (MLPClassifier)":    "#8e44ad",
        "Torch Tabular Net":      "#d35400",
        "Logistic Regression":    "#e74c3c",
        "SVM":                    "#27ae60",
        "Decision Tree":          "#f39c12",
    }
    linestyles = {
        "CatBoost (Full Model)":  "-",
        "CatBoost (Core + Labs)": "-",
        "CatBoost (Core Only)":   "-",
        "ANN (MLPClassifier)":    "--",
        "Torch Tabular Net":      "--",
        "Logistic Regression":    "-.",
        "SVM":                    "-.",
        "Decision Tree":          ":",
    }
    plot_order = [
        "CatBoost (Full Model)",
        "CatBoost (Core + Labs)",
        "Torch Tabular Net",
        "ANN (MLPClassifier)",
        "SVM",
        "Logistic Regression",
        "CatBoost (Core Only)",
        "Decision Tree",
    ]

    for name in plot_order:
        if name not in roc_data:
            continue
        d = roc_data[name]
        ax.plot(
            d["fpr"], d["tpr"],
            color=colors.get(name, "#333"),
            linestyle=linestyles.get(name, "-"),
            linewidth=2.0,
            label=f"{name} (AUC = {d['auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    title = "ROC Curve Comparison \u2014 All Models"
    if not torch_available:
        title += "\n(PyTorch Tabular Net skipped: torch not installed)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full 6-model bake-off and regenerate ``figures/roc_all_models.png``.

    Builds a shared stratified 60/20/20 split, trains each candidate model on
    the same data, writes per-model ROC curves to
    ``scripts/roc_comparison_data.json``, and saves the combined ROC chart.

    The PyTorch Tabular Net is skipped gracefully if ``torch`` is not
    installed; the chart title is annotated accordingly.
    """
    set_seeds()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_features = list(
        dict.fromkeys(
            CORE_NUMERIC_FEATURES
            + CORE_NOMINAL_FEATURES
            + NONLIPID_LAB_FEATURES
            + NONLIPID_MISSING_FLAGS
            + LIPID_FEATURES
            + LIPID_MISSING_FLAGS
        )
    )

    print("Loading data and building shared 60/20/20 split ...")
    (
        X_train, X_valid, X_test,
        y_train, y_valid, y_test,
        cw_dict, cw_list,
    ) = build_splits(all_features)
    print(
        f"Train: {len(y_train):,}  Valid: {len(y_valid):,}  Test: {len(y_test):,}  "
        f"pos-rate={y_train.mean():.3%}"
    )

    roc_data: dict[str, dict[str, Any]] = {}

    # CatBoost trio
    train_catboost_variants(
        X_train, X_valid, X_test, y_train, y_valid, y_test, cw_list, roc_data,
    )

    # sklearn ANN (MLPClassifier)
    train_ann(X_train, X_valid, X_test, y_train, y_valid, y_test, roc_data)

    # PyTorch Tabular Net (skips if torch isn't installed)
    torch_available = train_torch_tabular(
        X_train, X_valid, X_test, y_train, y_valid, y_test, roc_data,
    )

    # LR / LinearSVC / Decision Tree on all features
    available = [f for f in all_features if f in X_train.columns]
    train_sklearn_baselines(
        X_train, X_test, y_train, y_test, cw_dict, available, roc_data,
    )

    # Persist numeric ROC curves for downstream reuse.
    OUTPUT_JSON.write_text(json.dumps(roc_data, indent=2))
    print(f"\nROC curve data saved -> {OUTPUT_JSON}")

    plot_combined_roc(roc_data, OUTPUT_PNG, torch_available=torch_available)
    print(f"Figure saved -> {OUTPUT_PNG}")

    print("\nSummary:")
    for name, d in roc_data.items():
        print(f"  {name:<30} AUC = {d['auc']:.3f}")


if __name__ == "__main__":
    main()
