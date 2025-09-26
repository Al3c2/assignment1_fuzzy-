# tsk_teacher_style.py
# ------------------------------------------------------------
# Teacher-style TSK (Takagi–Sugeno–Kang) with PyTorch + FCM.
# LOGIC UNCHANGED: FCM -> Gaussian MFs -> First-order TSK -> LS fit.
# Metrics aligned with the teacher:
#   - classification: ACC
#   - regression:     MSE
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score  # teacher metrics
from sklearn.datasets import load_diabetes, fetch_openml
import skfuzzy as fuzz

# =========================
# CONFIG (ADJUST ONLY HERE)
# =========================
TASK = "regression"   # "regression" or "classification"
DATASET = "sklearn_diabetes"   # "sklearn_diabetes" | "pima_openml" | "excel"
EXCEL_PATH = "data.xlsx"  # used if DATASET="excel"
TARGET_COL = "target"     # target column name for excel (if present)

N_CLUSTERS = 6            # number of rules/clusters (e.g., 4–8)
M_FCM = 1.6               # fuzzifier (typical 1.6–2.4)
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_STRATIFY = True       # set False if you want EXACT teacher split style (no stratify)
# =========================

# -------- utils ----------
def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def _weighted_mean_std(Xz: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-rule, per-feature centers (mu) and sigmas using FCM memberships (U^m).
    U shape: (R, N). Returns centers (R,D) and sigmas (R,D).
    """
    R, N = U.shape
    D = Xz.shape[1]
    Um = U ** M_FCM
    centers = np.zeros((R, D))
    sigmas = np.zeros((R, D))
    for r in range(R):
        w = Um[r][:, None]  # (N,1)
        mu = (w * Xz).sum(axis=0) / (w.sum(axis=0) + 1e-12)
        centers[r] = mu
        var = (w * (Xz - mu) ** 2).sum(axis=0) / (w.sum(axis=0) + 1e-12)
        sigmas[r] = np.sqrt(var + 1e-6)  # avoid zero
    return centers, sigmas

def _design_matrix(Xz: np.ndarray, centers: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """
    Build TSK (order-1) design matrix Phi. For each sample i and rule r:
      features: [ w_r(i), w_r(i)*x_i1, ..., w_r(i)*x_iD ]
    with w_r normalized across rules for sample i.
    Returns Phi (N, R*(1+D)) and w_norm (N,R).
    """
    R, D = centers.shape
    N = Xz.shape[0]

    # compute unnormalized Gaussian firing strengths in log space for stability
    log_w = []
    for r in range(R):
        z = (Xz - centers[r]) / (sigmas[r] + 1e-12)  # (N,D)
        log_phi = -0.5 * (z ** 2).sum(axis=1)        # (N,)
        log_w.append(log_phi)
    log_w = np.stack(log_w, axis=1)  # (N,R)

    # normalize per sample (softmax-like)
    maxlog = np.max(log_w, axis=1, keepdims=True)
    w = np.exp(log_w - maxlog)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)  # (N,R)

    # design matrix
    ones = np.ones((N, 1))
    Phi_parts = []
    for r in range(R):
        wr = w[:, [r]]  # (N,1)
        Phi_r = np.hstack([wr * ones, wr * Xz])  # (N, 1+D)
        Phi_parts.append(Phi_r)
    Phi = np.hstack(Phi_parts)  # (N, R*(1+D))
    return Phi, w

# ------- TSK model -------
@dataclass
class TSKModel(nn.Module):
    centers: np.ndarray  # (R,D)
    sigmas: np.ndarray   # (R,D)
    D: int
    R: int

    def __post_init__(self):
        super().__init__()
        self.D = self.centers.shape[1]
        self.R = self.centers.shape[0]
        # consequent parameters per rule: [b_r, w_r1, ..., w_rD]
        self.theta = nn.Parameter(torch.zeros(self.R, self.D + 1), requires_grad=False)

    def forward(self, Xz: torch.Tensor):
        Phi_np, w_norm_np = _design_matrix(_to_numpy(Xz), self.centers, self.sigmas)
        Phi = torch.from_numpy(Phi_np).to(dtype=torch.float32, device=Xz.device)       # (N, R*(1+D))
        w_norm = torch.from_numpy(w_norm_np).to(dtype=torch.float32, device=Xz.device) # (N, R)
        theta_vec = self.theta.reshape(-1)  # (R*(1+D),)
        y = Phi @ theta_vec
        return y.view(-1, 1), w_norm, Phi

# ------ Least Squares ------
def train_ls(model: TSKModel, Xz: np.ndarray, y: np.ndarray, task: str):
    """
    Fit theta via LS:
      theta = (Phi^T Phi + λI)^(-1) Phi^T y
    For classification, follow teacher-style: fit LS directly on labels (0/1),
    then apply sigmoid at inference. (Keeping logic simple as requested.)
    """
    device = torch.device("cpu")
    model.to(device)
    Xz_t = torch.from_numpy(Xz.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    with torch.no_grad():
        _, _, Phi = model(Xz_t)  # (N, R*(1+D))

    lam = 1e-6
    A = Phi.T @ Phi + lam * torch.eye(Phi.shape[1])
    b = Phi.T @ y_t
    # modern PyTorch: use linalg.solve
    theta_vec = torch.linalg.solve(A, b)
    theta = theta_vec.view(model.R, model.D + 1)
    with torch.no_grad():
        model.theta.copy_(theta)

# --------- Data loaders ----------
def load_data():
    if DATASET == "sklearn_diabetes":
        # REGRESSION
        ds = load_diabetes()
        X = ds.data.astype(float)
        y = ds.target.astype(float)
        names = list(ds.feature_names)
        return X, y, names

    elif DATASET == "pima_openml":
        # CLASSIFICATION
        df = fetch_openml(name="diabetes", version=1, as_frame=True)
        X_df = df.data.copy()

        # Optional cleaning: replace impossible zeros, then median impute
        zero_bad = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for c in zero_bad:
            if c in X_df.columns:
                X_df[c] = X_df[c].replace(0, np.nan)
        X_df = X_df.fillna(X_df.median(numeric_only=True))

        # Map labels -> 0/1 (teacher dataset uses strings)
        y_ser = df.target.astype(str).str.strip().str.lower()
        y = y_ser.isin(["tested_positive", "positive", "pos", "1", "true", "yes"]).astype(int).to_numpy()

        X = X_df.to_numpy().astype(float)
        names = list(X_df.columns)
        return X, y, names

    elif DATASET == "excel":
        # Read from Excel (use TARGET_COL if present; else last column as target)
        X_df = pd.read_excel(EXCEL_PATH)
        if TARGET_COL in X_df.columns:
            y = X_df[TARGET_COL].to_numpy()
            X_df = X_df.drop(columns=[TARGET_COL])
        else:
            y = X_df.iloc[:, -1].to_numpy()
            X_df = X_df.iloc[:, :-1]
        X = X_df.to_numpy().astype(float)
        names = list(X_df.columns)
        return X, y, names

    else:
        raise ValueError("Invalid DATASET. Use 'sklearn_diabetes', 'pima_openml', or 'excel'.")

# -------------- Main --------------
def main():
    X, y, feat_names = load_data()

    # Ensure correct type by task
    if TASK == "regression":
        y = y.astype(float)
    else:
        y = (y > 0).astype(int)

    # Scale features
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)

    # Train/test split
    stratify_arg = (y if (TASK == "classification" and USE_STRATIFY) else None)
    Xtr, Xte, ytr, yte = train_test_split(
        Xz, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_arg
    )

    # FCM on training data (scaled space)
    centers, U, *_ = fuzz.cluster.cmeans(
        data=Xtr.T, c=N_CLUSTERS, m=M_FCM, error=1e-5, maxiter=300, init=None, seed=RANDOM_STATE
    )  # centers: (R,D), U: (R,Ntr)

    # Estimate per-rule sigmas (weighted by U^m)
    centers_w, sigmas_w = _weighted_mean_std(Xtr, U)  # shapes (R,D)
    centers_use = centers
    sigmas_use = sigmas_w

    # Build TSK model and fit by LS (closed-form)
    R, D = centers_use.shape
    model = TSKModel(centers=centers_use, sigmas=sigmas_use, D=D, R=R)
    train_ls(model, Xtr, ytr, TASK)

    # Evaluate
    y_pred_tr, _, _ = model(torch.from_numpy(Xtr.astype(np.float32)))
    y_pred_te, _, _ = model(torch.from_numpy(Xte.astype(np.float32)))

    if TASK == "regression":
        # Teacher metric: MSE (not RMSE)
        yhat_te = _to_numpy(y_pred_te).ravel()
        mse = mean_squared_error(yte, yhat_te)
        print(f"[REG] Test MSE: {mse:.3f}")
        # Optional extras (commented): R2 / RMSE
        # from sklearn.metrics import r2_score
        # print(f"R2: {r2_score(yte, yhat_te):.3f} | RMSE: {np.sqrt(mse):.3f}")

    else:
        # Teacher metric: ACC (threshold 0.5 on sigmoid)
        yhat_proba_te = 1 / (1 + np.exp(-_to_numpy(y_pred_te).ravel()))
        yhat_te = (yhat_proba_te >= 0.5).astype(int)
        acc = accuracy_score(yte, yhat_te)
        print(f"[CLS] Test ACC: {acc:.3f}")
        # Optional extras (commented): F1 / AUC
        # from sklearn.metrics import f1_score, roc_auc_score
        # print(f"F1: {f1_score(yte, yhat_te):.3f} | AUC: {roc_auc_score(yte, yhat_proba_te):.3f}")

    # Print rules (centers/sigmas in standardized space)
    print("\nRegras (centros/sigmas em espaço padronizado):")
    for r in range(R):
        c_txt = ", ".join([f"{feat_names[d]}≈{centers_use[r,d]:+.2f}σ" for d in range(D)])
        s_txt = ", ".join([f"σ_{feat_names[d]}={sigmas_use[r,d]:.2f}" for d in range(D)])
        print(f"- Regra {r+1}: centro[{c_txt}] | {s_txt}")

if __name__ == "__main__":
    main()
