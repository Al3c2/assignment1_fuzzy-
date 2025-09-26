# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 14:53:52 2025

@author: Alexandrehb
"""

# tsk_teacher_style.py
# ------------------------------------------------------------
# TSK (Takagi–Sugeno–Kang) com PyTorch + Fuzzy C-Means (FCM)
# Treino por Least Squares (fecho analítico), sem mexer na lógica.
# >>> SÓ ALTERAR A SECÇÃO "CONFIG" ABAIXO <<< 
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
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.datasets import load_diabetes, fetch_openml
import skfuzzy as fuzz

# =========================
# CONFIG  (SÓ ALTERAR AQUI)
# =========================
TASK = "classification"   # "regression" ou "classification"
DATASET = "pima_openml"   # "sklearn_diabetes" | "pima_openml" | "excel"
EXCEL_PATH = "data.xlsx"  # se DATASET="excel", apontar para o ficheiro
TARGET_COL = "target"     # nome da coluna target para o Excel

N_CLUSTERS = 6            # nº de regras / clusters (ex: 4–8)
M_FCM = 1.6               # fuzzifier (tipicamente 1.6–2.4)
TEST_SIZE = 0.2
RANDOM_STATE = 42
# =========================

# -------- utils ----------
def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def _weighted_mean_std(Xz: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estima média (centro) e sigma por regra/feature com pesos U^m (consistentes com FCM).
    Retorna centers (R,D) e sigmas (R,D).
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
        # var ponderada
        var = (w * (Xz - mu) ** 2).sum(axis=0) / (w.sum(axis=0) + 1e-12)
        sigmas[r] = np.sqrt(var + 1e-6)  # evitar sigma=0
    return centers, sigmas

def _design_matrix(Xz: np.ndarray, centers: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """
    Constroi Phi para TSK (ordem 1): para cada amostra i,
    concatena para cada regra r:  [w_r_normalizada(i), w_r_normalizada(i)*x(i)]
    (i.e., b_r e W_r partilham a mesma ponderação normalizada).
    Resultado: Phi shape (N, R*(1+D))
    """
    R, D = centers.shape
    N = Xz.shape[0]
    # Gaussian MFs por feature
    # w_r(i) = prod_d exp(-0.5 * ((x_id - c_rd)/sigma_rd)^2)
    # Para estabilidade, somar logs:
    exps = []
    for r in range(R):
        z = (Xz - centers[r]) / (sigmas[r] + 1e-12)  # (N,D)
        log_phi = -0.5 * (z ** 2).sum(axis=1)        # (N,)
        exps.append(log_phi)
    log_w = np.stack(exps, axis=1)  # (N,R)
    # normalizar por regra para cada amostra
    # w_norm = softmax(log_w) sem “temperatura”
    maxlog = np.max(log_w, axis=1, keepdims=True)
    w = np.exp(log_w - maxlog)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)  # (N,R) normalizado

    # Construir Phi
    Phi_parts = []
    ones = np.ones((N, 1))
    for r in range(R):
        wr = w[:, [r]]  # (N,1)
        Phi_r = np.hstack([wr * ones, wr * Xz])  # (N, 1+D)
        Phi_parts.append(Phi_r)
    Phi = np.hstack(Phi_parts)  # (N, R*(1+D))
    return Phi, w

# ------- Modelo TSK -------
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
        # Parâmetros consequentes (empilhados): para cada regra r: [b_r, w_r1, ..., w_rD]
        # Inicializa zeros; serão aprendidos por LS
        self.theta = nn.Parameter(torch.zeros(self.R, self.D + 1), requires_grad=False)

    def forward(self, Xz: torch.Tensor):
        """
        Xz: (N,D) padronizado
        Retorna:
          y_pred: (N,1)
          w_norm: (N,R) firing strengths normalizados
          Phi:    (N,R*(1+D)) design matrix usada no LS
        """
        X = Xz  # (N,D)
        N = X.shape[0]
        # computar w_norm e Phi em numpy (mais simples) e converter
        Phi_np, w_norm_np = _design_matrix(_to_numpy(X), self.centers, self.sigmas)
        Phi = torch.from_numpy(Phi_np).to(dtype=torch.float32, device=X.device)       # (N, R*(1+D))
        w_norm = torch.from_numpy(w_norm_np).to(dtype=torch.float32, device=X.device) # (N,R)

        # y = sum_r ( w_norm_r * (b_r + w_r^T x) )
        # Podemos obter y via Phi @ vec(theta)
        theta_vec = self.theta.reshape(-1)  # (R*(1+D),)
        y = Phi @ theta_vec  # (N,)
        return y.view(-1, 1), w_norm, Phi

# ------ Least Squares ------
def train_ls(model: TSKModel, Xz: np.ndarray, y: np.ndarray, task: str):
    """
    Ajusta theta por LS:
      theta = (Phi^T Phi)^(-1) Phi^T y
    Para classificação, ajusta LS no espaço do 'logit' (aproximação):
      y_tilde = log(p/(1-p))  com clipping p∈[1e-3, 1-1e-3]
    """
    device = torch.device("cpu")
    model.to(device)
    Xz_t = torch.from_numpy(Xz.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    # Para classificação: transformar rótulos (0/1) em valores-alvo contínuos via logit
    if task == "classification":
        p = y_t.clamp(1e-3, 1 - 1e-3)  # evita inf
        y_ls = torch.log(p / (1 - p))
    else:
        y_ls = y_t

    # Obter Phi
    with torch.no_grad():
        _, _, Phi = model(Xz_t)  # (N, R*(1+D))

    # Resolver LS: theta = (Phi^T Phi + λI)^(-1) Phi^T y
    lam = 1e-6
    A = Phi.T @ Phi + lam * torch.eye(Phi.shape[1])
    b = Phi.T @ y_ls
    # >>> FIX AQUI: usar torch.linalg.solve <<<
    theta_vec = torch.linalg.solve(A, b)
    theta = theta_vec.view(model.R, model.D + 1)
    with torch.no_grad():
        model.theta.copy_(theta)


# --------- Dados ----------
def load_data():
    if DATASET == "sklearn_diabetes":
        # REGRESSÃO
        ds = load_diabetes()
        X = ds.data.astype(float)
        y = ds.target.astype(float)
        names = list(ds.feature_names)
        return X, y, names

    elif DATASET == "pima_openml":
        # CLASSIFICAÇÃO
        df = fetch_openml(name="diabetes", version=1, as_frame=True)
        X_df = df.data.copy()
        # corrigir zeros impossíveis e imputar
        zero_bad = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for c in zero_bad:
            if c in X_df.columns:
                X_df[c] = X_df[c].replace(0, np.nan)
        X_df = X_df.fillna(X_df.median(numeric_only=True))
        # target string -> binário
        y_ser = df.target.astype(str).str.strip().str.lower()
        y = y_ser.isin(["tested_positive", "positive", "pos", "1", "true", "yes"]).astype(int).to_numpy()
        X = X_df.to_numpy().astype(float)
        names = list(X_df.columns)
        return X, y, names

    elif DATASET == "excel":
        # Lê de Excel (última coluna = target, a não ser que TARGET_COL esteja definido)
        X_df = pd.read_excel(EXCEL_PATH)
        if TARGET_COL in X_df.columns:
            y = X_df[TARGET_COL].to_numpy()
            X_df = X_df.drop(columns=[TARGET_COL])
        else:
            # assume última coluna é o target
            y = X_df.iloc[:, -1].to_numpy()
            X_df = X_df.iloc[:, :-1]
        X = X_df.to_numpy().astype(float)
        names = list(X_df.columns)
        return X, y, names

    else:
        raise ValueError("DATASET inválido. Use 'sklearn_diabetes', 'pima_openml' ou 'excel'.")

# --------- Main -----------
def main():
    X, y, feat_names = load_data()

    # Força coerência com TASK
    if TASK == "regression":
        y = y.astype(float)
    else:
        # garante 0/1
        y = (y > 0).astype(int)

    # Escalonamento
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)

    # Split train/test
    Xtr, Xte, ytr, yte = train_test_split(
        Xz, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=(y if TASK=="classification" else None)
    )

    # FCM sobre treino (em espaço escalonado)
    centers, U, *_ = fuzz.cluster.cmeans(
        data=Xtr.T, c=N_CLUSTERS, m=M_FCM, error=1e-5, maxiter=300, init=None, seed=RANDOM_STATE
    )  # centers: (R,D), U: (R,Ntr)

    # Estimar sigmas ponderados
    centers_w, sigmas_w = _weighted_mean_std(Xtr, U)    # (R,D), (R,D)
    # Usa centers do FCM + sigmas ponderadas
    centers_use = centers
    sigmas_use = sigmas_w

    # Construir modelo TSK
    R, D = centers_use.shape
    model = TSKModel(centers=centers_use, sigmas=sigmas_use, D=D, R=R)

    # Treino por LS (fecho analítico)
    train_ls(model, Xtr, ytr, TASK)

    # Avaliação
    y_pred_tr, _, _ = model(torch.from_numpy(Xtr.astype(np.float32)))
    y_pred_te, _, _ = model(torch.from_numpy(Xte.astype(np.float32)))

    if TASK == "regression":
        yhat_te = _to_numpy(y_pred_te).ravel()
        rmse = np.sqrt(mean_squared_error(yte, yhat_te))
        print(f"[REG] Test RMSE: {rmse:.3f}")
    else:
        # para classificação, aplicar sigmoid ao output TSK (logit aproximado)
        yhat_proba_te = 1 / (1 + np.exp(-_to_numpy(y_pred_te).ravel()))
        yhat_te = (yhat_proba_te >= 0.5).astype(int)
        acc = accuracy_score(yte, yhat_te)
        f1 = f1_score(yte, yhat_te)
        try:
            auc = roc_auc_score(yte, yhat_proba_te)
        except Exception:
            auc = float("nan")
        print(f"[CLS] Test Acc: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")

    # Info de regras (centros/sigmas em z-score)
    print("\nRegras (centros/sigmas em espaço padronizado):")
    for r in range(R):
        c_txt = ", ".join([f"{feat_names[d]}≈{centers_use[r,d]:+.2f}σ" for d in range(D)])
        s_txt = ", ".join([f"σ_{feat_names[d]}={sigmas_use[r,d]:.2f}" for d in range(D)])
        print(f"- Regra {r+1}: centro[{c_txt}] | {s_txt}")

if __name__ == "__main__":
    main()
