#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topological_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline: (K_o, h, ε₂, ε₃) → Chern number → topological phase classifier.

Modules
-------
1. Hamiltonian Engine   — 6×6 Bloch matrix, multipolar spin liquid / honeycomb
2. FHS Chern Integrator — Fukui-Hatsugai-Suzuki (2005), fully vectorized
3. Monte Carlo Gen.     — Uniform parameter sampling → labeled CSV
4. MLP Classifier       — Refatorado para Classificação Binária (Topological vs Trivial)
                          Injeção de Focal Loss, Calibração Dinâmica e Regularização L2/Dropout.

Dependencies: numpy, pandas, torch, scikit-learn, tqdm, imblearn
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — BULK HAMILTONIAN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_r3i = 1.0 / np.sqrt(3.0)
NN: np.ndarray = np.array(
    [[0.0, _r3i], [0.5, -0.5 * _r3i], [-0.5, -0.5 * _r3i]], dtype=np.float64
)

_r2 = np.sqrt(2.0)
Jx: np.ndarray = (np.array([[0, _r2, 0], [_r2, 0, _r2], [0, _r2, 0]], dtype=complex) * 0.5)
Jy: np.ndarray = (np.array([[0, -1j * _r2, 0], [1j * _r2, 0, -1j * _r2], [0, 1j * _r2, 0]], dtype=complex) * 0.5)
Jz: np.ndarray = np.diag([1.0, 0.0, -1.0]).astype(complex)
I3: np.ndarray = np.eye(3, dtype=complex)

O20: np.ndarray = 3.0 * (Jz @ Jz) - 2.0 * I3
O22c: np.ndarray = Jx @ Jx - Jy @ Jy
O22s: np.ndarray = Jx @ Jy + Jy @ Jx

def _hamiltonian_batch(
    kx_g: np.ndarray,
    ky_g: np.ndarray,
    Ko: float,
    h: float,
    eps2: float,
    eps3: float,
    alpha: float = 0.5,
) -> np.ndarray:
    phi = kx_g[:, :, None] * NN[:, 0] + ky_g[:, :, None] * NN[:, 1]
    f_k = np.exp(1j * phi).sum(axis=-1)
    
    H_cf = Ko * O20 + h * Jz + eps2 * O22c + eps3 * O22s
    T    = I3 + alpha * Ko * (Jx + Jy)

    H_AB = f_k[:, :, None, None] * T[None, None]

    H = np.zeros((*kx_g.shape, 6, 6), dtype=complex)
    H[:, :, :3, :3] =  H_cf
    H[:, :, 3:, 3:] = -H_cf
    H[:, :, :3, 3:] =  H_AB
    H[:, :, 3:, :3] =  H_AB.conj().transpose(0, 1, 3, 2)
    return H


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — FHS CHERN INTEGRATOR
# ══════════════════════════════════════════════════════════════════════════════

def check_gap(eigenvalues: np.ndarray, n_occ: int, tol: float = 1e-6) -> bool:
    gap_min = np.min(eigenvalues[..., n_occ] - eigenvalues[..., n_occ - 1])
    return gap_min > tol

def fhs_chern_number(H_batch: np.ndarray, n_occ: int) -> int | None:
    eigvals, psi_all = np.linalg.eigh(H_batch)
    if not check_gap(eigvals, n_occ):
        return None
        
    psi = psi_all[:, :, :, :n_occ]

    def _link(ax: int) -> np.ndarray:
        psi_fwd = np.roll(psi, -1, axis=ax)
        M = np.einsum("...ia,...ib->...ab", psi.conj(), psi_fwd)
        det_M = np.linalg.det(M)
        det_M = np.where(np.abs(det_M) < 1e-12, 1.0 + 0j, det_M)
        return det_M / np.abs(det_M)

    U1 = _link(ax=0)
    U2 = _link(ax=1)

    U_plaquette = (
        U1
        * np.roll(U2, -1, axis=0)
        * np.roll(U1, -1, axis=1).conj()
        * U2.conj()
    )
    F_tilde = np.angle(U_plaquette + 1e-10j)
    
    return int(np.round(F_tilde.sum() / (2.0 * np.pi)))

def compute_chern(Ko: float, h: float, eps2: float, eps3: float, N: int = 60, n_occ: int = 3) -> int | None:
    k1d = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    kx_g, ky_g = np.meshgrid(k1d, k1d, indexing="ij")
    H_batch = _hamiltonian_batch(kx_g, ky_g, Ko, h, eps2, eps3, alpha=0.5)
    return fhs_chern_number(H_batch, n_occ)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — MONTE CARLO DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

CSV_PATH = Path("topological_dataset.csv")

_BOUNDS: dict[str, tuple[float, float]] = {
    "Ko":   (0.0, 3.0),
    "h":    (-5.0, 5.0),
    "eps2": (0.0, 1.5),
    "eps3": (0.0, 1.5),
}

def generate_dataset(n_samples: int = 5000, N_bz: int = 60, n_occ: int = 3, seed: int = 42, out: Path = CSV_PATH) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    valid_rows = []
    
    pbar = tqdm(total=n_samples, desc="FHS Integrator")

    while len(valid_rows) < n_samples:
        batch_size = min(500, n_samples - len(valid_rows) + 200)
        Ko_v   = rng.uniform(*_BOUNDS["Ko"], batch_size)
        h_v    = rng.uniform(*_BOUNDS["h"], batch_size)
        eps2_v = rng.uniform(*_BOUNDS["eps2"], batch_size)
        eps3_v = rng.uniform(*_BOUNDS["eps3"], batch_size)

        for i in range(batch_size):
            if len(valid_rows) >= n_samples:
                break
            c = compute_chern(Ko_v[i], h_v[i], eps2_v[i], eps3_v[i], N=N_bz, n_occ=n_occ)
            if c is not None:
                valid_rows.append((Ko_v[i], h_v[i], eps2_v[i], eps3_v[i], c))
                pbar.update(1)

    pbar.close()
    df = pd.DataFrame(valid_rows, columns=["Ko", "h", "eps2", "eps3", "chern"])
    df.to_csv(out, index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — BINARY TOPOLOGICAL CLASSIFIER (PyTorch MLP)
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = ["Ko", "h", "eps2", "eps3"]

# ARQUITETURA: Implementação estrita da Focal Loss. 
# Equação matemática mapeada: FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
# Objetivo: Forçar o cálculo de gradientes a punir instâncias mal calibradas
# (falsos positivos/negativos latentes) oriundas da classe minoritária.
class FocalLossBinary(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_with_logits(logits, targets)
        pt = torch.exp(-bce_loss)  # p_t matemático exato a partir da BCE
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class ChernDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        # Targets convertidos para float32 devido à exigência arquitetural da BCEWithLogits
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1)
        
    def __len__(self) -> int: return len(self.y)
    def __getitem__(self, i: int): return self.X[i], self.y[i]

class TopoPhaseMLP(nn.Module):
    # ARQUITETURA: Regulação de Custo
    # Aplicação agressiva de Dropout (p = 0.4) para interrupção de co-adaptação 
    # de features. Mitiga a falha de generalização estrutural após a época 40.
    def __init__(self, p: float = 0.4) -> None:
        super().__init__()
        def _block(d_in: int, d_out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(p))

        self.net = nn.Sequential(
            _block(4,   128),
            _block(128, 256),
            _block(256, 128),
            _block(128,  64),
            nn.Linear(64, 1)  # Output unitário não-ativado (logits) para classificação binária
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(csv_path: Path = CSV_PATH, epochs: int = 150, batch_size: int = 256, lr: float = 1e-3, val_frac: float = 0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(csv_path)
    X_raw = df[FEATURES].values.astype(np.float32)
    
    # Adaptação para problema binário: Topological (Chern != 0) -> 1, Trivial (Chern == 0) -> 0
    y_raw = (df["chern"].values != 0).astype(np.float32)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y_raw))
    n_val = int(len(y_raw) * val_frac)
    tr_idx, va_idx = idx[n_val:], idx[:n_val]

    X_tr_raw, y_tr_raw = X_raw[tr_idx], y_raw[tr_idx]

    smote = SMOTE(random_state=42)
    X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_raw, y_tr_raw)
    
    scaler = StandardScaler().fit(X_tr_smote)
    X_tr = scaler.transform(X_tr_smote)
    X_va = scaler.transform(X_raw[va_idx])

    tr_loader = DataLoader(ChernDataset(X_tr, y_tr_smote), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(ChernDataset(X_va, y_raw[va_idx]), batch_size=512, shuffle=False)

    model = TopoPhaseMLP(p=0.4).to(device)

    # ARQUITETURA: Substituição Direta do Critério de Otimização
    criterion = FocalLossBinary(alpha=0.75, gamma=2.0).to(device)

    # ARQUITETURA: Regulação de Custo (Optimizer)
    # Injeção de weight_decay = 1e-4 (L2 Penalty) no AdamW para achatamento 
    # forçado da curva de erro de validação.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None
    patience = 15
    epochs_no_improve = 0
    
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(y_tr_smote)

        model.eval()
        va_loss = 0.0
        all_probs, all_targets = [], []
        
        with torch.no_grad():
            for Xb, yb in va_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                va_loss += criterion(logits, yb).item() * len(yb)
                
                # Extração de probabilidades puras para calibração de limiar
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy().flatten())
                all_targets.extend(yb.cpu().numpy().flatten())
                
        va_loss /= len(va_idx)
        
        # ARQUITETURA: Calibração Dinâmica do Limiar (Threshold Calibration)
        # Substitui t = 0.5 por uma busca estrita na faixa matemática de [0.65, 0.70]
        # visando assimetria favorável à Precision.
        precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
        
        # Filtragem na fronteira operacional t ∈ [0.65, 0.70]
        valid_idx = np.where((thresholds >= 0.65) & (thresholds <= 0.70))[0]
        
        if len(valid_idx) > 0:
            # Maximiza estritamente a Precision dentro do intervalo autorizado
            best_idx = valid_idx[np.argmax(precisions[valid_idx])]
            optimal_threshold = thresholds[best_idx]
        else:
            # Fallback seguro caso o range não intercepte a curva na iteração atual
            optimal_threshold = 0.675 

        # Predição computada sob novo limiar matemático calibrado
        all_preds = (np.array(all_probs) >= optimal_threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='binary', zero_division=0
        )

        scheduler.step()

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model, scaler

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not CSV_PATH.exists():
        df = generate_dataset(n_samples=2000, N_bz=60, n_occ=3, seed=42)

    model, scaler = train_classifier(csv_path=CSV_PATH, epochs=150, batch_size=256, lr=1e-3)

    torch.save(
        {
            "model_state":   model.state_dict(),
            "scaler_mean":   scaler.mean_,
            "scaler_scale":  scaler.scale_,
        },
        "topological_mlp_binary.pt",
    )

