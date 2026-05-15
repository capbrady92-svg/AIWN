"""
Wine Quality dataset loading and training utilities.

11 continuous chemical features, predict quality (regression or classification).
No binary features, moderate complexity, genuine regional structure.

Features: fixed acidity, volatile acidity, citric acid, residual sugar,
          chlorides, free sulfur dioxide, total sulfur dioxide, density,
          pH, sulphates, alcohol
Target: quality score (3-8)
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from aiwn.layers.indexed_linear import IndexedLinear
from aiwn.layers.indexed_linear_v2 import IndexedLinearV2


def load_wine(device: torch.device, val_split: float = 0.1,
              test_split: float = 0.1, seed: int = 42):
    try:
        from sklearn.datasets import load_wine as _load_wine
        from sklearn.preprocessing import QuantileTransformer
        import numpy as np
    except ImportError:
        raise ImportError("pip install scikit-learn")

    # Use UCI wine quality dataset (red + white combined) via pandas
    try:
        import pandas as pd
        red   = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",   sep=';')
        white = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
        df    = pd.concat([red, white], ignore_index=True)
        X     = df.drop('quality', axis=1).values.astype("float32")
        y     = df['quality'].values.astype("float32")
    except Exception:
        # Fallback to sklearn wine dataset (different but clean)
        print("  Using sklearn wine dataset (classification, 3 classes)...")
        data  = _load_wine()
        X     = data.data.astype("float32")
        y     = data.target.astype("float32")

    print(f"  Loading Wine dataset...")

    # All features continuous — QuantileTransformer → uniform[-1, 1]
    qt = QuantileTransformer(output_distribution='uniform', random_state=seed)
    X  = qt.fit_transform(X).astype("float32") * 2.0 - 1.0

    # Normalize target to [0, 1]
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min)

    # Shuffle and split
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=rng).numpy()
    X, y_norm = X[idx], y_norm[idx]

    n       = len(X)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_test - n_val

    def to_gpu(arr, dtype=torch.float32):
        return torch.from_numpy(arr).to(dtype=dtype, device=device)

    train = (to_gpu(X[:n_train]),             to_gpu(y_norm[:n_train]))
    val   = (to_gpu(X[n_train:n_train+n_val]), to_gpu(y_norm[n_train:n_train+n_val]))
    test  = (to_gpu(X[n_train+n_val:]),       to_gpu(y_norm[n_train+n_val:]))

    print(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")
    print(f"  Features: {X.shape[1]} | Target range: {y_min:.0f}–{y_max:.0f}")
    print(f"  Distribution: min={X.min():.3f} max={X.max():.3f} "
          f"mean={X.mean():.3f} std={X.std():.3f}")

    return train, val, test, X.shape[1], float(y_min), float(y_max)


class StandardModel(nn.Module):
    """Single linear layer."""
    def __init__(self, n_features: int, hidden: int = 0):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden), nn.ReLU(),
                nn.Linear(hidden, 1))
            desc = f"Linear({n_features}→{hidden}→1)"
        else:
            self.net = nn.Linear(n_features, 1)
            desc = f"Linear({n_features}→1)"
        n = sum(p.numel() for p in self.parameters())
        print(f"  StandardModel: {desc} | {n:,} params")

    def forward(self, x):
        return self.net(x).squeeze(-1)


class IndexedModel(nn.Module):
    """Single or 2-layer indexed model."""
    def __init__(self, n_features: int, K: int, hidden: int = 0):
        super().__init__()
        self.K = K
        if hidden:
            self.l1   = IndexedLinearV2(n_features, hidden, K)
            self.l2   = IndexedLinearV2(hidden, 1, K)
            self.hidden = hidden
            desc = f"IndexedLinearV2({n_features}→{hidden}→1, K={K})"
        else:
            self.l1     = IndexedLinearV2(n_features, 1, K)
            self.l2     = None
            self.hidden = 0
            desc = f"IndexedLinearV2({n_features}→1, K={K})"
        n = sum(p.numel() for p in self.parameters())
        print(f"  IndexedModel: {desc} | {n:,} params")

    def forward(self, x):
        h = self.l1(x)
        if self.l2 is not None:
            h = self.l2(F.relu(h))
        return h.squeeze(-1)


def evaluate(model, data, batch_size, device):
    x, y = data
    model.eval()
    mse = mae = n = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb, yb = x[i:i+batch_size], y[i:i+batch_size]
            pred = model(xb)
            mse += F.mse_loss(pred, yb).item()
            mae += (pred - yb).abs().mean().item()
            n   += 1
    return mse / n, mae / n


def train_and_eval(model, train_data, val_data, test_data,
                   epochs, batch_size, lr, weight_decay, device, label):
    train_x, train_y = train_data
    N = len(train_x)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\n{'='*55}")
    print(f"Training: {label}")
    print(f"{'='*55}")

    step_times, best_val = [], float("inf")
    curve = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=device)
        eloss, nb = 0.0, 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = train_x[idx], train_y[idx]
            t0   = time.perf_counter()
            loss = F.mse_loss(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t0) * 1000)
            eloss += loss.item(); nb += 1

        val_mse, val_mae = evaluate(model, val_data, batch_size, device)
        best_val = min(best_val, val_mse)
        curve.append((epoch, eloss/nb, val_mse, val_mae))
        print(f"  epoch {epoch:3d} | loss {eloss/nb:.4f} "
              f"| val_mse {val_mse:.4f} | val_mae {val_mae:.4f}")

    test_mse, test_mae = evaluate(model, test_data, batch_size, device)
    avg_ms = sum(step_times) / len(step_times)
    print(f"\n  Best val MSE  : {best_val:.4f}")
    print(f"  Test MSE      : {test_mse:.4f} | Test MAE: {test_mae:.4f}")
    print(f"  Avg step time : {avg_ms:.3f}ms")

    return {"label": label, "best_val_mse": best_val,
            "test_mse": test_mse, "test_mae": test_mae,
            "avg_step_ms": avg_ms, "curve": curve}