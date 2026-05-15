"""
California Housing dataset loading and training utilities.

8 purely continuous features:
  MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

Target: median house value (regression)

Strong regional structure — Bay Area, LA, Central Valley, rural areas
have fundamentally different price-to-feature relationships.
A single global linear layer cannot capture this. IndexedLinear should.

Provides:
  - load_housing()     — loads, preprocesses, splits dataset
  - StandardModel      — nn.Linear(8, 1) — global affine
  - IndexedModelV2     — IndexedLinearV2(8, 1, K) — piecewise regional
  - train_and_eval()   — training loop returning metrics
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from aiwn.layers.indexed_linear import IndexedLinear
from aiwn.layers.indexed_linear_v2 import IndexedLinearV2


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_housing(device: torch.device, val_split: float = 0.1,
                 test_split: float = 0.1, seed: int = 42):
    """
    Load California Housing via sklearn, preprocess to [-1, 1],
    return train/val/test splits as GPU tensors.

    Features (all continuous):
      0: MedInc      — median income
      1: HouseAge    — median house age
      2: AveRooms    — average rooms per household
      3: AveBedrms   — average bedrooms per household
      4: Population  — block population
      5: AveOccup    — average occupancy
      6: Latitude    — geographic latitude
      7: Longitude   — geographic longitude

    Target: MedHouseVal (median house value in $100k)
    """
    try:
        from sklearn.datasets import fetch_california_housing
        from sklearn.preprocessing import QuantileTransformer
    except ImportError:
        raise ImportError("pip install scikit-learn")

    print("  Loading California Housing dataset...")
    data = fetch_california_housing()
    X    = data.data.astype("float32")   # (20640, 8)
    y    = data.target.astype("float32") # (20640,) — values in [0.15, 5.0]

    # All features are continuous — use QuantileTransformer for true uniform[-1,1]
    # This gives each feature equal bucket coverage regardless of skew
    qt = QuantileTransformer(output_distribution='uniform', random_state=seed)
    X  = qt.fit_transform(X).astype("float32")
    X  = X * 2.0 - 1.0   # [0,1] → [-1, 1]

    # Normalize target to [0, 1] for stable training
    y_min, y_max = y.min(), y.max()
    y = (y - y_min) / (y_max - y_min)

    # Shuffle and split
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=rng).numpy()
    X, y = X[idx], y[idx]

    n       = len(X)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_test - n_val

    def to_gpu(arr, dtype=torch.float32):
        return torch.from_numpy(arr).to(dtype=dtype, device=device)

    train = (to_gpu(X[:n_train]),            to_gpu(y[:n_train]))
    val   = (to_gpu(X[n_train:n_train+n_val]), to_gpu(y[n_train:n_train+n_val]))
    test  = (to_gpu(X[n_train+n_val:]),      to_gpu(y[n_train+n_val:]))

    print(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")
    print(f"  Features: {X.shape[1]} (all continuous) | Target: regression")
    print(f"  Feature distribution: min={X.min():.3f} max={X.max():.3f} "
          f"mean={X.mean():.3f} std={X.std():.3f}")

    return train, val, test, X.shape[1], y_min, y_max


# ── Models ────────────────────────────────────────────────────────────────────

class StandardModel(nn.Module):
    """Single linear layer — one global affine: Linear(8, 1)."""
    def __init__(self, n_features: int = 8):
        super().__init__()
        self.layer = nn.Linear(n_features, 1)
        n = sum(p.numel() for p in self.parameters())
        print(f"  StandardModel: Linear({n_features}, 1) | {n:,} params")

    def forward(self, x):
        return self.layer(x).squeeze(-1)


class IndexedModelV2(nn.Module):
    """
    IndexedLinearV2(n_features, 1, K) — piecewise regional regression.

    Gaussian CDF normalization ensures all 8 continuous features
    are uniformly distributed across all K buckets from the start.
    """
    def __init__(self, n_features: int = 8, K: int = 32):
        super().__init__()
        self.layer = IndexedLinearV2(n_features, 1, K)
        self.K = K
        n = sum(p.numel() for p in self.parameters())
        std_n = n_features + 1
        print(f"  IndexedModelV2: IndexedLinearV2({n_features}, 1, K={K}) | "
              f"{n:,} params (vs std {std_n} = {n/std_n:.0f}x)")

    def forward(self, x):
        return self.layer(x).squeeze(-1)


class IndexedModelV1(nn.Module):
    """IndexedLinear(n_features, 1, K) — no CDF normalization."""
    def __init__(self, n_features: int = 8, K: int = 32):
        super().__init__()
        self.layer = IndexedLinear(n_features, 1, K)
        self.K = K
        n = sum(p.numel() for p in self.parameters())
        std_n = n_features + 1
        print(f"  IndexedModelV1: IndexedLinear({n_features}, 1, K={K}) | "
              f"{n:,} params (vs std {std_n} = {n/std_n:.0f}x)")

    def forward(self, x):
        return self.layer(x).squeeze(-1)


# ── Eval ──────────────────────────────────────────────────────────────────────

def evaluate(model, data, batch_size, device):
    """Compute MSE and MAE on a dataset."""
    x, y = data
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i+batch_size]
            yb = y[i:i+batch_size]
            pred = model(xb)
            total_mse += F.mse_loss(pred, yb).item()
            total_mae += (pred - yb).abs().mean().item()
            n_batches += 1
    return total_mse / n_batches, total_mae / n_batches


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_eval(
    model: nn.Module,
    train_data: tuple,
    val_data: tuple,
    test_data: tuple,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    label: str,
) -> dict:
    train_x, train_y = train_data
    N = len(train_x)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"{'='*60}")

    step_times   = []
    best_val_mse = float("inf")
    curve        = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm       = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, N, batch_size):
            idx    = perm[i:i+batch_size]
            xb, yb = train_x[idx], train_y[idx]

            t0   = time.perf_counter()
            loss = F.mse_loss(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t0) * 1000)

            epoch_loss += loss.item()
            n_batches  += 1

        val_mse, val_mae = evaluate(model, val_data, batch_size, device)
        best_val_mse = min(best_val_mse, val_mse)
        curve.append((epoch, epoch_loss / n_batches, val_mse, val_mae))
        print(f"  epoch {epoch:3d} | loss {epoch_loss/n_batches:.4f} "
              f"| val_mse {val_mse:.4f} | val_mae {val_mae:.4f}")

    test_mse, test_mae = evaluate(model, test_data, batch_size, device)
    avg_ms = sum(step_times) / len(step_times)

    print(f"\n  Best val MSE  : {best_val_mse:.4f}")
    print(f"  Test MSE      : {test_mse:.4f}")
    print(f"  Test MAE      : {test_mae:.4f}")
    print(f"  Avg step time : {avg_ms:.3f}ms")

    return {
        "label":        label,
        "best_val_mse": best_val_mse,
        "test_mse":     test_mse,
        "test_mae":     test_mae,
        "avg_step_ms":  avg_ms,
        "curve":        curve,
    }