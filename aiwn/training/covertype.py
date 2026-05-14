"""
Covertype dataset loading and training utilities for AIWN layer comparison.

Covertype predicts forest cover type (7 classes) from 54 cartographic features.
It has strong regional structure — elevation bands, soil types, wilderness areas
all behave differently. A single global linear layer structurally cannot fit
this well. IndexedLinear's K local transformations should handle it better.

Dataset: UCI Covertype (~581k samples, 54 features, 7 classes)
Source: sklearn.datasets.fetch_covtype

Provides:
  - load_covertype()    — loads and preprocesses dataset
  - StandardModel       — single linear layer: Linear(54, 7)
  - IndexedModel        — single IndexedLinear(54, 7, K) — true drop-in
  - train_and_eval()    — training loop returning metrics
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from aiwn.layers import IndexedLinear
from aiwn.layers.indexed_linear import TRITON_OK
try:
    from aiwn.layers.indexed_linear_v2 import IndexedLinearV2
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_covertype(device: torch.device, val_split: float = 0.1,
                   test_split: float = 0.1, seed: int = 42):
    """
    Load and preprocess UCI Covertype dataset via sklearn.

    - Standardizes features to zero mean, unit variance
    - Scales to [-1, 1] for IndexedLinear bucket domain
    - Returns train/val/test splits as GPU tensors

    Returns
    -------
    (train_x, train_y), (val_x, val_y), (test_x, test_y), n_features, n_classes
    """
    try:
        from sklearn.datasets import fetch_covtype
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("pip install scikit-learn")

    print("  Loading Covertype dataset...")
    data = fetch_covtype()
    X    = data.data.astype("float32")
    y    = (data.target - 1).astype("int64")   # 1-7 → 0-6

    # Per-feature min-max scaling to [-1, 1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # After standardization most values are in [-3, 3] — scale to [-1, 1]
    X = X / 3.0
    X = X.clip(-1.0, 1.0).astype("float32")

    # Reproducible shuffle + split
    rng  = torch.Generator().manual_seed(seed)
    idx  = torch.randperm(len(X), generator=rng).numpy()
    X, y = X[idx], y[idx]

    n      = len(X)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)
    n_train = n - n_test - n_val

    X_tr, y_tr = X[:n_train],            y[:n_train]
    X_va, y_va = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_te, y_te = X[n_train+n_val:],       y[n_train+n_val:]

    def to_gpu(arr, dtype=torch.float32):
        return torch.from_numpy(arr).to(dtype=dtype, device=device)

    train = (to_gpu(X_tr), to_gpu(y_tr, torch.long))
    val   = (to_gpu(X_va), to_gpu(y_va, torch.long))
    test  = (to_gpu(X_te), to_gpu(y_te, torch.long))

    n_features = X.shape[1]   # 54
    n_classes  = int(y.max()) + 1  # 7

    print(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")
    print(f"  Features: {n_features} | Classes: {n_classes}")
    return train, val, test, n_features, n_classes


# ── Models ────────────────────────────────────────────────────────────────────

class StandardModel(nn.Module):
    """
    Single linear layer: Linear(n_features, n_classes).
    This is the baseline — one global affine transformation.
    Cannot represent piecewise structure in the input space.
    """
    def __init__(self, n_features: int = 54, n_classes: int = 7):
        super().__init__()
        self.layer = nn.Linear(n_features, n_classes)
        n = sum(p.numel() for p in self.parameters())
        print(f"  StandardModel: Linear({n_features}, {n_classes}) | {n:,} params")

    def forward(self, x):
        return self.layer(x)


class IndexedModel(nn.Module):
    """
    Single IndexedLinear layer: IndexedLinear(n_features, n_classes, K).

    True drop-in replacement for nn.Linear — same input/output dimensions.
    K weight regimes instead of one global matrix.

    No proj_in/proj_out — input is already in [-1, 1] after preprocessing.
    Input dimensions are small enough (54) that no dimension reduction needed.

    Parameter count: K * n_features * n_classes (K times more than standard).
    This is intentional — we're testing whether K local transformations
    can learn what one global transformation structurally cannot.
    """
    def __init__(self, n_features: int = 54, n_classes: int = 7, K: int = 32):
        super().__init__()
        self.layer = IndexedLinear(n_features, n_classes, K)
        self.K = K
        n = sum(p.numel() for p in self.parameters())
        std_params = n_features * n_classes + n_classes
        print(f"  IndexedModel: IndexedLinear({n_features}, {n_classes}, K={K}) | "
              f"{n:,} params (vs std {std_params:,} = {n/std_params:.1f}x)")

    def forward(self, x):
        return self.layer(x)



class IndexedModelV2(nn.Module):
    """
    Single IndexedLinearV2 layer — Gaussian CDF normalization.
    LayerNorm + erf maps input to uniform[-1,1] before bucket indexing.
    No auxiliary loss needed.
    """
    def __init__(self, n_features: int = 54, n_classes: int = 7,
                 K: int = 32, entropy_weight: float = 0.01):
        super().__init__()
        # entropy_weight kept for API compatibility but no longer used
        self.layer = IndexedLinearV2(n_features, n_classes, K)
        self.K = K
        n = sum(p.numel() for p in self.parameters())
        print(f"  IndexedModelV2: IndexedLinearV2({n_features}, {n_classes}, K={K}) | "
              f"{n:,} params (Gaussian CDF norm)")

    def forward(self, x):
        return self.layer(x)

# ── Training ──────────────────────────────────────────────────────────────────

def evaluate(model, data, batch_size, device):
    """Compute accuracy and loss on a dataset."""
    x, y   = data
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i+batch_size]
            yb = y[i:i+batch_size]
            logits = model(xb)
            total_loss    += F.cross_entropy(logits, yb).item()
            total_correct += (logits.argmax(1) == yb).sum().item()
            n_batches += 1

    return total_loss / n_batches, total_correct / len(x)


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
    """
    Train model on Covertype for `epochs` epochs.
    Evaluates on val set each epoch, test set at end.

    Returns
    -------
    dict with keys: label, best_val_acc, test_acc, avg_step_ms, curve
    """
    train_x, train_y = train_data
    N = len(train_x)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"{'='*60}")

    step_times   = []
    best_val_acc = 0.0
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
            loss = F.cross_entropy(model(xb), yb)
            # No auxiliary loss for V2 (Gaussian CDF needs none)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t0) * 1000)

            epoch_loss += loss.item()
            n_batches  += 1

        val_loss, val_acc = evaluate(model, val_data, batch_size, device)
        best_val_acc = max(best_val_acc, val_acc)
        avg_loss = epoch_loss / n_batches
        curve.append((epoch, avg_loss, val_acc))

        print(f"  epoch {epoch:3d} | train_loss {avg_loss:.4f} "
              f"| val_acc {val_acc*100:.2f}%")

    # Final test evaluation
    _, test_acc = evaluate(model, test_data, batch_size, device)
    avg_ms      = sum(step_times) / len(step_times)

    print(f"\n  Best val acc  : {best_val_acc*100:.2f}%")
    print(f"  Final test acc: {test_acc*100:.2f}%")
    print(f"  Avg step time : {avg_ms:.3f}ms")

    return {
        "label":        label,
        "best_val_acc": best_val_acc,
        "test_acc":     test_acc,
        "avg_step_ms":  avg_ms,
        "curve":        curve,
    }