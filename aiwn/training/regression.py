"""
Synthetic regression task for layer accuracy comparison.

Task: learn  Y = tanh(X @ W_true + b_true)  from  X ~ Uniform[-1, 1].

Why this task
-------------
- Input domain [-1, 1] matches IndexedLinear's bucket range exactly.
- tanh nonlinearity means the indexed layer's input-dependent weighting is
  actually exercised — a linear target collapses to an identical solution
  for both layers regardless of architecture.
- W_true is frozen so both layers solve the exact same problem.
- Reproducible: same seed → same dataset → same training order every run.

Any experiment that needs a trained accuracy signal can import this module.
"""

import math
import torch
import torch.nn.functional as F

from aiwn.layers import IndexedLinear, StandardLinear
from aiwn.layers.indexed_linear import indexed_lr


def make_dataset(
    in_d: int,
    out_d: int,
    n: int,
    device: torch.device,
    seed: int = 42,
):
    """
    Generate a fixed synthetic regression dataset.

    Returns
    -------
    (X_train, Y_train), (X_val, Y_val)  — 80/20 split, all on `device`.
    """
    rng    = torch.Generator(device=device).manual_seed(seed)
    W_true = torch.randn(in_d, out_d, generator=rng, device=device) / math.sqrt(in_d)
    b_true = torch.zeros(out_d, device=device)
    X      = torch.empty(n, in_d, device=device).uniform_(-1.0, 1.0)
    with torch.no_grad():
        Y = torch.tanh(X @ W_true + b_true)
    split = int(0.8 * n)
    return (X[:split], Y[:split]), (X[split:], Y[split:])


def train_and_eval(
    layer,
    train_xy,
    val_xy,
    steps: int,
    lr: float,
    batch_size: int,
    ckpt_every: int,
    device: torch.device,
):
    """
    Train `layer` on a regression task with AdamW for `steps` steps.

    For IndexedLinear the LR is automatically scaled by indexed_lr() to
    correct for sparse gradient updates, keeping effective update magnitude
    comparable to a standard dense layer.

    Returns
    -------
    val_mse : float   — final validation MSE
    curve   : list of (step, train_mse) tuples recorded every ckpt_every steps
    """
    if isinstance(layer, IndexedLinear):
        lr = indexed_lr(lr, layer.K)

    opt  = torch.optim.AdamW(layer.parameters(), lr=lr, weight_decay=1e-4)
    X_tr, Y_tr = train_xy
    X_val, Y_val = val_xy
    N_tr  = X_tr.shape[0]
    curve = []

    for step in range(1, steps + 1):
        idx  = torch.randint(0, N_tr, (batch_size,), device=device)
        loss = F.mse_loss(layer(X_tr[idx]), Y_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % ckpt_every == 0:
            curve.append((step, loss.item()))

    with torch.no_grad():
        val_mse = F.mse_loss(layer(X_val), Y_val).item()

    return val_mse, curve


def run_perplexity(
    d_std: int,
    K: int,
    device: torch.device,
    steps: int,
    lr: float,
    batch_size: int,
    n_data: int,
    ckpt_every: int,
    indexed_dims_fn,
) -> dict:
    """
    Train matched Standard and Indexed layers on the synthetic regression task
    and return perplexity-style accuracy metrics.

    StandardLinear uses full (d_std, 4*d_std) dimensions — matching the speed
    benchmark. IndexedLinear uses (d_idx, ff_idx) — equal-parameter scaling
    via indexed_dims. Each layer trains on its own appropriately shaped dataset
    since input dimensions differ.

    ppl_ratio = ppl_idx / ppl_std. Both layers solve the same style of task
    (tanh regression) at their respective scales. <1.0 means indexed is better.

    Returns
    -------
    dict with keys:
        ppl_std        exp(val_mse_std)  — lower is better
        ppl_idx        exp(val_mse_idx)
        ppl_ratio      ppl_idx / ppl_std — <1.0 means indexed is better
        val_mse_std    raw validation MSE for standard layer
        val_mse_idx    raw validation MSE for indexed layer
        loss_curve_std list of (step, mse) checkpoints
        loss_curve_idx list of (step, mse) checkpoints
    """
    d_idx, ff_idx, _ = indexed_dims_fn(d_std, 4 * d_std, K)

    print(f"run_perplexity: d_std={d_std}, K={K}, d_idx={d_idx}, ff_idx={ff_idx}")
    print(f"  layer_std: ({d_std}, {4*d_std}) — params: {d_std * 4 * d_std + 4 * d_std}")
    print(f"  layer_idx: (K={K}, {d_idx}, {ff_idx}) — params: {K * d_idx * ff_idx + ff_idx}")

    layer_std = StandardLinear(d_std, 4 * d_std).to(device)
    layer_idx = IndexedLinear(d_idx, ff_idx, K).to(device)

    train_std, val_std = make_dataset(d_std, 4 * d_std, n_data, device)
    train_idx, val_idx = make_dataset(d_idx, ff_idx, n_data, device)

    val_mse_std, curve_std = train_and_eval(
        layer_std, train_std, val_std, steps, lr, batch_size, ckpt_every, device)
    val_mse_idx, curve_idx = train_and_eval(
        layer_idx, train_idx, val_idx, steps, lr, batch_size, ckpt_every, device)

    ppl_std   = math.exp(min(val_mse_std, 20))
    ppl_idx   = math.exp(min(val_mse_idx, 20))
    ppl_ratio = ppl_idx / max(ppl_std, 1e-9)

    return {
        'ppl_std':        ppl_std,
        'ppl_idx':        ppl_idx,
        'ppl_ratio':      ppl_ratio,
        'val_mse_std':    val_mse_std,
        'val_mse_idx':    val_mse_idx,
        'loss_curve_std': curve_std,
        'loss_curve_idx': curve_idx,
    }