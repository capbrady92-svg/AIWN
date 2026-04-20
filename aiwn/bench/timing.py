"""
GPU-synchronised timing utilities and layer-level benchmarking.
"""

import time
import numpy as np
import torch

from aiwn.layers import IndexedLinear, StandardLinear
from aiwn.layers.indexed_linear import TRITON_OK, _Fn  # type: ignore[attr-defined]


def sync(device: torch.device):
    """Synchronise CUDA if applicable."""
    if device.type == 'cuda':
        torch.cuda.synchronize()


def bench_fn(fn, n_warm: int, n_bench: int, device: torch.device):
    """
    Time `fn` with GPU synchronisation.

    Returns (median_ms, p95_ms).
    """
    for _ in range(n_warm):
        fn()
    sync(device)
    times = []
    for _ in range(n_bench):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def bench_layer(
    layer_std: StandardLinear,
    layer_idx: IndexedLinear,
    B: int,
    IN: int,
    OUT: int,
    K: int,
    n_warm: int,
    n_bench: int,
    device: torch.device,
) -> dict:
    """
    Benchmark a (StandardLinear, IndexedLinear) pair at batch size B.

    Returns a dict of timing and FLOP metrics. Accuracy metrics are handled
    separately by the training module so this function stays pure-timing.
    """
    x_std = torch.randn(B, IN,               device=device, requires_grad=True)
    x_idx = torch.empty(B, layer_idx.in_dim, device=device).uniform_(-1, 1).requires_grad_(True)

    fwd_std, _ = bench_fn(lambda: layer_std(x_std), n_warm, n_bench, device)
    fwd_idx, _ = bench_fn(lambda: layer_idx(x_idx), n_warm, n_bench, device)

    def full_std():
        xs = x_std.detach().requires_grad_(True)
        layer_std(xs).sum().backward()

    def full_idx():
        xi = x_idx.detach().requires_grad_(True)
        t  = layer_idx.table.detach().requires_grad_(True)
        b  = layer_idx.bias.detach().requires_grad_(True)
        if TRITON_OK and xi.is_cuda:
            _Fn.apply(xi, t, b, layer_idx.bw).sum().backward()
        else:
            layer_idx._eager(xi).sum().backward()

    step_std_ms, _ = bench_fn(full_std, n_warm, n_bench, device)
    step_idx_ms, _ = bench_fn(full_idx, n_warm, n_bench, device)

    flops_std = layer_std.flops()
    flops_idx = layer_idx.flops()

    return {
        'fwd_std_ms':      fwd_std,
        'fwd_idx_ms':      fwd_idx,
        'fwd_speedup':     fwd_std / fwd_idx,
        'step_std_ms':     step_std_ms,
        'step_idx_ms':     step_idx_ms,
        'step_speedup':    step_std_ms / step_idx_ms,
        'flops_std':       flops_std,
        'flops_idx':       flops_idx,
        'flop_ratio':      flops_std / flops_idx,
        'tflops_std':      flops_std * B / fwd_std  / 1e9,
        'tflops_idx':      flops_idx * B / fwd_idx  / 1e9,
        'idx_faster_fwd':  fwd_idx     < fwd_std,
        'idx_faster_step': step_idx_ms < step_std_ms,
        'active_kb':       layer_idx.active_bytes() / 1024,
    }