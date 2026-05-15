"""
IndexedLinear v2 — LayerNorm + Gaussian CDF normalization.

Wraps the original IndexedLinear with a GaussianCDFNorm that maps
any input distribution to uniform[-1, 1].

Falls back to eager path automatically if Triton is unavailable.
Failure is detected ONCE at init, not retried every forward call.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from aiwn.layers.indexed_linear import IndexedLinear, TRITON_OK


class GaussianCDFNorm(nn.Module):
    """
    LayerNorm + Gaussian CDF — maps any distribution to uniform[-1, 1].

      1. LayerNorm(x)  →  N(0, 1)
      2. 0.5*(1 + erf(x/sqrt(2)))  →  Uniform[0, 1]
      3. * 2 - 1  →  Uniform[-1, 1]

    Only 2*in_d learnable parameters (LayerNorm scale/shift).
    No auxiliary loss needed.
    """
    def __init__(self, in_d: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        u = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return u * 2.0 - 1.0


class IndexedLinearV2(nn.Module):
    """
    IndexedLinear with Gaussian CDF normalization.

    Detects Triton availability ONCE at init via a test forward pass.
    Uses eager path permanently if Triton fails — no per-call overhead.

    Parameters
    ----------
    in_d  : input dimension
    out_d : output dimension
    K     : number of buckets
    """
    def __init__(self, in_d: int, out_d: int, K: int):
        super().__init__()
        self.in_d  = in_d
        self.out_d = out_d
        self.K     = K

        self.cdf    = GaussianCDFNorm(in_d)
        self.linear = IndexedLinear(in_d, out_d, K)

        # Detect Triton availability ONCE with a test forward pass
        self._use_triton = False
        if TRITON_OK:
            try:
                test_x = torch.zeros(1, in_d, device='cuda' 
                    if torch.cuda.is_available() else 'cpu')
                self.linear(test_x)
                self._use_triton = True
            except Exception:
                self._use_triton = False

        n_cdf    = sum(p.numel() for p in self.cdf.parameters())
        n_linear = self.linear.table.numel() + self.linear.bias.numel()
        backend  = "Triton" if self._use_triton else "eager"
        print(f"  IndexedLinearV2: ({in_d}, {out_d}, K={K}) | "
              f"cdf={n_cdf} + linear={n_linear:,} = {n_cdf+n_linear:,} params "
              f"| backend={backend}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        xf    = x.reshape(-1, self.in_d)
        xf    = self.cdf(xf)                    # → uniform[-1, 1]
        if self._use_triton:
            out = self.linear(xf)               # Triton kernel
        else:
            out = self.linear._eager(xf)        # pure PyTorch
        return out.reshape(*shape[:-1], self.out_d)

    def flops(self) -> int:
        return self.linear.flops()

    def active_bytes(self) -> int:
        return self.linear.active_bytes()

    def gradient_density(self) -> float:
        return 2.0 / self.K