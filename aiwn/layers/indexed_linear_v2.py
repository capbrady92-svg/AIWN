"""
IndexedLinear v2 — faithful implementation of the AIWN whitepaper.

Key differences from v1:
  1. Table shape is (d_in, K, d_out) — per-input-unit lookup, not (K, d_in, d_out)
  2. CDF uses learnable knot positions pushed toward empirical quantiles
     via L_unif = MSE(knots, quantile(x, [1/K, ..., (K-1)/K]))
  3. Knots are sorted to guarantee monotonicity
  4. Gradient locality: exactly 2/K table entries get nonzero gradient per sample

From the paper:
  y_j = sum_i w_ij(a_i) * a_i + b_j
  w_ij(a_i) = table[i, k(a_i), j] * (1-alpha) + table[i, k(a_i)+1, j] * alpha

  where k(a_i) is the bucket index from the CDF-transformed activation,
  and alpha is the fractional position within the bucket.

CDF Activation (Section 3.2):
  K-1 learnable knot positions per neuron.
  L_unif = MSE(knots, quantile(x, [1/K, 2/K, ..., (K-1)/K]))
  Output bounded in (-1, 1), monotone by construction, gradient never zero.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedCDFActivation(nn.Module):
    """
    Learned CDF Activation — maps any input distribution to uniform[-1, 1].

    Per-neuron piecewise-linear monotonic function with K-1 learnable knot
    positions. Trained via uniformity loss to push knots toward empirical
    quantiles of the actual input distribution.

    From paper Section 3.2:
        CDF(x_i) = (k(x_i) + alpha(x_i)) / K * 2 - 1
        L_unif = MSE(knots, quantile(x_i, [1/K, ..., (K-1)/K]))

    Parameters
    ----------
    in_d     : number of input dimensions (one CDF per dimension)
    K        : number of buckets
    unif_weight : weight for uniformity loss
    """

    def __init__(self, in_d: int, K: int, unif_weight: float = 0.1):
        super().__init__()
        self.in_d        = in_d
        self.K           = K
        self.unif_weight = unif_weight

        # K-1 learnable knot positions per input dimension
        # Initialize evenly spaced in [-1, 1]
        knots_init = torch.linspace(-1.0, 1.0, K + 1)[1:-1]  # K-1 interior knots
        # Shape: (in_d, K-1)
        self.knots_raw = nn.Parameter(
            knots_init.unsqueeze(0).repeat(in_d, 1))

        # Fixed boundary knots at -1 and 1
        self.register_buffer("lo", torch.tensor(-1.0))
        self.register_buffer("hi", torch.tensor( 1.0))

    def _sorted_knots(self) -> torch.Tensor:
        """
        Return sorted knot positions with boundaries.
        Shape: (in_d, K+1) — includes -1 and 1 at endpoints.
        Sorting guarantees monotonicity.
        """
        # Sort interior knots and clamp to (-1, 1)
        interior = self.knots_raw.sort(dim=-1).values.clamp(-0.9999, 0.9999)
        lo = self.lo.expand(self.in_d, 1)
        hi = self.hi.expand(self.in_d, 1)
        return torch.cat([lo, interior, hi], dim=-1)  # (in_d, K+1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map x to approximately uniform[-1, 1] via piecewise-linear CDF.

        x shape: (N, in_d)
        output:  (N, in_d)
        """
        knots = self._sorted_knots()  # (in_d, K+1)

        # For each input dimension, find which bucket x falls in
        # and compute fractional position within that bucket
        x_exp    = x.unsqueeze(-1)                          # (N, in_d, 1)
        knots_exp = knots.unsqueeze(0)                      # (1, in_d, K+1)

        # Bucket index: number of knots <= x, minus 1
        bucket = (x_exp >= knots_exp).sum(dim=-1) - 1      # (N, in_d)
        bucket = bucket.clamp(0, self.K - 1)

        # Knot positions at bucket boundaries
        in_d_idx = torch.arange(self.in_d, device=x.device)
        k_lo = knots[in_d_idx, bucket]                     # (N, in_d)
        k_hi = knots[in_d_idx, (bucket + 1).clamp(max=self.K)]  # (N, in_d)

        # Fractional position within bucket
        width = (k_hi - k_lo).clamp(min=1e-8)
        alpha = ((x - k_lo) / width).clamp(0.0, 1.0)      # (N, in_d)

        # Map to uniform[-1, 1]: bucket index + alpha, scaled
        y = (bucket.float() + alpha) / self.K * 2.0 - 1.0  # (N, in_d)
        return y

    def uniformity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        L_unif = MSE(knots, quantile(x, [1/K, ..., (K-1)/K]))

        Pushes knot positions toward empirical quantiles of the current
        input distribution, ensuring equal bucket occupancy.
        """
        # Target quantile positions: [1/K, 2/K, ..., (K-1)/K]
        q = torch.linspace(1.0/self.K, (self.K-1.0)/self.K,
                           self.K - 1, device=x.device)

        # Compute empirical quantiles per input dimension
        # x: (N, in_d) — compute quantiles over N dimension
        with torch.no_grad():
            targets = torch.quantile(x, q, dim=0).T  # (in_d, K-1)

        # MSE between learned knots and empirical quantiles
        interior = self.knots_raw.sort(dim=-1).values
        loss = F.mse_loss(interior, targets)
        return loss * self.unif_weight


class IndexedLinearV2(nn.Module):
    """
    Indexed Linear Layer — faithful to AIWN whitepaper.

    y_j = sum_i table[i, k(a_i), j] * (1-alpha) + table[i, k(a_i)+1, j] * alpha) * a_i + b_j

    Table shape: (d_in, K, d_out) — per-input-unit lookup.
    Each input unit i has its own K weight entries for each output unit j.

    Preceded by LearnedCDFActivation to ensure uniform bucket occupancy.

    Parameters
    ----------
    in_d         : input dimension
    out_d        : output dimension
    K            : number of buckets
    unif_weight  : uniformity loss weight for CDF
    """

    def __init__(self, in_d: int, out_d: int, K: int,
                 unif_weight: float = 0.1):
        super().__init__()
        self.in_d  = in_d
        self.out_d = out_d
        self.K     = K

        # CDF activation — maps input to uniform[-1, 1]
        self.cdf = LearnedCDFActivation(in_d, K, unif_weight)

        # Weight table: (d_in, K, d_out) — per paper Section 3.1
        std = math.sqrt(2.0 / (in_d * K))
        self.table = nn.Parameter(torch.randn(in_d, K, out_d) * std)
        self.bias  = nn.Parameter(torch.zeros(out_d))

        n_table = self.table.numel()
        n_cdf   = self.cdf.knots_raw.numel()
        n_bias  = self.bias.numel()
        print(f"  IndexedLinearV2: ({in_d}, {out_d}, K={K}) | "
              f"table={n_table:,} cdf={n_cdf:,} bias={n_bias} "
              f"total={n_table+n_cdf+n_bias:,}")

    def _forward_eager(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure PyTorch forward pass.
        x: (N, in_d) — already CDF-transformed, in [-1, 1]
        """
        N   = x.shape[0]
        bw  = 2.0 / self.K

        # Bucket index for each input element
        bk  = ((x + 1.0) / bw).long().clamp(0, self.K - 1)  # (N, in_d)
        hi  = (bk + 1).clamp(max=self.K - 1)                  # (N, in_d)

        # Interpolation fraction
        fr  = ((x - (-1.0 + bk.float() * bw)) / bw).clamp(0.0, 1.0)  # (N, in_d)

        # Gather weight slices — table shape: (in_d, K, out_d)
        # Need to index table[i, bk[n,i], :] for each n, i
        in_d_idx = torch.arange(self.in_d, device=x.device)  # (in_d,)

        # Expand for batch indexing
        # bk: (N, in_d), in_d_idx: (in_d,)
        tlo = self.table[in_d_idx, bk]   # (N, in_d, out_d)
        thi = self.table[in_d_idx, hi]   # (N, in_d, out_d)

        # Interpolated weights: (N, in_d, out_d)
        fr_exp = fr.unsqueeze(-1)         # (N, in_d, 1)
        w_eff  = tlo * (1.0 - fr_exp) + thi * fr_exp  # (N, in_d, out_d)

        # Compute output: sum over input dims of w_eff[n,i,j] * x[n,i]
        # x: (N, in_d) → (N, in_d, 1)
        out = (w_eff * x.unsqueeze(-1)).sum(dim=1) + self.bias  # (N, out_d)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        xf    = x.reshape(-1, self.in_d)

        # Apply CDF transform
        xf_cdf = self.cdf(xf)

        # Indexed linear forward
        out = self._forward_eager(xf_cdf)
        return out.reshape(*shape[:-1], self.out_d)

    def uniformity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute CDF uniformity loss on raw (pre-CDF) input.
        Call during training and add to task loss.
        """
        xf = x.reshape(-1, self.in_d)
        return self.cdf.uniformity_loss(xf)

    def flops(self) -> int:
        """Active FLOPs per forward call — same formula as standard linear."""
        return 2 * self.in_d * self.out_d

    def active_bytes(self) -> int:
        """Bytes of table read per forward call (2 slices per input dim)."""
        return 2 * self.in_d * self.out_d * 4

    def gradient_density(self) -> float:
        """Theoretical gradient density — 2/K per paper Section 4.1."""
        return 2.0 / self.K