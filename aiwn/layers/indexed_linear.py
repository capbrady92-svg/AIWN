"""
IndexedLinear — input-indexed weight table with linear interpolation.

The layer maintains a table of shape (K, in_d, out_d).  For each input
element x_i the active weight slice is selected by bucketing x_i into one
of K equally-spaced bins over [-1, 1] and linearly interpolating between
the two nearest bin entries.  The output is the sum over input dimensions:

    out_j = sum_i  lerp(table[lo_i, i, j], table[hi_i, i, j], frac_i) * x_i

A Triton fused kernel is used when available; the pure-PyTorch eager path
is used as fallback (CPU or when Triton is not installed).

Helper functions
----------------
indexed_dims(d, ff, K, n_heads=4)
    Derive equal-parameter indexed dimensions from standard d_model and K.
    Returns (d_idx, ff_idx, h_dn).

indexed_lr(base_lr, K)
    AdamW LR correction for sparse gradient updates: base * sqrt(K/2).
"""

import math
import torch
import torch.nn as nn

# ── Triton kernel (optional) ───────────────────────────────────────────────────
TRITON_OK = False
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fwd(x_ptr, tab_ptr, out_ptr, N, IN, OUT, K, bw,
             sx0, sx1, st0, st1, st2, so0, so1, BLOCK: tl.constexpr):
        n = tl.program_id(0); b = tl.program_id(1)
        j = b * BLOCK + tl.arange(0, BLOCK); msk = j < OUT
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for i in range(IN):
            xi  = tl.load(x_ptr + n * sx0 + i * sx1)
            lo  = tl.minimum(((xi + 1.0) / bw).to(tl.int32), K - 1)
            hi  = tl.minimum(lo + 1, K - 1)
            fr  = tl.clamp((xi - (-1.0 + lo.to(tl.float32) * bw)) / bw, 0.0, 1.0)
            tlo = tl.load(tab_ptr + lo * st0 + i * st1 + j * st2, mask=msk, other=0.0)
            thi = tl.load(tab_ptr + hi * st0 + i * st1 + j * st2, mask=msk, other=0.0)
            acc += (tlo * (1.0 - fr) + thi * fr) * xi
        tl.store(out_ptr + n * so0 + j * so1, acc, mask=msk)

    @triton.jit
    def _bwd_x(go_ptr, x_ptr, tab_ptr, gx_ptr, N, IN, OUT, K, bw,
               sg0, sg1, sx0, sx1, st0, st1, st2, sgx0, sgx1, BLOCK: tl.constexpr):
        n = tl.program_id(0); i = tl.program_id(1)
        xi  = tl.load(x_ptr + n * sx0 + i * sx1)
        lo  = tl.minimum(((xi + 1.0) / bw).to(tl.int32), K - 1)
        hi  = tl.minimum(lo + 1, K - 1)
        fr  = tl.clamp((xi - (-1.0 + lo.to(tl.float32) * bw)) / bw, 0.0, 1.0)
        gxa = 0.0
        for b in range(tl.cdiv(OUT, BLOCK)):
            j   = b * BLOCK + tl.arange(0, BLOCK); msk = j < OUT
            go  = tl.load(go_ptr  + n * sg0 + j * sg1, mask=msk, other=0.0)
            tlo = tl.load(tab_ptr + lo * st0 + i * st1 + j * st2, mask=msk, other=0.0)
            thi = tl.load(tab_ptr + hi * st0 + i * st1 + j * st2, mask=msk, other=0.0)
            W   = tlo * (1.0 - fr) + thi * fr
            gxa = gxa + tl.sum((W + xi * (thi - tlo) / bw) * go, axis=0)
        tl.store(gx_ptr + n * sgx0 + i * sgx1, gxa.to(tl.float32))

    @triton.jit
    def _bwd_t(go_ptr, x_ptr, gt_ptr, N, IN, OUT, K, bw,
               sg0, sg1, sx0, sx1, sgt0, sgt1, sgt2, BLOCK: tl.constexpr):
        n = tl.program_id(0); i = tl.program_id(1); b = tl.program_id(2)
        j   = b * BLOCK + tl.arange(0, BLOCK); msk = j < OUT
        xi  = tl.load(x_ptr + n * sx0 + i * sx1)
        lo  = tl.minimum(((xi + 1.0) / bw).to(tl.int32), K - 1)
        hi  = tl.minimum(lo + 1, K - 1)
        fr  = tl.clamp((xi - (-1.0 + lo.to(tl.float32) * bw)) / bw, 0.0, 1.0)
        go  = tl.load(go_ptr + n * sg0 + j * sg1, mask=msk, other=0.0)
        tl.atomic_add(gt_ptr + lo * sgt0 + i * sgt1 + j * sgt2, xi * (1.0 - fr) * go, mask=msk)
        tl.atomic_add(gt_ptr + hi * sgt0 + i * sgt1 + j * sgt2, xi *        fr  * go, mask=msk)

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, tab, bias, bw):
            N, IN = x.shape; K, _, OUT = tab.shape
            out = torch.zeros(N, OUT, device=x.device, dtype=x.dtype)
            BL  = min(128, triton.next_power_of_2(OUT))
            _fwd[(N, triton.cdiv(OUT, BL))](
                x, tab, out, N, IN, OUT, K, bw,
                x.stride(0), x.stride(1),
                tab.stride(0), tab.stride(1), tab.stride(2),
                out.stride(0), out.stride(1), BLOCK=BL)
            ctx.save_for_backward(x, tab); ctx.bw = bw
            return out + bias

        @staticmethod
        def backward(ctx, go):
            x, tab = ctx.saved_tensors; bw = ctx.bw
            N, IN = x.shape; K, _, OUT = tab.shape
            go = go.contiguous()
            gx = torch.zeros_like(x); gt = torch.zeros_like(tab)
            BL = min(128, triton.next_power_of_2(OUT))
            _bwd_x[(N, IN)](
                go, x, tab, gx, N, IN, OUT, K, bw,
                go.stride(0), go.stride(1), x.stride(0), x.stride(1),
                tab.stride(0), tab.stride(1), tab.stride(2),
                gx.stride(0), gx.stride(1), BLOCK=BL)
            _bwd_t[(N, IN, triton.cdiv(OUT, BL))](
                go, x, gt, N, IN, OUT, K, bw,
                go.stride(0), go.stride(1), x.stride(0), x.stride(1),
                gt.stride(0), gt.stride(1), gt.stride(2), BLOCK=BL)
            return gx, gt, go.sum(0), None

    TRITON_OK = True

except Exception:
    pass  # Triton unavailable — eager fallback used automatically


# ── Module ─────────────────────────────────────────────────────────────────────
class IndexedLinear(nn.Module):
    """
    Input-indexed linear layer with K-bucket interpolated weight table.

    Parameters
    ----------
    in_d  : input dimension
    out_d : output dimension
    K     : number of buckets (must be >= 2; sweet spot 16-32)

    The input domain is assumed to be [-1, 1].  Values outside this range
    are clamped to the nearest bucket boundary.
    """

    def __init__(self, in_d: int, out_d: int, K: int):
        super().__init__()
        assert K >= 2, "K must be at least 2"
        self.K      = K
        self.bw     = 2.0 / K
        self.in_dim  = in_d
        self.out_dim = out_d
        std = math.sqrt(math.sqrt(3.0 / in_d) * (2.0 / K))
        self.table = nn.Parameter(torch.randn(K, in_d, out_d) * std)
        self.bias  = nn.Parameter(torch.zeros(out_d))

    # ── eager (CPU / no-Triton) path ──────────────────────────────────────────
    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        N, IN = x.shape; bw = self.bw
        bk  = ((x + 1) / bw).long().clamp(0, self.K - 1)
        hi  = (bk + 1).clamp(max=self.K - 1)
        fr  = ((x - (-1 + bk.float() * bw)) / bw).clamp(0, 1)
        ie  = torch.arange(IN, device=x.device).unsqueeze(0).expand(N, -1)
        tlo = self.table[bk, ie]
        thi = self.table[hi, ie]
        return ((tlo * (1 - fr.unsqueeze(-1)) +
                 thi *      fr.unsqueeze(-1)) * x.unsqueeze(-1)).sum(1) + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        xf    = x.reshape(-1, self.in_dim).contiguous()
        if TRITON_OK and xf.is_cuda:
            out = _Fn.apply(xf, self.table, self.bias, self.bw)
        else:
            out = self._eager(xf)
        return out.reshape(*shape[:-1], self.out_dim)

    def flops(self) -> int:
        """Active FLOPs per forward call (2 * in * out, same formula as linear)."""
        return 2 * self.in_dim * self.out_dim

    def active_bytes(self) -> int:
        """Bytes of table read per forward call (2 slices per input dim)."""
        return 2 * self.in_dim * self.out_dim * 4


# ── Helpers ────────────────────────────────────────────────────────────────────
def indexed_dims(d: int, ff: int, K: int, n_heads: int = 4):
    """
    Derive equal-parameter indexed dimensions from standard d_model and K.

    Equal-param rule: d_idx ≈ d / sqrt(K), rounded to nearest n_heads multiple.

    Returns
    -------
    d_idx   : indexed hidden dimension
    ff_idx  : indexed FFN intermediate dimension (4 * d_idx)
    h_dn    : number of heads for the indexed down-projection
    """
    s     = math.sqrt(K)
    d_idx = max(n_heads, (int(d / s) // n_heads) * n_heads)
    ff_idx = 4 * d_idx
    h_dn  = max(1, d_idx // max(1, int(round(s))))
    return d_idx, ff_idx, h_dn


def indexed_lr(base_lr: float, K: int) -> float:
    """
    AdamW learning-rate correction for IndexedLinear.

    Indexed gradients are sparse — only 2 of K table slices receive gradient
    per input element.  Scaling LR by sqrt(K/2) restores effective update
    magnitude to be comparable with a standard dense layer.
    """
    return base_lr * math.sqrt(K / 2)


# ── Module-level Triton status (readable by callers) ──────────────────────────
def triton_available() -> bool:
    return TRITON_OK