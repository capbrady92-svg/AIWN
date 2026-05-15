"""
IndexedLinear — fixed Triton kernels.

Forward: Triton fused kernel (fast).
gx backward: Triton _bwd_x kernel (safe — no atomics).
gt backward: PyTorch scatter_add (safe — avoids atomic_add race on RTX 50xx/Blackwell).
"""

import math
import torch
import torch.nn as nn

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
            j_s = tl.minimum(j, OUT - 1)
            tlo = tl.load(tab_ptr + lo * st0 + i * st1 + j_s * st2, mask=msk, other=0.0)
            thi = tl.load(tab_ptr + hi * st0 + i * st1 + j_s * st2, mask=msk, other=0.0)
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
            j     = b * BLOCK + tl.arange(0, BLOCK)
            msk   = j < OUT
            j_s   = tl.minimum(j, OUT - 1)
            go    = tl.load(go_ptr  + n * sg0 + j_s * sg1,  mask=msk, other=0.0)
            tlo   = tl.load(tab_ptr + lo * st0 + i * st1 + j_s * st2, mask=msk, other=0.0)
            thi   = tl.load(tab_ptr + hi * st0 + i * st1 + j_s * st2, mask=msk, other=0.0)
            W     = tlo * (1.0 - fr) + thi * fr
            gxa   = gxa + tl.sum((W + xi * (thi - tlo) / bw) * go, axis=0)
        tl.store(gx_ptr + n * sgx0 + i * sgx1, gxa.to(tl.float32))

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
            ctx.save_for_backward(x, tab)
            ctx.bw = bw
            return out + bias

        @staticmethod
        def backward(ctx, go):
            x, tab = ctx.saved_tensors; bw = ctx.bw
            N, IN = x.shape; K, _, OUT = tab.shape
            go = go.contiguous()

            # gx via Triton _bwd_x — reads only, no atomics, safe
            gx = torch.zeros_like(x)
            BL = min(128, triton.next_power_of_2(OUT))
            _bwd_x[(N, IN)](
                go, x, tab, gx, N, IN, OUT, K, bw,
                go.stride(0), go.stride(1),
                x.stride(0),  x.stride(1),
                tab.stride(0), tab.stride(1), tab.stride(2),
                gx.stride(0), gx.stride(1), BLOCK=BL)

            # gt via vectorized einsum + index_add — no Python loops
            # Avoids atomic_add (Blackwell crash) and Python loops (slow)
            bk = ((x + 1) / bw).long().clamp(0, K - 1)   # (N, IN)
            hi = (bk + 1).clamp(max=K - 1)                # (N, IN)
            fr = ((x - (-1.0 + bk.float() * bw)) / bw).clamp(0.0, 1.0)

            # weights: (N, IN)
            w_lo = x * (1.0 - fr)
            w_hi = x * fr

            # contributions: (N, IN, OUT)
            c_lo = w_lo.unsqueeze(-1) * go.unsqueeze(1)
            c_hi = w_hi.unsqueeze(-1) * go.unsqueeze(1)

            gt = torch.zeros_like(tab)  # (K, IN, OUT)

            # Flatten (N, IN) → single index, scatter into (K*IN, OUT)
            # bk[n,i] indexes into K, i indexes into IN
            # Combined flat index: bk[n,i] * IN + i
            flat_lo = (bk * IN + torch.arange(IN, device=x.device).unsqueeze(0)).reshape(-1)  # (N*IN,)
            flat_hi = (hi * IN + torch.arange(IN, device=x.device).unsqueeze(0)).reshape(-1)  # (N*IN,)

            # c_lo reshaped: (N*IN, OUT)
            c_lo_flat = c_lo.reshape(-1, OUT)
            c_hi_flat = c_hi.reshape(-1, OUT)

            # scatter into gt viewed as (K*IN, OUT)
            gt_flat = gt.reshape(K * IN, OUT)
            gt_flat.scatter_add_(0, flat_lo.unsqueeze(1).expand(-1, OUT), c_lo_flat)
            gt_flat.scatter_add_(0, flat_hi.unsqueeze(1).expand(-1, OUT), c_hi_flat)
            gt = gt_flat.reshape(K, IN, OUT)

            return gx, gt, go.sum(0), None

    TRITON_OK = True

except Exception:
    pass


class IndexedLinear(nn.Module):
    """Input-indexed linear layer with K-bucket interpolated weight table."""

    def __init__(self, in_d: int, out_d: int, K: int):
        super().__init__()
        assert K >= 2
        self.K       = K
        self.bw      = 2.0 / K
        self.in_dim  = in_d
        self.out_dim = out_d
        std = math.sqrt(math.sqrt(3.0 / in_d) * (2.0 / K))
        self.table = nn.Parameter(torch.randn(K, in_d, out_d) * std)
        self.bias  = nn.Parameter(torch.zeros(out_d))

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
        return 2 * self.in_dim * self.out_dim

    def active_bytes(self) -> int:
        return 2 * self.in_dim * self.out_dim * 4


def indexed_dims(d: int, ff: int, K: int, n_heads: int = 4):
    s      = math.sqrt(K)
    d_idx  = max(n_heads, (int(d / s) // n_heads) * n_heads)
    ff_idx = 4 * d_idx
    h_dn   = max(1, d_idx // max(1, int(round(s))))
    return d_idx, ff_idx, h_dn


def indexed_lr(base_lr: float, K: int) -> float:
    return base_lr * math.sqrt(K / 2)


def triton_available() -> bool:
    return TRITON_OK