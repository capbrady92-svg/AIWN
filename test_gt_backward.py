"""
Standalone benchmark for gt backward computation methods.
Tests different approaches for computing IndexedLinear table gradients.

Run from AIWN repo root:
    python test_gt_backward.py
"""

import math
import time
import torch

def bench(fn, n_warmup=5, n_bench=20):
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_bench):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_bench * 1000


def test_gt_methods(N=4096, IN=64, OUT=64, K=16, device='cuda'):
    print(f"\ngt backward benchmark: N={N} IN={IN} OUT={OUT} K={K}")
    print(f"Table size: ({K}, {IN}, {OUT}) = {K*IN*OUT:,} params")
    print(f"{'='*55}")

    x  = torch.randn(N, IN, device=device)
    go = torch.randn(N, OUT, device=device)
    bw = 2.0 / K

    bk = ((x + 1) / bw).long().clamp(0, K - 1)
    hi = (bk + 1).clamp(max=K - 1)
    fr = ((x - (-1.0 + bk.float() * bw)) / bw).clamp(0.0, 1.0)
    w_lo = x * (1.0 - fr)
    w_hi = x * fr

    # Method 1 — K-loop with einsum
    def method_k_loop():
        gt = torch.zeros(K, IN, OUT, device=device)
        for k in range(K):
            mask_lo = (bk == k).float()
            mask_hi = (hi == k).float()
            c = w_lo * mask_lo + w_hi * mask_hi
            gt[k] = torch.einsum('ni,no->io', c, go)
        return gt

    # Method 2 — one-hot + single einsum
    def method_onehot():
        oh_lo = torch.zeros(N, IN, K, device=device)
        oh_hi = torch.zeros(N, IN, K, device=device)
        oh_lo.scatter_(2, bk.unsqueeze(-1), 1.0)
        oh_hi.scatter_(2, hi.unsqueeze(-1), 1.0)
        weighted = w_lo.unsqueeze(-1) * oh_lo + w_hi.unsqueeze(-1) * oh_hi
        return torch.einsum('nik,nj->kij', weighted, go)

    # Method 3 — flat scatter_add
    def method_scatter():
        flat_lo = (bk * IN + torch.arange(IN, device=device).unsqueeze(0)).reshape(-1)
        flat_hi = (hi * IN + torch.arange(IN, device=device).unsqueeze(0)).reshape(-1)
        c_lo = (w_lo.unsqueeze(-1) * go.unsqueeze(1)).reshape(-1, OUT)
        c_hi = (w_hi.unsqueeze(-1) * go.unsqueeze(1)).reshape(-1, OUT)
        gt_flat = torch.zeros(K * IN, OUT, device=device)
        gt_flat.scatter_add_(0, flat_lo.unsqueeze(1).expand(-1, OUT), c_lo)
        gt_flat.scatter_add_(0, flat_hi.unsqueeze(1).expand(-1, OUT), c_hi)
        return gt_flat.reshape(K, IN, OUT)

    # Method 4 — reshape x and go, batch matmul per bucket
    # gt[k] = X_k.T @ go_k where X_k are rows where bk==k or hi==k
    def method_bmm():
        # Build (K, IN, OUT) via batched operations
        # Encode: weighted_x (N, IN) where entry is x*(1-fr) for lo, x*fr for hi
        # Then for each k: gt[k] = (weighted_x * (bk==k or hi==k)).T @ go
        # Vectorize: (K, N, IN) mask then batched matmul
        bk_k = (bk.unsqueeze(0) == torch.arange(K, device=device).view(K,1,1)).float()  # (K,N,IN)
        hi_k = (hi.unsqueeze(0) == torch.arange(K, device=device).view(K,1,1)).float()  # (K,N,IN)
        # weighted contributions per bucket: (K, N, IN)
        contrib = w_lo.unsqueeze(0) * bk_k + w_hi.unsqueeze(0) * hi_k
        # gt[k] = contrib[k].T @ go = einsum('kni,no->kio', contrib, go)
        return torch.einsum('kni,no->kio', contrib, go)

    # Method 5 — Triton _bwd_t_seg if available
    def method_triton_seg():
        try:
            from aiwn.layers.indexed_linear import _bwd_t_seg, TRITON_OK
            import triton
            if not TRITON_OK:
                return None
            tab = torch.zeros(K, IN, OUT, device=device)
            gt  = torch.zeros_like(tab)
            BL  = min(128, triton.next_power_of_2(OUT))
            _bwd_t_seg[(K, IN, triton.cdiv(OUT, BL))](
                go, x, gt, N, IN, OUT, K, bw,
                go.stride(0), go.stride(1),
                x.stride(0),  x.stride(1),
                gt.stride(0), gt.stride(1), gt.stride(2), BLOCK=BL)
            return gt
        except Exception as e:
            return None

    methods = [
        ("K-loop einsum",    method_k_loop),
        ("one-hot einsum",   method_onehot),
        ("flat scatter_add", method_scatter),
        ("batched matmul",   method_bmm),
    ]

    # Verify all methods agree
    ref = method_k_loop()
    print(f"\nCorrectness check (vs K-loop):")
    for name, fn in methods[1:]:
        out = fn()
        err = (out - ref).abs().max().item()
        print(f"  {name:<20}: max_err={err:.2e} {'✓' if err < 1e-4 else '✗'}")

    # Try Triton seg
    seg_result = method_triton_seg()
    if seg_result is not None:
        err = (seg_result - ref).abs().max().item()
        print(f"  {'Triton _bwd_t_seg':<20}: max_err={err:.2e} {'✓' if err < 1e-4 else '✗'}")

    print(f"\nSpeed benchmark:")
    for name, fn in methods:
        ms = bench(fn)
        print(f"  {name:<20}: {ms:.2f}ms")

    if seg_result is not None:
        ms = bench(lambda: method_triton_seg())
        print(f"  {'Triton _bwd_t_seg':<20}: {ms:.2f}ms")

    print(f"\nBest method: pick lowest ms above.")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test at transformer-relevant sizes
    test_gt_methods(N=4096, IN=64,  OUT=64,  K=16,  device=device)
    test_gt_methods(N=4096, IN=64,  OUT=256, K=16,  device=device)
    test_gt_methods(N=4096, IN=256, OUT=64,  K=16,  device=device)
    test_gt_methods(N=4096, IN=32,  OUT=128, K=64,  device=device)


def test_gx_bwd(N=4096, IN=64, OUT=64, K=16, device='cuda'):
    """Benchmark _bwd_x Triton kernel separately."""
    print(f"\ngx backward benchmark: N={N} IN={IN} OUT={OUT} K={K}")
    print(f"{'='*55}")
    try:
        import triton
        from aiwn.layers.indexed_linear import _bwd_x, TRITON_OK
        if not TRITON_OK:
            print("Triton not available")
            return

        x   = torch.randn(N, IN,  device=device)
        go  = torch.randn(N, OUT, device=device)
        tab = torch.randn(K, IN, OUT, device=device)
        gx  = torch.zeros_like(x)
        bw  = 2.0 / K
        BL  = min(128, triton.next_power_of_2(OUT))

        def run_bwd_x():
            gx.zero_()
            _bwd_x[(N, IN)](
                go, x, tab, gx, N, IN, OUT, K, bw,
                go.stride(0), go.stride(1),
                x.stride(0),  x.stride(1),
                tab.stride(0), tab.stride(1), tab.stride(2),
                gx.stride(0), gx.stride(1), BLOCK=BL)

        ms = bench(run_bwd_x)
        print(f"  _bwd_x Triton:  {ms:.2f}ms")
        print(f"  Full bwd estimate: {(ms + 2.0) * 24:.1f}ms for 24 layers")

    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_gx_bwd(N=4096, IN=64,  OUT=64,  K=16, device=device)
    test_gx_bwd(N=4096, IN=64,  OUT=256, K=16, device=device)
    test_gx_bwd(N=4096, IN=256, OUT=64,  K=16, device=device)