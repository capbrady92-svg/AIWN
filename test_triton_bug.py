"""
Minimal Triton SM120 bug reproduction.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(out_ptr, x_ptr, N, K, bw, BLOCK: tl.constexpr):
    k = tl.program_id(0)
    i = tl.program_id(1)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for n in range(N):
        xi       = tl.load(x_ptr + n * K + i)
        lo       = tl.minimum(((xi + 1.0) / bw).to(tl.int32), K - 1)
        lo_match = (lo == k).to(tl.float32)
        acc      = acc + lo_match * xi
    j = tl.arange(0, BLOCK)
    tl.store(out_ptr + k * K * BLOCK + i * BLOCK + j, acc)


def reference_cpu(x, K, bw):
    N, IN = x.shape
    out = torch.zeros(K, IN, 4)
    for n in range(N):
        for i in range(IN):
            xi = x[n, i].item()
            lo = min(int((xi + 1.0) / bw), K - 1)
            out[lo, i, :] += xi
    return out


if __name__ == "__main__":
    device = 'cuda'
    torch.manual_seed(42)
    N, IN, K = 16, 4, 4
    bw = 2.0 / K
    BLOCK = 4

    x   = torch.randn(N, IN, device=device)
    out = torch.zeros(K, IN, BLOCK, device=device)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    test_kernel[(K, IN)](out, x, N, K, bw, BLOCK=BLOCK)
    torch.cuda.synchronize()

    ref = reference_cpu(x.cpu(), K, bw)
    err = (out.cpu() - ref).abs().max().item()

    print(f"Max error: {err:.4e} {'FAIL' if err > 1e-3 else 'PASS'}\n")

    for k in range(K):
        diff = (out[k].cpu() - ref[k]).abs().max().item()
        print(f"Bucket {k}: max_err={diff:.4e} {'✗' if diff > 1e-3 else '✓'}")
        if diff > 1e-3:
            print(f"  kernel: {out[k,0].cpu().tolist()}")
            print(f"  ref:    {ref[k,0].tolist()}")

    # Now find the PTX
    import os, glob
    cache_dirs = glob.glob(os.path.expanduser("~/.triton/cache/*/"))
    if cache_dirs:
        latest = max(cache_dirs, key=os.path.getmtime)
        ptx_files = glob.glob(latest + "*.ptx")
        if ptx_files:
            print(f"\nPTX file: {ptx_files[0]}")
            with open(ptx_files[0]) as f:
                ptx = f.read()
            # Find the int32 cast and comparison
            lines = ptx.split('\n')
            for i, line in enumerate(lines):
                if any(x in line for x in ['cvt.rzi', 'setp.eq', 'selp', 'lo_match']):
                    print(f"  {i}: {line}")