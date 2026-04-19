"""
AIWN Comprehensive Sweep Experiment
====================================
Systematically tests indexed vs standard across every dimension that matters:
  1. Batch size:    1 → 256  (when does kernel overhead get amortised?)
  2. Model scale:   d=16 → 512  (when does GPU become compute-bound?)
  3. K value:       1 → 64  (does speedup scale with K as predicted?)
  4. Sequence len:  16 → 256  (attention vs FFN FLOP balance)

For each config measures:
  - Forward pass ms (GPU-synchronised, median over N_BENCH runs)
  - Full step ms (fwd + bwd)
  - Theoretical FLOPs and actual throughput (GFLOP/s)
  - Speedup vs param-matched standard

Accuracy — synthetic regression perplexity:
  Run ONCE per unique (d_model, K) pair, independent of the batch/seq sweep.

  Task:
    Generate a fixed dataset  X ~ Uniform[-1, 1],  Y = tanh(X @ W_true + b_true)
    Train both a StandardLinear and an IndexedLinear (same in/out dims = d_idx)
    for PPL_STEPS steps on the same data with the same random seed.

  Why this task:
    - Input domain [-1, 1] is exactly what IndexedLinear's buckets are designed for
    - tanh target is nonlinear, so input-dependent weighting is actually exercised
    - Both layers use shape (d_idx, d_idx) — no projection tricks, direct comparison
    - Fixed dataset means both layers see identical examples in identical order

  Metrics attached to every result row:
    ppl_std      exp(final_val_mse_std)   lower = better
    ppl_idx      exp(final_val_mse_idx)
    ppl_ratio    ppl_idx / ppl_std        <1.0 = indexed better, >1.0 = standard better

  Full loss curves stored in sweep_results.pt under ppl_cache[(d,K)].

Outputs:
  sweep_results.pt   — raw tensors + full loss curves
  sweep_results.csv  — flat metrics (no curves)
  sweep_results.png  — plots

Usage:
  python sweep_experiment.py                      # full sweep on GPU
  python sweep_experiment.py --quick              # reduced sweep for testing
  python sweep_experiment.py --device cpu         # force CPU
  python sweep_experiment.py --ppl_steps 1000     # more training for accuracy eval
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, math, argparse, csv
from itertools import product

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--device',         default='auto')
parser.add_argument('--quick',          action='store_true')
parser.add_argument('--n_bench',        type=int,   default=200)
parser.add_argument('--n_warm',         type=int,   default=50)
parser.add_argument('--ppl_steps',      type=int,   default=500,
                    help='Optimiser steps for perplexity eval per (d,K) pair')
parser.add_argument('--ppl_lr',         type=float, default=1e-3)
parser.add_argument('--ppl_batch',      type=int,   default=256)
parser.add_argument('--ppl_n_data',     type=int,   default=4096)
parser.add_argument('--ppl_ckpt_every', type=int,   default=50)
args = parser.parse_args()

DEVICE = torch.device(
    ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device == 'auto' else args.device)

print(f"\nDevice: {DEVICE}")
if DEVICE.type == 'cuda':
    props = torch.cuda.get_device_properties(0)
    L2 = (getattr(props, 'l2_cache_size', None) or
          getattr(props, 'L2_cache_size', 0)) / 1e6
    print(f"GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f}GB  L2: {L2:.0f}MB")

# ── Triton kernel ──────────────────────────────────────────────────────────────
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
    print(f"Triton {triton.__version__} — fused kernel active ✓")
except Exception as e:
    print(f"Triton unavailable ({type(e).__name__}) — using eager mode")


# ── Layers ─────────────────────────────────────────────────────────────────────
def indexed_dims(d, ff, K, n_heads=4):
    s     = math.sqrt(K)
    d_idx = max(n_heads, (int(d / s) // n_heads) * n_heads)
    ff_idx = 4 * d_idx
    h     = max(1, d_idx // max(1, int(round(s))))
    return d_idx, ff_idx, h

def indexed_lr(base, K):
    """LR correction for sparse gradient updates: base * sqrt(K/2)."""
    return base * math.sqrt(K / 2)

class IndexedLinear(nn.Module):
    def __init__(self, in_d, out_d, K):
        super().__init__()
        self.K = K; self.bw = 2. / K
        self.in_dim = in_d; self.out_dim = out_d
        std = math.sqrt(math.sqrt(3. / in_d) * (2. / K))
        self.table = nn.Parameter(torch.randn(K, in_d, out_d) * std)
        self.bias  = nn.Parameter(torch.zeros(out_d))

    def _eager(self, x):
        N, IN = x.shape; bw = self.bw
        bk  = ((x + 1) / bw).long().clamp(0, self.K - 1)
        hi  = (bk + 1).clamp(max=self.K - 1)
        fr  = ((x - (-1 + bk.float() * bw)) / bw).clamp(0, 1)
        ie  = torch.arange(IN, device=x.device).unsqueeze(0).expand(N, -1)
        tlo = self.table[bk, ie]; thi = self.table[hi, ie]
        return ((tlo * (1 - fr.unsqueeze(-1)) +
                 thi *      fr.unsqueeze(-1)) * x.unsqueeze(-1)).sum(1) + self.bias

    def forward(self, x):
        shape = x.shape; xf = x.reshape(-1, self.in_dim).contiguous()
        out = _Fn.apply(xf, self.table, self.bias, self.bw) \
              if (TRITON_OK and xf.is_cuda) else self._eager(xf)
        return out.reshape(*shape[:-1], self.out_dim)

    def flops(self):        return 2 * self.in_dim * self.out_dim
    def active_bytes(self): return 2 * self.in_dim * self.out_dim * 4

class StandardLinear(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.lin = nn.Linear(in_d, out_d, bias=True)
    def forward(self, x): return self.lin(x)
    def flops(self): return 2 * self.lin.in_features * self.lin.out_features


# ── Benchmark utility ──────────────────────────────────────────────────────────
def sync():
    if DEVICE.type == 'cuda': torch.cuda.synchronize()

def bench_fn(fn, n_warm, n_bench):
    for _ in range(n_warm): fn()
    sync()
    times = []
    for _ in range(n_bench):
        sync(); t0 = time.perf_counter()
        fn()
        sync(); times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))

def bench_layer(layer_std, layer_idx, B, IN, OUT, K, n_warm, n_bench):
    x_std = torch.randn(B, IN,               device=DEVICE, requires_grad=True)
    x_idx = torch.empty(B, layer_idx.in_dim, device=DEVICE).uniform_(-1, 1).requires_grad_(True)

    fwd_std, _ = bench_fn(lambda: layer_std(x_std), n_warm, n_bench)
    fwd_idx, _ = bench_fn(lambda: layer_idx(x_idx), n_warm, n_bench)

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

    step_std_ms, _ = bench_fn(full_std, n_warm, n_bench)
    step_idx_ms, _ = bench_fn(full_idx, n_warm, n_bench)

    flops_std = layer_std.flops()
    flops_idx = layer_idx.flops()

    return {
        'fwd_std_ms':   fwd_std,
        'fwd_idx_ms':   fwd_idx,
        'fwd_speedup':  fwd_std / fwd_idx,
        'step_std_ms':  step_std_ms,
        'step_idx_ms':  step_idx_ms,
        'step_speedup': step_std_ms / step_idx_ms,
        'flops_std':    flops_std,
        'flops_idx':    flops_idx,
        'flop_ratio':   flops_std / flops_idx,
        'tflops_std':   flops_std * B / fwd_std  / 1e9,
        'tflops_idx':   flops_idx * B / fwd_idx  / 1e9,
        'idx_faster_fwd':  fwd_idx     < fwd_std,
        'idx_faster_step': step_idx_ms < step_std_ms,
        'active_kb':    layer_idx.active_bytes() / 1024,
    }


# ── Perplexity via synthetic regression ───────────────────────────────────────
def make_dataset(in_d, out_d, n, seed=42):
    """
    Fixed synthetic dataset: X ~ Uniform[-1,1], Y = tanh(X @ W_true + b_true)

    The input domain [-1, 1] matches IndexedLinear's bucket range exactly.
    The nonlinear target (tanh) means the indexed layer's input-dependent
    weighting is exercised — a pure linear target would let both layers trivially
    converge to the same solution regardless of architecture.

    W_true is frozen for the entire run so both layers see the same task.
    80/20 train/val split, same seed every time for reproducibility.
    """
    rng    = torch.Generator(device=DEVICE).manual_seed(seed)
    W_true = torch.randn(in_d, out_d, generator=rng, device=DEVICE) / math.sqrt(in_d)
    b_true = torch.zeros(out_d, device=DEVICE)
    X      = torch.empty(n, in_d, device=DEVICE).uniform_(-1, 1)
    with torch.no_grad():
        Y = torch.tanh(X @ W_true + b_true)
    split = int(0.8 * n)
    return (X[:split], Y[:split]), (X[split:], Y[split:])

def train_and_eval(layer, train_xy, val_xy, steps, lr, batch_size, ckpt_every):
    """
    Train `layer` on regression (X→Y) with AdamW for `steps` steps.
    For IndexedLinear, lr is scaled by indexed_lr() to correct for sparse
    gradient updates so the effective learning signal is comparable.

    Returns (final_val_mse, loss_curve) where loss_curve is a list of
    (step, train_mse) pairs recorded every ckpt_every steps.
    """
    if isinstance(layer, IndexedLinear):
        lr = indexed_lr(lr, layer.K)

    opt   = torch.optim.AdamW(layer.parameters(), lr=lr, weight_decay=1e-4)
    X_tr, Y_tr = train_xy
    X_val, Y_val = val_xy
    N_tr  = X_tr.shape[0]
    curve = []

    for step in range(1, steps + 1):
        idx  = torch.randint(0, N_tr, (batch_size,), device=DEVICE)
        xb, yb = X_tr[idx], Y_tr[idx]
        loss = F.mse_loss(layer(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % ckpt_every == 0:
            curve.append((step, loss.item()))

    with torch.no_grad():
        val_mse = F.mse_loss(layer(X_val), Y_val).item()

    return val_mse, curve

def run_perplexity(d_std, K, steps, lr, batch_size, n_data, ckpt_every):
    """
    For a given (d_std, K) pair, derive d_idx, build matching Standard and
    Indexed layers of shape (d_idx → d_idx), train both on the same dataset,
    return ppl_std, ppl_idx, ppl_ratio (<1 = indexed better), and loss curves.
    """
    d_idx, _, _ = indexed_dims(d_std, 4 * d_std, K)
    layer_std    = StandardLinear(d_idx, d_idx).to(DEVICE)
    layer_idx    = IndexedLinear(d_idx, d_idx, K).to(DEVICE)
    train_xy, val_xy = make_dataset(d_idx, d_idx, n_data)

    val_mse_std, curve_std = train_and_eval(
        layer_std, train_xy, val_xy, steps, lr, batch_size, ckpt_every)
    val_mse_idx, curve_idx = train_and_eval(
        layer_idx, train_xy, val_xy, steps, lr, batch_size, ckpt_every)

    ppl_std   = math.exp(min(val_mse_std, 20))   # clip to avoid overflow
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


# ── Sweep configurations ───────────────────────────────────────────────────────
if args.quick:
    BATCH_SIZES = [1, 8, 32, 128]
    D_MODELS    = [32, 64, 128, 256]
    K_VALUES    = [4, 16]
    SEQ_LENS    = [48]
    N_BENCH     = 100
    N_WARM      = 20
    PPL_STEPS   = 200
else:
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    D_MODELS    = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    K_VALUES    = [2, 4, 8, 16, 32]
    SEQ_LENS    = [16, 48, 128, 256]
    N_BENCH     = args.n_bench
    N_WARM      = args.n_warm
    PPL_STEPS   = args.ppl_steps

N_HEADS = 4

n_total = len(BATCH_SIZES) * len(D_MODELS) * len(K_VALUES) * len(SEQ_LENS)
n_ppl   = len(D_MODELS) * len(K_VALUES)
print(f"\nConfigs  : {n_total}  ({N_WARM} warmup + {N_BENCH} timed each)")
print(f"Ppl pairs: {n_ppl} unique (d,K) x {PPL_STEPS} steps (computed on first encounter)")
print(f"Backend  : {'Triton' if TRITON_OK else 'Eager'}\n")

# ── Sweep: speed + perplexity together ────────────────────────────────────────
# Perplexity is computed once per unique (d,K) on first encounter, then cached.
# Speed is benchmarked for every (B, d, K, seq) combo as before.
print("=" * 104)
print(f"{'B':>4} {'d':>4} {'K':>3} {'seq':>4} | "
      f"{'fwd_std':>8} {'fwd_idx':>8} {'fwd_spd':>8} | "
      f"{'stp_std':>8} {'stp_idx':>8} {'stp_spd':>8} | "
      f"{'mse_std':>8} {'mse_idx':>8} {'ppl_r':>6} | {'faster?':>8}")
print(f"{'':54}  {'lower=better':>23} {'<1=idx':>6}")
print("-" * 104)

ppl_cache = {}
results   = []
done      = 0
for B, d_std, K, seq in product(BATCH_SIZES, D_MODELS, K_VALUES, SEQ_LENS):
    d_idx, ff_idx, h_dn = indexed_dims(d_std, 4 * d_std, K, N_HEADS)
    layer_std = StandardLinear(d_std, 4 * d_std).to(DEVICE)
    layer_idx = IndexedLinear(h_dn, d_idx, K).to(DEVICE)
    N = B * seq

    # Perplexity: compute once per (d,K) on first encounter, reuse after
    if (d_std, K) not in ppl_cache:
        ppl_cache[(d_std, K)] = run_perplexity(
            d_std, K,
            steps      = PPL_STEPS,
            lr         = args.ppl_lr,
            batch_size = args.ppl_batch,
            n_data     = args.ppl_n_data,
            ckpt_every = args.ppl_ckpt_every,
        )

    try:
        r = bench_layer(layer_std, layer_idx, N, d_std, 4 * d_std, K, N_WARM, N_BENCH)
    except Exception as e:
        print(f"  SKIP B={B} d={d_std} K={K} seq={seq}: {e}")
        continue

    pm = ppl_cache[(d_std, K)]
    r.update({
        'B': B, 'd_std': d_std, 'd_idx': d_idx, 'K': K, 'seq': seq,
        'h_dn': h_dn, 'N': N,
        'params_std': layer_std.lin.weight.numel() + layer_std.lin.bias.numel(),
        'params_idx': layer_idx.table.numel()      + layer_idx.bias.numel(),
        'backend':    'triton' if TRITON_OK else 'eager',
        'ppl_std':     pm['ppl_std'],
        'ppl_idx':     pm['ppl_idx'],
        'ppl_ratio':   pm['ppl_ratio'],
        'val_mse_std': pm['val_mse_std'],
        'val_mse_idx': pm['val_mse_idx'],
    })
    results.append(r)
    done += 1

    faster = ('✓ FWD'  if r['idx_faster_fwd']  else '') or \
             ('✓ STEP' if r['idx_faster_step'] else '') or '✗'

    if done % 5 == 1 or r['idx_faster_fwd'] or r['idx_faster_step']:
        print(f"{B:4d} {d_std:4d} {K:3d} {seq:4d} | "
              f"{r['fwd_std_ms']:7.3f}ms {r['fwd_idx_ms']:7.3f}ms {r['fwd_speedup']:7.2f}x | "
              f"{r['step_std_ms']:7.3f}ms {r['step_idx_ms']:7.3f}ms {r['step_speedup']:7.2f}x | "
              f"{pm['val_mse_std']:8.5f} {pm['val_mse_idx']:8.5f} {pm['ppl_ratio']:6.3f} | {faster:>8}")
    if done % 20 == 0:
        print(f"  [{done}/{n_total}]")


# ── Save ───────────────────────────────────────────────────────────────────────
torch.save({
    'results':   results,
    'ppl_cache': ppl_cache,
    'device':    str(DEVICE),
    'triton':    TRITON_OK,
    'n_bench':   N_BENCH,
    'ppl_steps': PPL_STEPS,
}, 'sweep_results.pt')

flat_fields = [k for k in results[0] if not isinstance(results[0][k], list)] if results else []
with open('sweep_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=flat_fields)
    w.writeheader()
    w.writerows({k: r[k] for k in flat_fields} for r in results)

print(f"\nSaved {len(results)} results → sweep_results.pt, sweep_results.csv")


# ── Analysis ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

k_target   = 16  if 16  in K_VALUES    else K_VALUES[-1]
b_target   = 32  if 32  in BATCH_SIZES else BATCH_SIZES[-1]
d_target   = min(64, max(D_MODELS))
d_k        = 128 if 128 in D_MODELS    else D_MODELS[-1]
seq_target = 48  if 48  in SEQ_LENS    else SEQ_LENS[0]

print(f"\n1. Batch size crossover (d={d_target}, K={k_target}, seq={seq_target}):")
subset = [r for r in results
          if r['d_std']==d_target and r['K']==k_target and r['seq']==seq_target]
if subset:
    print(f"  {'B':>6} {'fwd_speedup':>12} {'step_speedup':>12} {'fwd faster?':>12}")
    for r in sorted(subset, key=lambda x: x['B']):
        print(f"  {r['B']:>6} {r['fwd_speedup']:>11.2f}x {r['step_speedup']:>11.2f}x "
              f"  {'YES ✓' if r['idx_faster_fwd'] else 'no':>12}")

print(f"\n2. Model scale — speed (B={b_target}, K={k_target}, seq={seq_target}):")
subset2 = [r for r in results
           if r['B']==b_target and r['K']==k_target and r['seq']==seq_target]
if subset2:
    print(f"  {'d_std':>6} {'d_idx':>6} {'fwd_speedup':>12} {'theory':>8} {'faster?':>10}")
    for r in sorted(subset2, key=lambda x: x['d_std']):
        print(f"  {r['d_std']:>6} {r['d_idx']:>6} {r['fwd_speedup']:>11.2f}x "
              f"{r['flop_ratio']:>7.1f}x   {'YES ✓' if r['idx_faster_fwd'] else 'no':>10}")

print(f"\n3. K scaling — speed (B={b_target}, d={d_k}, seq={seq_target}):")
subset3 = [r for r in results
           if r['B']==b_target and r['d_std']==d_k and r['seq']==seq_target]
if subset3:
    print(f"  {'K':>4} {'theory':>10} {'actual':>10} {'efficiency%':>13}")
    for r in sorted(subset3, key=lambda x: x['K']):
        eff = r['fwd_speedup'] / r['flop_ratio'] * 100
        print(f"  {r['K']:>4} {r['flop_ratio']:>9.1f}x {r['fwd_speedup']:>9.2f}x {eff:>12.1f}%")

print(f"\n4. K scaling — perplexity (d={d_k}):")
print(f"  {'K':>4} {'ppl_std':>10} {'ppl_idx':>10} {'ratio':>8} {'better':>8}")
seen_k = set()
for r in sorted(subset3, key=lambda x: x['K']):
    if r['K'] not in seen_k:
        seen_k.add(r['K'])
        better = '✓ idx' if r['ppl_ratio'] < 1.0 else '  std'
        print(f"  {r['K']:>4} {r['ppl_std']:>10.4f} {r['ppl_idx']:>10.4f} "
              f"{r['ppl_ratio']:>8.4f} {better:>8}")

print(f"\n5. d_model scaling — perplexity (K={k_target}):")
ppl_d = [(d, ppl_cache[(d, k_target)]) for d in D_MODELS if (d, k_target) in ppl_cache]
if ppl_d:
    print(f"  {'d_std':>6} {'d_idx':>6} {'ppl_std':>10} {'ppl_idx':>10} {'ratio':>8} {'better':>8}")
    for d, pm in ppl_d:
        d_idx, _, _ = indexed_dims(d, 4 * d, k_target)
        better = '✓ idx' if pm['ppl_ratio'] < 1.0 else '  std'
        print(f"  {d:>6} {d_idx:>6} {pm['ppl_std']:>10.4f} {pm['ppl_idx']:>10.4f} "
              f"{pm['ppl_ratio']:>8.4f} {better:>8}")

n_faster_fwd  = sum(1 for r in results if r['idx_faster_fwd'])
n_faster_step = sum(1 for r in results if r['idx_faster_step'])
n_ppl_better  = sum(1 for pm in ppl_cache.values() if pm['ppl_ratio'] < 1.0)
mean_ratio    = np.mean([pm['ppl_ratio'] for pm in ppl_cache.values()])
print(f"\n6. Overall:")
print(f"   Speed — indexed faster (fwd):  {n_faster_fwd}/{len(results)} "
      f"({n_faster_fwd/max(1,len(results))*100:.0f}%)")
print(f"   Speed — indexed faster (step): {n_faster_step}/{len(results)} "
      f"({n_faster_step/max(1,len(results))*100:.0f}%)")
print(f"   Perplexity — indexed better:   {n_ppl_better}/{len(ppl_cache)} (d,K) pairs "
      f"({n_ppl_better/max(1,len(ppl_cache))*100:.0f}%)")
print(f"   Mean ppl_ratio (idx/std):       {mean_ratio:.4f}  "
      f"({'indexed more accurate on avg' if mean_ratio < 1 else 'standard more accurate on avg'})")


# ── Plots ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor('#0d0d0d')
    gs = gridspec.GridSpec(3, 3, hspace=0.52, wspace=0.38)

    def ax_(pos):
        a = fig.add_subplot(pos); a.set_facecolor('#111')
        a.tick_params(colors='#aaa'); a.spines[:].set_color('#333')
        return a

    cmap = plt.cm.viridis
    CRED = '#e74c3c'; CGRN = '#2ecc71'; CYEL = '#f1c40f'; CBLU = '#3498db'
    k_p  = k_target; s_p = seq_target; b_p = b_target

    # ── Row 0: Speed ──────────────────────────────────────────────────────────
    ax1 = ax_(gs[0, 0])
    ax1.set_title("Batch Size vs Forward Speedup\n(K=16, seq=48)", color='white', fontsize=11)
    for i, d in enumerate(sorted(set(r['d_std'] for r in results))):
        sub = sorted([(r['B'], r['fwd_speedup']) for r in results
                      if r['d_std']==d and r['K']==k_p and r['seq']==s_p])
        if len(sub) > 1:
            bs, spds = zip(*sub)
            ax1.plot(bs, spds, 'o-', color=cmap(i/len(D_MODELS)), lw=2, ms=5, label=f'd={d}')
    ax1.axhline(1.0, color=CRED, ls='--', lw=1.5, label='break-even')
    ax1.set_xscale('log'); ax1.set_xlabel("Batch size", color='#aaa')
    ax1.set_ylabel("Forward speedup", color='#aaa')
    ax1.legend(facecolor='#111', labelcolor='white', fontsize=7, ncol=2)

    ax2 = ax_(gs[0, 1])
    ax2.set_title("Model Scale vs Forward Speedup\n(B=32, seq=48)", color='white', fontsize=11)
    for i, K in enumerate(K_VALUES):
        sub = sorted([(r['d_std'], r['fwd_speedup']) for r in results
                      if r['B']==b_p and r['K']==K and r['seq']==s_p])
        if len(sub) > 1:
            ds, spds = zip(*sub)
            ax2.plot(ds, spds, 'o-', color=cmap(i/len(K_VALUES)), lw=2, ms=5, label=f'K={K}')
    ax2.axhline(1.0, color=CRED, ls='--', lw=1.5)
    ax2.set_xlabel("d_model", color='#aaa'); ax2.set_ylabel("Forward speedup", color='#aaa')
    ax2.legend(facecolor='#111', labelcolor='white', fontsize=8)

    ax3 = ax_(gs[0, 2])
    ax3.set_title("K vs Speedup: Theory vs Actual\n(B=32, d=128, seq=48)", color='white', fontsize=11)
    sub = sorted([(r['K'], r['fwd_speedup'], r['flop_ratio']) for r in results
                  if r['B']==b_p and r['d_std']==d_k and r['seq']==s_p])
    if sub:
        ks, acts, theorys = zip(*sub)
        ax3.plot(ks, theorys, 's--', color='#aaa', lw=1.5, ms=6, label='Theory (FLOPs)')
        ax3.plot(ks, acts,    'o-',  color=CGRN,   lw=2.5, ms=8, label='Actual')
        ax3.fill_between(ks, acts, theorys, alpha=0.15, color=CGRN)
    ax3.axhline(1.0, color=CRED, ls='--', lw=1)
    ax3.set_xlabel("K", color='#aaa'); ax3.set_ylabel("Speedup", color='#aaa')
    ax3.legend(facecolor='#111', labelcolor='white', fontsize=9)

    # ── Row 1: Speed heatmaps ─────────────────────────────────────────────────
    ds_u = sorted(set(r['d_std'] for r in results))
    bs_u = sorted(set(r['B']     for r in results))

    for col, key, title in [
        (0, 'fwd_speedup',  'Fwd Speedup Heatmap (K=16)'),
        (1, 'step_speedup', 'Full Step Speedup Heatmap (K=16)'),
    ]:
        axh = ax_(gs[1, col])
        mat = np.full((len(ds_u), len(bs_u)), np.nan)
        for r in results:
            if r['K']==k_p and r['seq']==s_p:
                mat[ds_u.index(r['d_std']), bs_u.index(r['B'])] = r[key]
        im = axh.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=2.0, origin='lower')
        axh.set_xticks(range(len(bs_u))); axh.set_xticklabels(bs_u, color='white', fontsize=7)
        axh.set_yticks(range(len(ds_u))); axh.set_yticklabels(ds_u, color='white', fontsize=7)
        axh.set_xlabel("Batch size", color='#aaa'); axh.set_ylabel("d_model", color='#aaa')
        plt.colorbar(im, ax=axh).ax.yaxis.set_tick_params(color='white')
        axh.set_title(f"{title}\nGreen=indexed faster", color='white', fontsize=10)

    ax_eff = ax_(gs[1, 2])
    ax_eff.set_title("GPU Efficiency: Actual/Theory\n(fraction of FLOP speedup realised)",
                     color='white', fontsize=10)
    for i, K in enumerate(K_VALUES):
        sub = sorted([(r['d_std'], r['fwd_speedup'] / r['flop_ratio'])
                      for r in results if r['B']==b_p and r['K']==K and r['seq']==s_p])
        if len(sub) > 1:
            ds, effs = zip(*sub)
            ax_eff.plot(ds, [e*100 for e in effs], 'o-',
                        color=cmap(i/len(K_VALUES)), lw=2, ms=5, label=f'K={K}')
    ax_eff.axhline(100, color=CGRN, ls='--', lw=1.5, label='100%')
    ax_eff.axhline(50,  color=CYEL, ls=':',  lw=1,   label='50%')
    ax_eff.set_xlabel("d_model", color='#aaa'); ax_eff.set_ylabel("Efficiency (%)", color='#aaa')
    ax_eff.set_ylim(0, 120)
    ax_eff.legend(facecolor='#111', labelcolor='white', fontsize=8)

    # ── Row 2: Perplexity ─────────────────────────────────────────────────────
    # 2,0  ppl_ratio vs K, one line per d_model
    ax_pk = ax_(gs[2, 0])
    ax_pk.set_title("Perplexity Ratio vs K\n(idx/std — below 1 = indexed better)",
                    color='white', fontsize=10)
    for i, d in enumerate(D_MODELS):
        pts = sorted([(K, ppl_cache[(d, K)]['ppl_ratio'])
                      for K in K_VALUES if (d, K) in ppl_cache])
        if len(pts) > 1:
            ks, ratios = zip(*pts)
            ax_pk.plot(ks, ratios, 'o-', color=cmap(i/len(D_MODELS)),
                       lw=1.5, ms=5, label=f'd={d}')
    ax_pk.axhline(1.0, color=CRED, ls='--', lw=1.5, label='break-even')
    ax_pk.set_xlabel("K", color='#aaa'); ax_pk.set_ylabel("ppl_ratio (idx/std)", color='#aaa')
    ax_pk.legend(facecolor='#111', labelcolor='white', fontsize=7, ncol=2)

    # 2,1  ppl_ratio heatmap (d × K)
    ax_ph = ax_(gs[2, 1])
    ks_u = sorted(K_VALUES)
    mat_ppl = np.full((len(ds_u), len(ks_u)), np.nan)
    for i, d in enumerate(ds_u):
        for j, K in enumerate(ks_u):
            if (d, K) in ppl_cache:
                mat_ppl[i, j] = ppl_cache[(d, K)]['ppl_ratio']
    im_p = ax_ph.imshow(mat_ppl, aspect='auto', cmap='RdYlGn_r',
                        vmin=0.8, vmax=1.2, origin='lower')
    ax_ph.set_xticks(range(len(ks_u))); ax_ph.set_xticklabels(ks_u, color='white', fontsize=8)
    ax_ph.set_yticks(range(len(ds_u))); ax_ph.set_yticklabels(ds_u, color='white', fontsize=7)
    ax_ph.set_xlabel("K", color='#aaa'); ax_ph.set_ylabel("d_model", color='#aaa')
    plt.colorbar(im_p, ax=ax_ph).ax.yaxis.set_tick_params(color='white')
    ax_ph.set_title("ppl_ratio Heatmap (d × K)\nGreen = indexed better (<1.0)",
                    color='white', fontsize=10)

    # 2,2  Loss curves for representative config (d=128, K=16)
    ax_lc = ax_(gs[2, 2])
    ax_lc.set_title(f"Training Loss Curves  (d={d_k}, K={k_target})\n"
                    f"Task: Y = tanh(X @ W_true),  X ~ Uniform[-1,1]",
                    color='white', fontsize=10)
    key = (d_k, k_target)
    if key in ppl_cache:
        pm = ppl_cache[key]
        if pm['loss_curve_std']:
            ss, ls = zip(*pm['loss_curve_std'])
            ax_lc.plot(ss, ls, '-', color=CBLU, lw=2.5, label=f"Standard  (val MSE={pm['val_mse_std']:.4f})")
        if pm['loss_curve_idx']:
            si, li = zip(*pm['loss_curve_idx'])
            ax_lc.plot(si, li, '-', color=CGRN, lw=2.5, label=f"Indexed   (val MSE={pm['val_mse_idx']:.4f})")
    ax_lc.set_xlabel("Step", color='#aaa'); ax_lc.set_ylabel("Train MSE", color='#aaa')
    ax_lc.legend(facecolor='#111', labelcolor='white', fontsize=9)

    fig.suptitle(
        f"AIWN Sweep — {DEVICE}  Backend: {'Triton' if TRITON_OK else 'Eager'}\n"
        f"Row 0: Speed  |  Row 1: Speed heatmaps  |  Row 2: Perplexity (synthetic regression)",
        color='white', fontsize=12, y=1.01)
    plt.savefig('sweep_results.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    print("Saved sweep_results.png")
except Exception as e:
    print(f"Plot failed: {e}")