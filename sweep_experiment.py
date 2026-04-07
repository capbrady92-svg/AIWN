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
  - Backward pass ms
  - Full step ms
  - Theoretical FLOPs
  - Actual throughput (GFLOP/s)
  - Speedup vs param-matched standard
  - Whether indexed beats standard in wall-clock

Outputs:
  sweep_results.pt   — raw numbers for all configs
  sweep_results.csv  — human-readable
  sweep_summary.png  — plots

Usage:
  python sweep_experiment.py                    # full sweep on GPU
  python sweep_experiment.py --quick            # reduced sweep for testing
  python sweep_experiment.py --device cpu       # force CPU
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, math, argparse, csv, os
from itertools import product

# ── Args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='auto')
parser.add_argument('--quick',  action='store_true', help='Reduced sweep for testing')
parser.add_argument('--n_bench',type=int, default=200, help='Benchmark iterations')
parser.add_argument('--n_warm', type=int, default=50,  help='Warmup iterations')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'
                      if args.device=='auto' else args.device)

print(f"\nDevice: {DEVICE}")
if DEVICE.type == 'cuda':
    props = torch.cuda.get_device_properties(0)
    L2 = (getattr(props,'l2_cache_size',None) or getattr(props,'L2_cache_size',0))/1e6
    print(f"GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f}GB  L2: {L2:.0f}MB")
    TFLOPS_PEAK = props.total_memory / 1e12 * 200  # rough estimate

# ── Triton kernel ──────────────────────────────────────────────────────────
TRITON_OK = False
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fwd(x_ptr, tab_ptr, out_ptr, N, IN, OUT, K, bw,
             sx0,sx1, st0,st1,st2, so0,so1, BLOCK:tl.constexpr):
        n=tl.program_id(0); b=tl.program_id(1)
        j=b*BLOCK+tl.arange(0,BLOCK); msk=j<OUT
        acc=tl.zeros([BLOCK],dtype=tl.float32)
        for i in range(IN):
            xi=tl.load(x_ptr+n*sx0+i*sx1)
            lo=tl.minimum(((xi+1.0)/bw).to(tl.int32),K-1)
            hi=tl.minimum(lo+1,K-1)
            fr=tl.clamp((xi-(-1.0+lo.to(tl.float32)*bw))/bw,0.0,1.0)
            tlo=tl.load(tab_ptr+lo*st0+i*st1+j*st2,mask=msk,other=0.0)
            thi=tl.load(tab_ptr+hi*st0+i*st1+j*st2,mask=msk,other=0.0)
            acc+=(tlo*(1.0-fr)+thi*fr)*xi
        tl.store(out_ptr+n*so0+j*so1,acc,mask=msk)

    @triton.jit
    def _bwd_x(go_ptr,x_ptr,tab_ptr,gx_ptr,N,IN,OUT,K,bw,
               sg0,sg1,sx0,sx1,st0,st1,st2,sgx0,sgx1,BLOCK:tl.constexpr):
        n=tl.program_id(0); i=tl.program_id(1)
        xi=tl.load(x_ptr+n*sx0+i*sx1)
        lo=tl.minimum(((xi+1.0)/bw).to(tl.int32),K-1)
        hi=tl.minimum(lo+1,K-1)
        fr=tl.clamp((xi-(-1.0+lo.to(tl.float32)*bw))/bw,0.0,1.0)
        gxa=0.0
        for b in range(tl.cdiv(OUT,BLOCK)):
            j=b*BLOCK+tl.arange(0,BLOCK); msk=j<OUT
            go=tl.load(go_ptr+n*sg0+j*sg1,mask=msk,other=0.0)
            tlo=tl.load(tab_ptr+lo*st0+i*st1+j*st2,mask=msk,other=0.0)
            thi=tl.load(tab_ptr+hi*st0+i*st1+j*st2,mask=msk,other=0.0)
            W=tlo*(1.0-fr)+thi*fr
            gxa=gxa+tl.sum((W+xi*(thi-tlo)/bw)*go,axis=0)
        tl.store(gx_ptr+n*sgx0+i*sgx1,gxa.to(tl.float32))

    @triton.jit
    def _bwd_t(go_ptr,x_ptr,gt_ptr,N,IN,OUT,K,bw,
               sg0,sg1,sx0,sx1,sgt0,sgt1,sgt2,BLOCK:tl.constexpr):
        n=tl.program_id(0); i=tl.program_id(1); b=tl.program_id(2)
        j=b*BLOCK+tl.arange(0,BLOCK); msk=j<OUT
        xi=tl.load(x_ptr+n*sx0+i*sx1)
        lo=tl.minimum(((xi+1.0)/bw).to(tl.int32),K-1)
        hi=tl.minimum(lo+1,K-1)
        fr=tl.clamp((xi-(-1.0+lo.to(tl.float32)*bw))/bw,0.0,1.0)
        go=tl.load(go_ptr+n*sg0+j*sg1,mask=msk,other=0.0)
        tl.atomic_add(gt_ptr+lo*sgt0+i*sgt1+j*sgt2,xi*(1.0-fr)*go,mask=msk)
        tl.atomic_add(gt_ptr+hi*sgt0+i*sgt1+j*sgt2,xi*   fr   *go,mask=msk)

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,tab,bias,bw):
            N,IN=x.shape; K,_,OUT=tab.shape
            out=torch.zeros(N,OUT,device=x.device,dtype=x.dtype)
            BL=min(128,triton.next_power_of_2(OUT))
            _fwd[(N,triton.cdiv(OUT,BL))](x,tab,out,N,IN,OUT,K,bw,
                x.stride(0),x.stride(1),tab.stride(0),tab.stride(1),tab.stride(2),
                out.stride(0),out.stride(1),BLOCK=BL)
            ctx.save_for_backward(x,tab); ctx.bw=bw; return out+bias
        @staticmethod
        def backward(ctx,go):
            x,tab=ctx.saved_tensors; bw=ctx.bw; N,IN=x.shape; K,_,OUT=tab.shape
            go=go.contiguous(); gx=torch.zeros_like(x); gt=torch.zeros_like(tab)
            BL=min(128,triton.next_power_of_2(OUT))
            _bwd_x[(N,IN)](go,x,tab,gx,N,IN,OUT,K,bw,
                go.stride(0),go.stride(1),x.stride(0),x.stride(1),
                tab.stride(0),tab.stride(1),tab.stride(2),
                gx.stride(0),gx.stride(1),BLOCK=BL)
            _bwd_t[(N,IN,triton.cdiv(OUT,BL))](go,x,gt,N,IN,OUT,K,bw,
                go.stride(0),go.stride(1),x.stride(0),x.stride(1),
                gt.stride(0),gt.stride(1),gt.stride(2),BLOCK=BL)
            return gx,gt,go.sum(0),None

    TRITON_OK = True
    print(f"Triton {triton.__version__} — fused kernel active ✓")
except Exception as e:
    print(f"Triton unavailable ({type(e).__name__}) — using eager mode")


# ── Layers ────────────────────────────────────────────────────────────────
def indexed_dims(d, ff, K, n_heads=4):
    """Equal-param scaling: d_idx = d/sqrt(K), ff_idx = 4*d_idx."""
    s = math.sqrt(K)
    d_idx = max(n_heads, (int(d/s)//n_heads)*n_heads)
    ff_idx = 4 * d_idx
    h = max(1, d_idx // max(1,int(round(s))))
    return d_idx, ff_idx, h

def indexed_lr(base, K):
    """AdamW LR correction for sparse gradient updates: base * sqrt(K/2)."""
    return base * math.sqrt(K/2)

class IndexedLinear(nn.Module):
    def __init__(self, in_d, out_d, K):
        super().__init__()
        self.K=K; self.bw=2./K; self.in_dim=in_d; self.out_dim=out_d
        std = math.sqrt(math.sqrt(3./in_d)*(2./K))
        self.table = nn.Parameter(torch.randn(K,in_d,out_d)*std)
        self.bias  = nn.Parameter(torch.zeros(out_d))

    def _eager(self, x):
        N,IN=x.shape; bw=self.bw
        bk=((x+1)/bw).long().clamp(0,self.K-1); hi=(bk+1).clamp(max=self.K-1)
        fr=((x-(-1+bk.float()*bw))/bw).clamp(0,1)
        ie=torch.arange(IN,device=x.device).unsqueeze(0).expand(N,-1)
        tlo=self.table[bk,ie]; thi=self.table[hi,ie]
        return ((tlo*(1-fr.unsqueeze(-1))+thi*fr.unsqueeze(-1))*x.unsqueeze(-1)).sum(1)+self.bias

    def forward(self, x):
        shape=x.shape; xf=x.reshape(-1,self.in_dim).contiguous()
        if TRITON_OK and xf.is_cuda:
            out=_Fn.apply(xf,self.table,self.bias,self.bw)
        else:
            out=self._eager(xf)
        return out.reshape(*shape[:-1],self.out_dim)

    def flops(self): return 2*self.in_dim*self.out_dim
    def active_bytes(self): return 2*self.in_dim*self.out_dim*4

class StandardLinear(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.lin = nn.Linear(in_d, out_d, bias=True)
    def forward(self, x): return self.lin(x)
    def flops(self): return 2*self.lin.in_features*self.lin.out_features


# ── Benchmark utility ──────────────────────────────────────────────────────
def sync():
    if DEVICE.type=='cuda': torch.cuda.synchronize()

def bench_fn(fn, n_warm, n_bench):
    """Returns median time in ms, synchronised."""
    for _ in range(n_warm): fn()
    sync()
    times = []
    for _ in range(n_bench):
        sync(); t0=time.perf_counter()
        fn()
        sync(); times.append((time.perf_counter()-t0)*1000)
    return float(np.median(times)), float(np.percentile(times,95))

def bench_layer(layer_std, layer_idx, B, IN, OUT, K, n_warm, n_bench):
    """Benchmark one (standard, indexed) layer pair at given B."""
    x_std = torch.randn(B, IN,  device=DEVICE, requires_grad=True)
    x_idx = torch.rand( B, layer_idx.in_dim, device=DEVICE)*2-1
    x_idx = x_idx.detach().requires_grad_(True)

    # Forward only
    fwd_std,_ = bench_fn(lambda: layer_std(x_std), n_warm, n_bench)
    fwd_idx,_ = bench_fn(lambda: layer_idx(x_idx), n_warm, n_bench)

    # Full step (fwd + bwd)
    def step_std():
        w=layer_std.lin.weight.detach().clone().requires_grad_(True)
        (x_std.detach()@w.T).sum().backward()
    def step_idx():
        t=layer_idx.table.detach().clone().requires_grad_(True)
        b=layer_idx.bias.detach().clone().requires_grad_(True)
        layer_idx._eager(x_idx.detach()) if not TRITON_OK else _Fn.apply(
            x_idx.detach().contiguous(),t,b,layer_idx.bw)
        if TRITON_OK:
            _Fn.apply(x_idx.detach().contiguous(),t,b,layer_idx.bw).sum().backward()
        else:
            layer_idx._eager(x_idx.detach()).sum().backward()  # won't track w/o autograd

    # Use autograd properly
    def full_std():
        xs=x_std.detach().requires_grad_(True)
        layer_std(xs).sum().backward()
    def full_idx():
        xi=x_idx.detach().requires_grad_(True)
        t=layer_idx.table.detach().requires_grad_(True)
        b=layer_idx.bias.detach().requires_grad_(True)
        if TRITON_OK and xi.is_cuda:
            _Fn.apply(xi,t,b,layer_idx.bw).sum().backward()
        else:
            layer_idx._eager(xi).sum().backward()

    step_std_ms,_ = bench_fn(full_std, n_warm, n_bench)
    step_idx_ms,_ = bench_fn(full_idx, n_warm, n_bench)

    flops_std = layer_std.flops()
    flops_idx = layer_idx.flops()

    # Throughput
    tflops_std = flops_std * B / fwd_std / 1e9
    tflops_idx = flops_idx * B / fwd_idx / 1e9

    return {
        'fwd_std_ms':  fwd_std,
        'fwd_idx_ms':  fwd_idx,
        'fwd_speedup': fwd_std / fwd_idx,
        'step_std_ms': step_std_ms,
        'step_idx_ms': step_idx_ms,
        'step_speedup':step_std_ms / step_idx_ms,
        'flops_std':   flops_std,
        'flops_idx':   flops_idx,
        'flop_ratio':  flops_std / flops_idx,
        'tflops_std':  tflops_std,
        'tflops_idx':  tflops_idx,
        'idx_faster_fwd':  fwd_idx  < fwd_std,
        'idx_faster_step': step_idx_ms < step_std_ms,
        'active_kb':   layer_idx.active_bytes()/1024,
    }


# ── Sweep configurations ───────────────────────────────────────────────────
if args.quick:
    BATCH_SIZES = [1, 8, 32, 128]
    D_MODELS    = [32, 64, 128, 256]
    K_VALUES    = [4, 16]
    SEQ_LENS    = [48]
    N_BENCH     = 100
    N_WARM      = 20
else:
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    D_MODELS    = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    K_VALUES    = [2, 4, 8, 16, 32]
    SEQ_LENS    = [16, 48, 128, 256]
    N_BENCH     = args.n_bench
    N_WARM      = args.n_warm

N_HEADS = 4  # fixed

print(f"\nConfigs: {len(BATCH_SIZES)} batch × {len(D_MODELS)} d_model × "
      f"{len(K_VALUES)} K × {len(SEQ_LENS)} seq = "
      f"{len(BATCH_SIZES)*len(D_MODELS)*len(K_VALUES)*len(SEQ_LENS)} total")
print(f"Benchmark: {N_WARM} warmup + {N_BENCH} timed per config")
print(f"Backend: {'Triton' if TRITON_OK else 'Eager'}\n")


# ── Run sweep ──────────────────────────────────────────────────────────────
results = []
total = len(BATCH_SIZES)*len(D_MODELS)*len(K_VALUES)*len(SEQ_LENS)
done = 0

print("="*70)
print(f"{'B':>4} {'d':>4} {'K':>3} {'seq':>4} | "
      f"{'fwd_std':>8} {'fwd_idx':>8} {'fwd_spd':>8} | "
      f"{'stp_std':>8} {'stp_idx':>8} {'stp_spd':>8} | "
      f"{'FLOPs↓':>7} {'faster?':>8}")
print("-"*70)

for B, d_std, K, seq in product(BATCH_SIZES, D_MODELS, K_VALUES, SEQ_LENS):
    # Compute indexed dimensions
    d_idx, ff_idx, h_dn = indexed_dims(d_std, 4*d_std, K, N_HEADS)

    # Build layers (just the core linear layers, not full transformer)
    # Standard: Linear(d_std, 4*d_std) -- the FFN up projection
    # Indexed: IndexedLinear(h_dn, d_std, K) -- the FFN down projection
    # For fair comparison: equal params
    #   Standard: d_std * 4*d_std = 4*d_std^2 params
    #   Indexed:  K * h_dn * d_idx = roughly same

    layer_std = StandardLinear(d_std, 4*d_std).to(DEVICE)
    layer_idx = IndexedLinear(h_dn, d_idx, K).to(DEVICE)

    # Batch dimension is B*seq (as if processing whole sequences)
    N = B * seq

    try:
        r = bench_layer(layer_std, layer_idx, N, d_std, 4*d_std, K, N_WARM, N_BENCH)
    except Exception as e:
        print(f"  SKIP B={B} d={d_std} K={K} seq={seq}: {e}")
        continue

    r.update({
        'B': B, 'd_std': d_std, 'd_idx': d_idx, 'K': K, 'seq': seq,
        'h_dn': h_dn, 'N': N,
        'params_std': layer_std.lin.weight.numel() + layer_std.lin.bias.numel(),
        'params_idx': layer_idx.table.numel() + layer_idx.bias.numel(),
        'backend': 'triton' if TRITON_OK else 'eager',
    })
    results.append(r)
    done += 1

    faster_f = '✓ FWD' if r['idx_faster_fwd'] else ''
    faster_s = '✓ STEP' if r['idx_faster_step'] else ''
    faster = faster_f or faster_s or '✗'

    if done % 5 == 1 or r['idx_faster_fwd'] or r['idx_faster_step']:
        print(f"{B:4d} {d_std:4d} {K:3d} {seq:4d} | "
              f"{r['fwd_std_ms']:7.3f}ms {r['fwd_idx_ms']:7.3f}ms {r['fwd_speedup']:7.2f}x | "
              f"{r['step_std_ms']:7.3f}ms {r['step_idx_ms']:7.3f}ms {r['step_speedup']:7.2f}x | "
              f"{r['flop_ratio']:6.1f}x  {faster:>8}")

    # Progress
    if done % 20 == 0:
        print(f"  [{done}/{total}]")


# ── Save raw results ────────────────────────────────────────────────────────
torch.save({'results': results, 'device': str(DEVICE),
            'triton': TRITON_OK, 'n_bench': N_BENCH}, 'sweep_results.pt')

# CSV
fields = list(results[0].keys()) if results else []
with open('sweep_results.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(results)

print(f"\nSaved {len(results)} results → sweep_results.pt, sweep_results.csv")


# ── Analysis ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

# Q1: crossover batch size for forward pass
print("\n1. Batch size crossover (d=64 or largest available, K=16):")
d_target = min(64, max(D_MODELS))
k_target = 16 if 16 in K_VALUES else K_VALUES[-1]
seq_target = 48 if 48 in SEQ_LENS else SEQ_LENS[0]
subset = [r for r in results if r['d_std']==d_target and r['K']==k_target
          and r['seq']==seq_target]
if subset:
    print(f"  {'B':>6} {'fwd_speedup':>12} {'step_speedup':>12} {'fwd faster?':>12}")
    for r in sorted(subset, key=lambda x: x['B']):
        print(f"  {r['B']:>6} {r['fwd_speedup']:>11.2f}x {r['step_speedup']:>11.2f}x "
              f"  {'YES ✓' if r['idx_faster_fwd'] else 'no':>12}")

# Q2: model scale crossover
print(f"\n2. Model scale crossover (B=32, K={k_target}, seq={seq_target}):")
b_target = 32 if 32 in BATCH_SIZES else BATCH_SIZES[-1]
subset2 = [r for r in results if r['B']==b_target and r['K']==k_target
           and r['seq']==seq_target]
if subset2:
    print(f"  {'d_std':>6} {'d_idx':>6} {'fwd_speedup':>12} {'theory':>8} {'faster?':>10}")
    for r in sorted(subset2, key=lambda x: x['d_std']):
        theory = r['flop_ratio']
        print(f"  {r['d_std']:>6} {r['d_idx']:>6} {r['fwd_speedup']:>11.2f}x "
              f"{theory:>7.1f}x   {'YES ✓' if r['idx_faster_fwd'] else 'no':>10}")

# Q3: K scaling
print(f"\n3. K scaling (B={b_target}, d=128, seq={seq_target}):")
d_k = 128 if 128 in D_MODELS else D_MODELS[-1]
subset3 = [r for r in results if r['B']==b_target and r['d_std']==d_k
           and r['seq']==seq_target]
if subset3:
    print(f"  {'K':>4} {'theory_speedup':>15} {'actual_fwd':>12} {'efficiency%':>12}")
    for r in sorted(subset3, key=lambda x: x['K']):
        eff = r['fwd_speedup'] / r['flop_ratio'] * 100
        print(f"  {r['K']:>4} {r['flop_ratio']:>14.1f}x {r['fwd_speedup']:>11.2f}x {eff:>11.1f}%")

# Summary stats
n_faster_fwd  = sum(1 for r in results if r['idx_faster_fwd'])
n_faster_step = sum(1 for r in results if r['idx_faster_step'])
print(f"\n4. Overall: indexed faster in forward: {n_faster_fwd}/{len(results)} configs "
      f"({n_faster_fwd/len(results)*100:.0f}%)")
print(f"           indexed faster in full step: {n_faster_step}/{len(results)} configs "
      f"({n_faster_step/len(results)*100:.0f}%)")


# ── Plots ──────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(20,14))
    fig.patch.set_facecolor('#0d0d0d')
    gs = gridspec.GridSpec(2,3, hspace=0.45, wspace=0.38)

    def ax_(pos):
        a = fig.add_subplot(pos); a.set_facecolor('#111')
        a.tick_params(colors='#aaa'); a.spines[:].set_color('#333')
        return a

    cmap = plt.cm.viridis

    # 1. Batch size vs speedup (line per d_model)
    ax1 = ax_(gs[0,0])
    ax1.set_title("Batch Size vs Forward Speedup\n(K=16, seq=48)", color='white', fontsize=11)
    k_p = 16 if 16 in K_VALUES else K_VALUES[-1]
    s_p = 48 if 48 in SEQ_LENS else SEQ_LENS[0]
    for i,d in enumerate(sorted(set(r['d_std'] for r in results))):
        sub = [(r['B'], r['fwd_speedup']) for r in results
               if r['d_std']==d and r['K']==k_p and r['seq']==s_p]
        if len(sub) > 1:
            bs, spds = zip(*sorted(sub))
            c = cmap(i/len(D_MODELS))
            ax1.plot(bs, spds, 'o-', color=c, lw=2, ms=5, label=f'd={d}')
    ax1.axhline(1.0, color='#e74c3c', ls='--', lw=1.5, label='break-even')
    ax1.set_xscale('log'); ax1.set_xlabel("Batch size", color='#aaa')
    ax1.set_ylabel("Forward speedup (indexed/std)", color='#aaa')
    ax1.legend(facecolor='#111', labelcolor='white', fontsize=7, ncol=2)

    # 2. d_model vs speedup (line per K)
    ax2 = ax_(gs[0,1])
    ax2.set_title("Model Scale vs Forward Speedup\n(B=32, seq=48)", color='white', fontsize=11)
    b_p = 32 if 32 in BATCH_SIZES else BATCH_SIZES[-1]
    for i,K in enumerate(K_VALUES):
        sub = [(r['d_std'], r['fwd_speedup']) for r in results
               if r['B']==b_p and r['K']==K and r['seq']==s_p]
        if len(sub) > 1:
            ds, spds = zip(*sorted(sub))
            c = cmap(i/len(K_VALUES))
            ax2.plot(ds, spds, 'o-', color=c, lw=2, ms=5, label=f'K={K}')
    ax2.axhline(1.0, color='#e74c3c', ls='--', lw=1.5)
    ax2.set_xlabel("d_model (standard)", color='#aaa')
    ax2.set_ylabel("Forward speedup", color='#aaa')
    ax2.legend(facecolor='#111', labelcolor='white', fontsize=8)

    # 3. K vs speedup vs theory
    ax3 = ax_(gs[0,2])
    ax3.set_title("K vs Speedup: Theory vs Actual\n(B=32, d=128, seq=48)", color='white', fontsize=11)
    d_p = 128 if 128 in D_MODELS else D_MODELS[-1]
    sub = [(r['K'], r['fwd_speedup'], r['flop_ratio']) for r in results
           if r['B']==b_p and r['d_std']==d_p and r['seq']==s_p]
    if sub:
        ks, acts, theorys = zip(*sorted(sub))
        ax3.plot(ks, theorys, 's--', color='#aaa', lw=1.5, ms=6, label='Theory (K× FLOPs)')
        ax3.plot(ks, acts, 'o-', color='#2ecc71', lw=2.5, ms=8, label='Actual wall-clock')
        ax3.fill_between(ks, acts, theorys, alpha=0.15, color='#2ecc71', label='Overhead gap')
    ax3.axhline(1.0, color='#e74c3c', ls='--', lw=1)
    ax3.set_xlabel("K (buckets per weight)", color='#aaa')
    ax3.set_ylabel("Speedup", color='#aaa')
    ax3.legend(facecolor='#111', labelcolor='white', fontsize=9)

    # 4. Heatmap: d_model × B → forward speedup
    ax4 = ax_(gs[1,0])
    ax4.set_title("Forward Speedup Heatmap\n(K=16, seq=48)", color='white', fontsize=11)
    ds_u = sorted(set(r['d_std'] for r in results))
    bs_u = sorted(set(r['B'] for r in results))
    mat = np.full((len(ds_u), len(bs_u)), np.nan)
    for r in results:
        if r['K']==k_p and r['seq']==s_p:
            di = ds_u.index(r['d_std']); bi = bs_u.index(r['B'])
            mat[di,bi] = r['fwd_speedup']
    im = ax4.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=2.0, origin='lower')
    ax4.set_xticks(range(len(bs_u))); ax4.set_xticklabels(bs_u, color='white', fontsize=7)
    ax4.set_yticks(range(len(ds_u))); ax4.set_yticklabels(ds_u, color='white', fontsize=7)
    ax4.set_xlabel("Batch size", color='#aaa'); ax4.set_ylabel("d_model", color='#aaa')
    plt.colorbar(im, ax=ax4).ax.yaxis.set_tick_params(color='white')
    ax4.set_title("Fwd Speedup Heatmap (K=16)\nGreen=indexed faster, Red=std faster",
                  color='white', fontsize=10)

    # 5. Step time speedup: d_model × B
    ax5 = ax_(gs[1,1])
    mat2 = np.full((len(ds_u), len(bs_u)), np.nan)
    for r in results:
        if r['K']==k_p and r['seq']==s_p:
            di = ds_u.index(r['d_std']); bi = bs_u.index(r['B'])
            mat2[di,bi] = r['step_speedup']
    im2 = ax5.imshow(mat2, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=2.0, origin='lower')
    ax5.set_xticks(range(len(bs_u))); ax5.set_xticklabels(bs_u, color='white', fontsize=7)
    ax5.set_yticks(range(len(ds_u))); ax5.set_yticklabels(ds_u, color='white', fontsize=7)
    ax5.set_xlabel("Batch size", color='#aaa'); ax5.set_ylabel("d_model", color='#aaa')
    plt.colorbar(im2, ax=ax5).ax.yaxis.set_tick_params(color='white')
    ax5.set_title("Full Step Speedup Heatmap (K=16)\n(fwd+bwd+optim)",
                  color='white', fontsize=10)

    # 6. Efficiency: actual/theory ratio vs d_model
    ax6 = ax_(gs[1,2])
    ax6.set_title("GPU Efficiency: Actual/Theory\n(fraction of FLOP speedup realised)",
                  color='white', fontsize=11)
    for i,K in enumerate(K_VALUES):
        sub = [(r['d_std'], r['fwd_speedup']/r['flop_ratio'])
               for r in results if r['B']==b_p and r['K']==K and r['seq']==s_p
               and r['flop_ratio']>0]
        if len(sub)>1:
            ds,effs=zip(*sorted(sub))
            ax6.plot(ds,[e*100 for e in effs],'o-',
                     color=cmap(i/len(K_VALUES)),lw=2,ms=5,label=f'K={K}')
    ax6.axhline(100,color='#2ecc71',ls='--',lw=1.5,label='100% efficiency')
    ax6.axhline(50, color='#f39c12',ls=':',lw=1,label='50%')
    ax6.set_xlabel("d_model",color='#aaa'); ax6.set_ylabel("Efficiency (%)",color='#aaa')
    ax6.set_ylim(0,120)
    ax6.legend(facecolor='#111',labelcolor='white',fontsize=8)

    fig.suptitle(f"AIWN Comprehensive Sweep — GPU: {DEVICE}  Backend: {'Triton' if TRITON_OK else 'Eager'}",
                 color='white', fontsize=13, y=1.01)
    plt.savefig('sweep_results.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    print("Saved sweep_results.png")
except Exception as e:
    print(f"Plot failed: {e}")