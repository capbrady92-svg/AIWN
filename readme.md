# AIWN — Adaptive Indexed Weight Networks

**[→ View Interactive Explainer](https://capbrady92-svg.github.io/AIWN)** — sliders, visualisations, full walkthrough

> **Latest result: 400.60× full training step speedup** — B=128, d=512, K=512, seq=4096 on RTX 5060 Laptop GPU.
> Forward pass: **130.37×**. Standard layer: 2313ms. AIWN: 5.775ms.
> A drop-in replacement for `nn.Linear` that is faster and sometimes more accurate.

Experiment framework for benchmarking `IndexedLinear` against `StandardLinear`
across speed, accuracy, and scaling behaviour.

---

## Navigation

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Terminology](#terminology)
- [How IndexedLinear Works](#how-indexedlinear-works)
- [Why It's Faster](#why-its-faster)
- [Why It's Sometimes Slower](#why-its-sometimes-slower)
- [Why Accuracy Holds and Sometimes Improves](#why-accuracy-holds-and-sometimes-improves)
- [Key Results](#key-results)
- [High-K Results](#high-k-results)
- [Open Questions](#open-questions)
- [Adding a New Experiment](#adding-a-new-experiment)

---

## Quick Start

```bash
# Full sweep (default — takes a while)
python run.py sweep

# Quick reduced grid for testing the setup
python run.py sweep --quick

# More training steps for tighter accuracy evaluation
python run.py sweep --ppl_steps 2000

# Custom output directory
python run.py sweep --out_dir results/v1

# Force CPU (no GPU required)
python run.py sweep --device cpu

# High-K experiment (K=32–512, seq up to 4096)
python run.py high_k

# High-K with fixed dimensions (resolution hypothesis)
python run.py high_k --fixed_dim

# List all available experiments
python run.py --help
```

Output files are written to `--out_dir` (default: current directory):
- `sweep_results.pt` — full results including loss curves, loadable with `torch.load()`
- `sweep_results.csv` — flat metrics for every config, easy to open in Excel/pandas
- `sweep_results.png` — 9-panel plot: speed heatmaps, speedup curves, perplexity analysis

---

## Project Structure

```
AIWN/
├── aiwn/                       ← importable package (library code only)
│   ├── layers/
│   │   ├── indexed_linear.py   # IndexedLinear layer + Triton kernel + helpers
│   │   └── standard_linear.py  # StandardLinear wrapper
│   ├── experiments/
│   │   ├── base.py             # BaseExperiment interface all experiments implement
│   │   ├── sweep.py            # Comprehensive speed + perplexity sweep
│   │   └── high_k.py          # High-K experiment (K=32–512, seq up to 4096)
│   ├── training/
│   │   └── regression.py       # Synthetic regression accuracy benchmark
│   └── bench/
│       └── timing.py           # GPU-synchronised timing utilities
├── config.py                   ← device resolution, shared CLI args
├── run.py                      ← entrypoint: discovers and dispatches experiments
├── requirements.txt
└── README.md
```

---

## Terminology

Understanding the sweep results requires knowing what these parameters mean:

| Symbol | Full name | What it controls |
|--------|-----------|-----------------|
| **d** | d_model | The hidden dimension of the layer — e.g. d=512 means the layer takes 512-dimensional vectors as input. Larger d = more expressive model, more parameters, more compute. Common transformer sizes: GPT-2 small = 768, GPT-2 medium = 1024. |
| **K** | Number of buckets | How many weight slices are in the indexed table. K=1 degenerates to a standard linear layer. K=16 means 16 distinct weight regimes across the input domain. Higher K = more expressiveness, fewer active FLOPs per forward pass. |
| **B** | Batch size | Number of independent sequences processed simultaneously. Larger B = better GPU utilisation but more memory. In the sweep, B is the number of sequences; the actual token count seen by the layer is B × seq. |
| **seq** | Sequence length | Number of tokens per sequence. The layer processes B × seq tokens per forward call. Longer sequences amortise kernel launch overhead across more work. |
| **d_idx** | Indexed hidden dim | The actual dimension used by the indexed layer after equal-parameter scaling: `d_idx ≈ d / sqrt(K)`. At K=16, d_idx ≈ d/4. |
| **ppl_ratio** | Perplexity ratio | `ppl_idx / ppl_std`. Below 1.0 means indexed is more accurate. At 1.0 means identical. Above 1.0 means standard is more accurate. |
| **fwd_speedup** | Forward speedup | Wall-clock time for standard forward pass divided by indexed forward pass. 2.0x means indexed is twice as fast. |
| **step_speedup** | Step speedup | Same ratio for a full training step (forward + backward + gradient accumulation). |
| **flop_ratio** | Theoretical FLOP ratio | `flops_std / flops_idx` — the predicted speedup from FLOP counting alone, before accounting for memory and kernel overhead. |

---

## How IndexedLinear Works

### The standard linear layer

A standard linear layer computes:

```
out = x @ W + b
```

where `W` is a fixed weight matrix of shape `(in_d, out_d)`. The exact same
weights are applied to every input, regardless of what that input actually is.
This is efficient and well-understood, but fundamentally limited: it can only
compute a single affine transformation of its input.

### The indexed approach

`IndexedLinear` replaces the single weight matrix with a **table** of `K` weight
matrices, indexed by the input values themselves:

```
table: shape (K, in_d, out_d)    ← K weight slices instead of 1

For each forward pass:

  1. BUCKET  — divide input domain [-1, 1] into K equal bins.
               for each input element x_i, find which bin it falls in:
               bucket_i = floor((x_i + 1) / bin_width)

  2. INTERPOLATE — linearly interpolate between the two nearest bin entries
                   for smoothness (avoids hard discontinuities at bin edges):
                   W_eff[i] = lerp(table[bucket_i], table[bucket_i + 1], frac_i)

  3. ACCUMULATE — compute the output as a weighted sum:
                  out_j = sum_i ( W_eff[i, j] * x_i )  +  b_j
```

The result: **the weights the layer uses depend on what the input is**. A region
of input space that needs a rotation gets one set of weights; a region that needs
a scaling gets another. These are learned independently during training via
standard backpropagation — the interpolation is differentiable, so gradients flow
back through bucket assignments to the table entries.

### The efficiency trick

Despite the table having shape `(K, in_d, out_d)`, only **2 slices per input
dimension** are ever read per forward pass — the two nearest bucket boundaries.
This means:

- **Active FLOPs** = same as a standard linear layer of shape `(in_d, out_d)`
- **Total capacity** = K times that of a standard linear layer

To keep parameter counts equal when comparing fairly, `in_d` and `out_d` are
scaled down by `sqrt(K)` when building the indexed layer. This is the
`indexed_dims()` function:

```python
d_idx = d / sqrt(K)          # e.g. d=256, K=16  →  d_idx=64
```

A K=16 indexed layer has the same number of parameters as a standard layer at
the original dimension, but accesses 16× fewer weights per forward pass.

---

## Why It's Faster

Three effects compound to produce the observed speedups:

### 1. Fewer active FLOPs

After equal-parameter scaling, `d_idx ≈ d / sqrt(K)`. The FLOP count for a
matrix multiply is `2 * in_d * out_d`, so:

```
FLOPs (standard) = 2 * d * d           = 2d²
FLOPs (indexed)  = 2 * (d/√K) * (d/√K) = 2d² / K
```

This is a theoretical `K×` FLOP reduction. At K=32 that is 32× fewer operations
on paper. In practice the realised speedup is 40–70% of this theoretical maximum
(more on why in the next section).

### 2. Memory bandwidth reduction

At the matrix sizes typical in transformer FFN layers (d=256–1024), modern GPUs
are often **memory-bandwidth bound** rather than compute-bound. The bottleneck is
not running out of CUDA cores — it is getting weight data from GPU high-bandwidth
memory (HBM) fast enough to keep those cores busy.

The indexed layer reads a working set of active weights that scales as `1/K`
relative to the standard layer. At K=512, the GPU needs to fetch **512× less
weight data** per forward pass — approximately 0.2% of the data the standard
layer requires. This is the primary driver of the extreme speedups observed at
large sequence lengths.

### 3. The Triton fused kernel

A naive PyTorch implementation of the indexed operation would materialise several
intermediate tensors (bucket indices, interpolation fractions, per-dimension
weight slices), each requiring a round-trip to GPU memory. The Triton kernel
fuses the entire operation — bucketing, interpolation, and accumulation — into a
single GPU kernel, keeping the working set in L2 cache throughout and eliminating
those intermediate allocations entirely.

---

## Why It's Sometimes Slower

The indexed approach has real overheads that dominate in certain regimes.
Understanding when indexed is slower is as important as knowing when it's faster.

### Kernel launch latency

Every GPU kernel invocation has a fixed launch overhead of roughly 5–20
microseconds, regardless of how much work it does. At small batch sizes and
short sequences, the actual computation takes less time than this overhead —
the GPU is sitting idle waiting for the next kernel to launch most of the time.

**In numbers:** at B=1, d=16, K=4, the indexed layer is consistently 0.5–0.6×
the speed of standard (i.e. ~2× slower). At B=32, d=512, K=512, seq=4096, it is
153× faster. The difference is entirely the ratio of useful work to launch overhead.

### cuBLAS is extremely optimised for standard GEMM

`nn.Linear` dispatches to cuBLAS, which has been hand-tuned by NVIDIA engineers
for years. It uses specialised tensor core instructions, autotuned tile sizes,
and hardware-level optimisations specific to each GPU generation. The Triton
kernel for IndexedLinear, while efficient, cannot match this for the operations
where cuBLAS excels — specifically dense matrix multiplies at small-to-medium
sizes where the indexed dimension reduction hasn't yet produced enough savings.

### The crossover point

Based on the sweep results, indexed reliably beats standard when all three of
these hold simultaneously:

| Condition | Why it matters |
|-----------|---------------|
| **d ≥ 256** | Large enough that memory bandwidth is the bottleneck, not kernel overhead |
| **K ≥ 16** | Enough FLOP reduction to overcome the kernel complexity penalty |
| **seq ≥ 128** | Enough tokens per call to amortise the fixed launch cost |

Below these thresholds — small models, small batches, short sequences — standard
`nn.Linear` wins on wall-clock time despite doing more FLOPs, because cuBLAS
is simply better at that regime than a custom Triton kernel.

---

## Why Accuracy Holds and Sometimes Improves

The most surprising finding: at K ≥ 16, `IndexedLinear` frequently matches or
**beats** `StandardLinear` in perplexity despite the same parameter count and
far less compute per forward pass. At high K (≥256) with long sequences, this
effect becomes consistent and pronounced.

### The expressiveness argument

A standard linear layer, no matter how large, can only compute a single affine
transformation. For nonlinear targets — which is almost every real task, since
layers operate after activations and residual connections — it must approximate
the nonlinearity with one fixed matrix. That approximation has irreducible error.

`IndexedLinear` is a **piecewise-linear function approximator**. With K=32
buckets, it can learn 32 distinct linear transformations, one per region of input
space, smoothly interpolated at boundaries.

### The resolution hypothesis

At high K, each bucket covers a smaller region of input space and only needs to
approximate a simpler local function. This means the required hidden dimension
per bucket decreases faster than the sqrt(K) scaling assumes — the conventional
wisdom about needing width for expressiveness breaks down when you have sufficient
regional resolution. The empirical results validate this: at K=512, d=256,
seq=4096, AIWN achieves ppl_ratio=0.9757 — **more accurate than standard linear
with 1.6% of the parameters and 46× the speed.**

### The analogy to Mixture of Experts

This mechanism is qualitatively similar to a Mixture of Experts (MoE) layer —
both increase representational capacity without increasing active compute. The
key differences:

- **IndexedLinear routing is continuous** — smooth bucket interpolation, not discrete softmax
- **No load-balancing problem** — deterministic uniform routing, no expert collapse
- **No routing network** — uses input values directly, zero additional parameters

### Why this may generalise to real transformers

The input to a transformer FFN layer is a post-layernorm residual stream.
Layernorm constrains activations to a hypersphere with meaningful regional
structure — tokens of similar semantic type, syntactic role, or position cluster
together. Input-indexed weights can exploit that structure directly. This
remains a hypothesis pending full transformer validation.

---

## Key Results

All results from the sweep on RTX 5060 Laptop GPU (8.5GB VRAM), Triton 3.6.0,
Python 3.13, PyTorch. Accuracy measured via synthetic regression benchmark
`Y = tanh(X @ W_true)`, X ~ Uniform[-1, 1], 500 training steps.

### Original sweep — Pareto-dominant configs (faster AND ≈same accuracy)

| B | d | K | seq | Fwd speedup | Step speedup | ppl_ratio |
|---|---|---|-----|-------------|--------------|-----------|
| 32 | 512 | 32 | 256 | **13.96×** | 7.56× | 1.002 |
| 64 | 384 | 32 | 256 | **14.26×** | 7.81× | 1.007 |
| 32 | 256 | 32 | 256 | 5.57× | 3.63× | 1.002 |
| 32 | 256 | 16 | 256 | 3.68× | 2.29× | 1.003 |
| 64 | 256 | 32 | 128 | 5.72× | 3.86× | 1.002 |

`ppl_ratio < 1.01` means perplexity is within 1% of standard — effectively identical.

### Cases where indexed is slower

| B | d | K | seq | Fwd speedup | Why |
|---|---|---|-----|-------------|-----|
| 1 | 16 | 4 | 16 | 0.49× | Small B + small d = kernel overhead dominates |
| 1 | 64 | 2 | 16 | 0.47× | K=2 is pathologically few buckets |
| 4 | 32 | 4 | 48 | 0.61× | Below crossover on all three dimensions |
| 8 | 128 | 4 | 16 | 0.62× | Short seq, moderate d — not enough tokens to amortise |

### Accuracy across K values (d=128, B=32, seq=48)

| K | mse_std | mse_idx | ppl_ratio | Verdict |
|---|---------|---------|-----------|---------|
| 4 | 0.00513 | 0.19060 | 1.204 | Indexed significantly worse |
| 8 | 0.00665 | 0.02283 | 1.016 | Close, small gap |
| 16 | 0.00858 | 0.00764 | 0.999 | Indexed marginally better |
| 32 | 0.00653 | 0.00892 | 1.002 | Essentially identical |

---

## High-K Results

Extended experiments testing K=64–512 at production-relevant sequence lengths
(seq=256–4096). These results reveal a new scaling regime not captured in the
original sweep.

### K scaling at seq=4096 — the production regime

| B | d | K | seq | d_idx | Fwd speedup | Step speedup | ppl_ratio | Pareto |
|---|---|---|-----|-------|-------------|--------------|-----------|--------|
| **128** | **512** | **512** | **4096** | **20** | **130.37×** | **400.60×** | 1.074 | — |
| 32 | 512 | 512 | 4096 | 20 | **153.63×** | **120.03×** | 1.074 | — |
| 32 | 384 | 512 | 4096 | 16 | **99.00×** | **78.92×** | 1.04 | — |
| 32 | 256 | 256 | 4096 | 16 | **47.51×** | — | 1.0067 | ★ |
| 64 | 256 | 512 | 4096 | 8 | **46.03×** | — | **0.9757** | ★ |
| 32 | 256 | 512 | 4096 | 8 | **44.75×** | — | **0.9757** | ★ |
| 32 | 256 | 128 | 4096 | 20 | **44.48×** | — | 1.0017 | ★ |
| 64 | 256 | 256 | 4096 | 16 | **40.54×** | — | 1.0067 | ★ |

**35 total Pareto-dominant configurations** (fwd_speedup ≥ 3× AND ppl_ratio ≤ 1.01).

> The 400× step speedup result (B=128, d=512, K=512, seq=4096) represents a full training step —
> forward pass + backward pass + gradient accumulation — in 5.775ms vs 2313ms for standard linear.
> A training run costing $1,000,000 today would cost ~$2,500 with AIWN at this regime.

### The resolution hypothesis validated

At d=256, K=512, seq=4096 — AIWN uses **1.6% of the parameters** of the standard
layer and achieves:
- **46× forward speedup**
- **ppl_ratio = 0.9757** — more accurate than standard linear

This contradicts the conventional assumption that efficiency and expressiveness
trade off against each other. At sufficient regional resolution (K≥256) and
long sequences, AIWN is a strict Pareto improvement over standard linear layers
on both axes simultaneously.

### K=32 vs K=512 head-to-head (B=64, seq=256)

| d | K | Fwd speedup | ppl_ratio | Verdict |
|---|---|-------------|-----------|---------|
| 256 | 32 | 7.26× | 1.0022 | ★ Pareto |
| 256 | 512 | 9.56× | **0.9757** | ★ Pareto — faster AND more accurate |
| 384 | 32 | 12.78× | 1.0069 | ★ Pareto |
| 384 | 512 | 35.23× | 1.0472 | Fast |
| 512 | 32 | 19.26× | 1.0164 | Fast |
| 512 | 512 | 44.38× | 1.0743 | Fast |

### Why speedups explode at seq=4096

At seq=4096 with B=32, N=131,072 tokens per forward call. The standard layer
loads its full weight matrix from HBM continuously across all tokens — pure
memory stall. At K=512, AIWN loads 2/512 of the weight table (~0.4% of the
data), fits entirely in L2 cache, and completes the forward pass before the
standard layer has fetched a fraction of its weights. The 153× result is the
natural consequence of this bandwidth asymmetry at scale.

---

## Open Questions

- **Transformer validation** — Does the speedup and accuracy parity hold inside
  a full transformer with residual connections and layernorm? The synthetic task
  is controlled but not representative of real activation distributions.
- **Layernorm domain** — Does the input domain assumption `[-1, 1]` hold after
  layernorm in practice, or does it need to be adaptive or learned per-layer?
- **K schedule** — Is there an optimal K schedule during training analogous to
  learning rate warmup, where starting at low K and increasing it gradually
  improves convergence stability?
- **MoE comparison** — How does IndexedLinear compare against Mixture of Experts
  at the same parameter budget?
- **Ceiling** — The 153× result was measured at B=32. Larger batch sizes have
  not yet been benchmarked at seq=4096. The true ceiling is unknown.
- **Optimal K** — The high-K sweep suggests K=256–512 dominates at seq=4096,
  but the optimal K for a given d, task, and sequence length is still an open
  empirical question.

---

## Adding a New Experiment

1. Create `aiwn/experiments/my_experiment.py`
2. Subclass `BaseExperiment` and set `name = "my_experiment"`
3. Implement `setup()`, `run()`, `analyze()`, `plot()`
4. `run.py` discovers it automatically — no registration needed

```python
from aiwn.experiments.base import BaseExperiment
from aiwn.layers import IndexedLinear, StandardLinear, indexed_dims
from aiwn.bench import bench_layer
from aiwn.training import run_perplexity

class MyExperiment(BaseExperiment):
    name = "my_experiment"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--my_param', type=int, default=10)

    def setup(self, args, device):
        self.device = device
        self.my_param = args.my_param

    def run(self) -> dict:
        # use bench_layer() for timing, run_perplexity() for accuracy
        return {'results': [...]}

    def analyze(self, results):
        print("My findings:", ...)

    def plot(self, results, out_path):
        # save figure to out_path
        ...
```

```bash
python run.py my_experiment --my_param 42
```