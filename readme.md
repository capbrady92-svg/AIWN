# AIWN — Adaptive Indexed Weight Networks

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
│   │   └── sweep.py            # Comprehensive speed + perplexity sweep
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
relative to the standard layer. At K=16, the GPU needs to fetch 16× less weight
data per forward pass. This is why the speedup is often larger than the FLOP
ratio alone would predict at large d and long seq: the GPU spends less time
stalled waiting for memory, not just less time computing.

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

The indexed layer launches a more complex kernel than a standard `nn.Linear`
(which maps to a single highly-optimised cuBLAS GEMM call). When the total
compute per kernel call is small — small B, small d, short seq — the indexed
kernel's overhead is larger in absolute terms, making it slower even if it does
fewer FLOPs.

**In numbers:** at B=1, d=16, K=4, the indexed layer is consistently 0.5–0.6×
the speed of standard (i.e. ~2× slower). At B=32, d=384, K=32, seq=256, it is
13× faster. The difference is entirely the ratio of useful work to launch overhead.

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
far less compute per forward pass.

### The expressiveness argument

A standard linear layer, no matter how large, can only compute a single affine
transformation. For nonlinear targets — which is almost every real task, since
layers operate after activations and residual connections — it must approximate
the nonlinearity with one fixed matrix. That approximation has irreducible error.

`IndexedLinear` is a **piecewise-linear function approximator**. With K=32
buckets, it can learn 32 distinct linear transformations, one per region of input
space, smoothly interpolated at boundaries. For a target like
`Y = tanh(X @ W_true)`:

- Near zero, where tanh ≈ linear, it uses one set of weights
- Near ±1, where tanh saturates and flattens, it uses another
- The transition between regimes is smooth, not discontinuous

The standard layer must find a single compromise matrix that partially fits all
regions. The indexed layer fits each region independently.

### Why convergence accelerates with K

At K=32, the layer has 32 weight regimes to draw on. Early in training, it can
quickly specialise different bucket regions to different parts of the target
function, reducing loss faster than a standard layer that must slowly adjust
a single global matrix. This is why the loss curves show indexed reaching lower
MSE faster — it is not a fluke of initialisation, it is the piecewise structure
enabling faster specialisation.

### The analogy to Mixture of Experts

This mechanism is qualitatively similar to a Mixture of Experts (MoE) layer —
both increase representational capacity without increasing active compute, by
routing different inputs to different weight matrices. The key differences:

- **IndexedLinear routing is continuous** — determined by a smooth bucket
  interpolation rather than a discrete softmax gate
- **No load-balancing problem** — discrete MoE routing famously causes expert
  collapse (some experts get most of the traffic, others are never used). Indexed
  routing is deterministic and uniform by construction
- **No routing network** — MoE requires a separate learned gating network.
  IndexedLinear uses the input values directly, adding zero parameters

### Why this may generalise to real transformers

The input to a transformer FFN layer is a post-layernorm residual stream.
Layernorm constrains activations to a hypersphere, and the representational
geometry literature consistently finds that this space has meaningful regional
structure — tokens of similar semantic type, syntactic role, or position cluster
together in activation space. If the residual stream is regionally structured,
then input-indexed weights can exploit that structure directly: different weight
regimes for semantically different input regions. A standard FFN layer applies
the same transformation everywhere and cannot do this without stacking multiple
nonlinear layers. IndexedLinear may achieve in one operation what standard
architectures need depth to approximate.

This remains a hypothesis — the synthetic regression task confirms the mechanism
works in a controlled setting, but real transformer validation is still needed.

---

## Key Results

All results from the sweep on RTX 5060 Laptop GPU (8.5GB VRAM), Triton 3.6.0,
Python 3.13, PyTorch. Accuracy measured via synthetic regression benchmark
`Y = tanh(X @ W_true)`, X ~ Uniform[-1, 1], 500 training steps.

### Pareto-dominant configs (faster AND ≈same accuracy)

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

The accuracy crossover happens at K=16. Below that, the reduced hidden dimension
hurts more than the extra expressiveness helps. At K=16 and above, the
piecewise approximation fully compensates for the smaller dimension.

---

## Open Questions

- Does the speedup and accuracy parity hold inside a full transformer with
  residual connections and layernorm? The synthetic task is controlled but
  not representative of real activation distributions.
- Does the input domain assumption `[-1, 1]` hold after layernorm in practice,
  or does it need to be adaptive or learned per-layer?
- Is there an optimal K schedule during training — analogous to learning rate
  warmup — where starting at low K and increasing it gradually improves
  convergence stability?
- How does IndexedLinear compare against Mixture of Experts at the same
  parameter budget? Both increase capacity without increasing active compute,
  but through fundamentally different routing mechanisms.
- Does the faster convergence at high K hold on real tasks, or is it an
  artefact of the synthetic regression target being particularly well-suited
  to piecewise-linear approximation?
- What is the optimal K for a given d and task? The sweep suggests K=16–32
  is the sweet spot, but this may be task and architecture dependent.

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