# AIWN — Adaptive Indexed Weight Networks

**[→ View Interactive Explainer](https://capbrady92-svg.github.io/AIWN)** — sliders, visualisations, full walkthrough

> **Best verified result: 15.17× forward speedup, 4.71× training step speedup** — B=32, d=512, K=512, seq=4096 on RTX 5060 Laptop GPU.
> Conservative operating point: **4.40× forward, 1.39× step, 0.7% accuracy cost** at K=128, d=256, seq=4096.
> A drop-in replacement for `nn.Linear` with a tunable speed/accuracy tradeoff.

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
- [Benchmark Methodology](#benchmark-methodology)
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

| Symbol | Full name | What it controls |
|--------|-----------|-----------------|
| **d** | d_model | The hidden dimension of the layer. Common transformer sizes: GPT-2 small = 768, GPT-2 medium = 1024. |
| **K** | Number of buckets | How many weight slices are in the indexed table. Higher K = more expressiveness, fewer active FLOPs per forward pass. Sweet spot: K=128 for Pareto efficiency, K=512 for maximum speed. |
| **B** | Batch size | Number of independent sequences processed simultaneously. |
| **seq** | Sequence length | Number of tokens per sequence. The layer processes B × seq tokens per forward call. |
| **d_idx** | Indexed hidden dim | The actual dimension used by the indexed layer after equal-parameter scaling: `d_idx ≈ d / sqrt(K)`. |
| **ppl_ratio** | Perplexity ratio | `ppl_idx / ppl_std`. Below 1.0 means indexed is more accurate. |
| **fwd_speedup** | Forward speedup | Wall-clock time ratio: standard / indexed. |
| **step_speedup** | Step speedup | Same ratio for a full training step (forward + backward + gradient accumulation). |

---

## How IndexedLinear Works

### The standard linear layer

A standard linear layer computes:

```
out = x @ W + b
```

where `W` is a fixed weight matrix applied identically to every input, regardless
of what that input actually is.

### The indexed approach

`IndexedLinear` replaces the single weight matrix with a **table** of `K` weight
matrices, indexed by the input values themselves:

```
table: shape (K, in_d, out_d)    ← K weight slices instead of 1

For each forward pass:

  1. BUCKET  — divide input domain [-1, 1] into K equal bins.
               for each input element x_i, find which bin it falls in.

  2. INTERPOLATE — linearly interpolate between the two nearest bin entries
                   for smoothness at bin boundaries.

  3. ACCUMULATE — compute the output as a weighted sum over input dimensions.
```

The result: **the weights the layer uses depend on what the input is**. Different
regions of input space get different transformations, learned independently during
training via standard backpropagation.

### The efficiency trick

Despite the table having shape `(K, in_d, out_d)`, only **2 slices per input
dimension** are ever read per forward pass. This means active FLOPs equal a
standard linear layer of shape `(in_d, out_d)`, while total capacity is K times
that of a standard linear layer.

To keep parameter counts comparable when benchmarking, `in_d` and `out_d` are
scaled down by `sqrt(K)` via `indexed_dims()`.

---

## Why It's Faster

Three effects compound to produce the observed speedups:

### 1. Fewer active FLOPs

After equal-parameter scaling, `d_idx ≈ d / sqrt(K)`. FLOP count scales as:

```
FLOPs (standard) = 2 * d * d           = 2d²
FLOPs (indexed)  = 2 * (d/√K) * (d/√K) = 2d² / K
```

Theoretical K× FLOP reduction. At K=128 that is 128× fewer operations on paper.

### 2. Memory bandwidth reduction

At transformer FFN dimensions, modern GPUs are often **memory-bandwidth bound**.
The indexed layer reads a working set of active weights that scales as `1/K`
relative to the standard layer. At K=512, the GPU fetches approximately 0.2% of
the weight data the standard layer requires. This is the primary driver of
speedup at large sequence lengths.

### 3. The Triton fused kernel

The Triton kernel fuses bucketing, interpolation, and accumulation into a single
GPU kernel, keeping the working set in L2 cache and eliminating intermediate
tensor allocations entirely.

---

## Why It's Sometimes Slower

### Kernel launch latency

At small batch sizes and short sequences, fixed kernel launch overhead dominates.
The indexed layer launches a more complex kernel than the cuBLAS-backed
`nn.Linear`. Below the crossover point, this overhead exceeds the savings.

### cuBLAS optimisation

`nn.Linear` dispatches to cuBLAS, hand-tuned by NVIDIA engineers with
specialised tensor core instructions. The Triton kernel cannot match cuBLAS in
regimes where the dimension reduction hasn't produced enough savings.

### The crossover point

Indexed reliably beats standard when all three conditions hold simultaneously:

| Condition | Why it matters |
|-----------|---------------|
| **d ≥ 256** | Large enough that memory bandwidth is the bottleneck |
| **K ≥ 64** | Enough FLOP reduction to overcome kernel complexity |
| **seq ≥ 256** | Enough tokens to amortise fixed launch cost |

### Backward pass overhead

At low K (K=32), the backward pass through the indexed kernel is slower than
standard backprop, resulting in step speedup below 1.0. Step speedup becomes
consistently positive at K≥64 and grows significantly at K≥128.

---

## Why Accuracy Holds and Sometimes Improves

`IndexedLinear` is a **piecewise-linear function approximator**. With K buckets,
it learns K distinct linear transformations — one per region of input space —
smoothly interpolated at boundaries. For nonlinear targets, the piecewise
structure can fit local regions more accurately than a single global matrix
must approximate everything at once.

At K≥16, the piecewise expressiveness consistently compensates for the reduced
hidden dimension. The accuracy crossover happens reliably at K=16 in the
synthetic benchmark.

**Note on accuracy methodology:** The accuracy benchmark trains each layer on a
synthetic regression task (`Y = tanh(X @ W_true)`) at its own operating
dimensions. The standard layer trains at full `(d_std, 4*d_std)` dimensions;
the indexed layer trains at reduced `(d_idx, ff_idx)` dimensions. The ppl_ratio
measures how well each architecture fits its version of the task. This is a
directional signal, not a perfect apples-to-apples comparison. Transformer
validation on real data remains the definitive accuracy test.

---

## Key Results

All results from corrected benchmarks on RTX 5060 Laptop GPU (8.5GB VRAM),
Triton 3.6.0, Python 3.13, PyTorch. Speed benchmark compares
`StandardLinear(d_std, 4*d_std)` vs `IndexedLinear(d_idx, ff_idx, K)` with
approximately equal parameter counts via `indexed_dims()` equal-parameter scaling.

### Two operating points

**Conservative — K=128: Pareto dominant configs (faster AND ≈same accuracy)**

| B | d | K | seq | Fwd speedup | Step speedup | ppl_ratio |
|---|---|---|-----|-------------|--------------|-----------|
| 32 | 256 | 128 | 256 | 3.26× | 1.62× | 1.007 |
| 32 | 256 | 128 | 1024 | 4.58× | 1.93× | 1.007 |
| 32 | 256 | 128 | 4096 | 4.40× | 1.39× | 1.007 |
| 64 | 256 | 128 | 1024 | 4.55× | 1.82× | 1.007 |
| 64 | 256 | 128 | 4096 | 4.36× | 1.41× | 1.007 |

`ppl_ratio ≤ 1.01` = within 1% accuracy of standard. These configs are Pareto
dominant — faster and essentially same accuracy.

**Aggressive — K=512: Maximum speed with accuracy tradeoff**

| B | d | K | seq | Fwd speedup | Step speedup | ppl_ratio |
|---|---|---|-----|-------------|--------------|-----------|
| 32 | 256 | 512 | 1024 | 8.45× | 5.51× | 1.012 |
| 32 | 256 | 512 | 4096 | 11.41× | 5.60× | 1.012 |
| 64 | 256 | 512 | 4096 | 10.63× | 4.27× | 1.012 |
| 32 | 384 | 512 | 4096 | 10.84× | 3.89× | 1.066 |
| 32 | 512 | 512 | 1024 | 16.27× | 6.18× | 1.083 |
| 32 | 512 | 512 | 4096 | 15.17× | 4.71× | 1.083 |
| 64 | 512 | 512 | 4096 | 15.40× | 4.51× | 1.083 |

### Cases where indexed is slower

| B | d | K | seq | Fwd speedup | Step speedup | Why |
|---|---|---|-----|-------------|--------------|-----|
| 32 | 256 | 32 | any | ~1.1× | 0.37–0.51× | K too low — backward pass overhead dominates |
| 32 | 384 | 32 | any | ~1.5× | 0.48–0.59× | Same — K=32 below sweet spot |
| 32 | 512 | 32 | any | ~1.4× | 0.50× | Same |

**Key finding:** K=32 consistently produces negative step speedup. K=64 is the
crossover — step speedup reaches ~1.0× at K=64 and grows from there.

### K scaling summary (d=256, B=32, seq=4096)

| K | d_idx | Fwd speedup | Step speedup | ppl_ratio | Params ratio |
|---|-------|-------------|--------------|-----------|--------------|
| 32 | 44 | 1.09× | 0.37× | 1.003 | 0.94× |
| 64 | 32 | 2.66× | 0.81× | 1.005 | 1.00× |
| 128 | 20 | 4.40× | 1.39× | 1.007 | 0.78× |
| 256 | 16 | 5.45× | 1.90× | 1.021 | 1.00× |
| 512 | 8 | 11.41× | 5.60× | 1.012 | 0.50× |

---

## High-K Results

Extended experiments testing K=32–512 at production-relevant sequence lengths.

### K scaling at seq=4096 — full picture

| B | d | K | seq | d_idx | Fwd speedup | Step speedup | ppl_ratio |
|---|---|---|-----|-------|-------------|--------------|-----------|
| 32 | 256 | 128 | 4096 | 20 | 4.40× | 1.39× | 1.007 |
| 32 | 256 | 256 | 4096 | 16 | 5.45× | 1.92× | 1.021 |
| 32 | 256 | 512 | 4096 | 8 | 11.41× | 5.60× | 1.012 |
| 32 | 384 | 128 | 4096 | 32 | 5.45× | 1.65× | 1.020 |
| 32 | 384 | 256 | 4096 | 24 | 7.58× | 2.36× | 1.043 |
| 32 | 384 | 512 | 4096 | 16 | 10.84× | 3.89× | 1.066 |
| 32 | 512 | 128 | 4096 | 44 | 3.78× | 1.25× | 1.041 |
| 32 | 512 | 256 | 4096 | 32 | 9.18× | 2.74× | 1.067 |
| 32 | 512 | 512 | 4096 | 20 | 15.17× | 4.71× | 1.083 |

### Practical guidance

Use **K=128** when accuracy matters — Pareto dominant at d=256, minimal accuracy
cost, positive step speedup for training.

Use **K=512** when inference speed is the priority and ~8% accuracy degradation
is acceptable. Best results at d≥384, seq≥1024.

Avoid **K=32** for training — negative step speedup across all configs tested.
Acceptable for inference-only at large d and seq.

---

## Benchmark Methodology

**Speed benchmark:** `StandardLinear(d_std, 4*d_std)` vs
`IndexedLinear(d_idx, ff_idx, K)` where `d_idx, ff_idx = indexed_dims(d_std, K)`.
Parameter counts are approximately equal (within ~6–22% depending on n_heads
rounding). Both layers are timed on GPU with full CUDA synchronisation,
`n_warm=50` warmup iterations and `n_bench=200` timed iterations, median reported.

**Accuracy benchmark:** Each layer trains independently on a synthetic regression
task `Y = tanh(X @ W_true)`, X ~ Uniform[-1,1], 500 AdamW steps. Standard layer
trains at `(d_std, 4*d_std)` dimensions; indexed layer trains at `(d_idx, ff_idx)`
dimensions. `ppl_ratio = exp(val_mse_idx) / exp(val_mse_std)`. Because the two
layers solve the task at different scales, ppl_ratio is a directional signal
rather than a strict apples-to-apples comparison. Transformer validation on real
data is the definitive accuracy test.

**Parameter matching:** `indexed_dims()` targets equal parameters via
`d_idx = d_std / sqrt(K)`, rounded to nearest `n_heads` multiple. At high K,
rounding can produce ~50% parameter mismatch. Actual param counts are logged
in the CSV output for every config.

---

## Open Questions

- **Transformer validation** — Does the speedup hold inside a full transformer
  with residual connections and layernorm? The synthetic task confirms the
  mechanism works but real activation distributions may differ.
- **Layernorm domain** — Post-layernorm activations are constrained to a
  hypersphere, not uniform [-1,1]. Adaptive bucket boundaries may be needed.
- **K schedule** — Is there an optimal K warmup schedule analogous to learning
  rate warmup?
- **Step speedup at K=32** — Why does the backward pass dominate at low K?
  Understanding this may unlock improvements to the backward kernel.
- **Transformer-scale dimensions** — All benchmarks use d≤512. GPT-2 uses
  d=768–1600. Results at transformer-scale dimensions are not yet characterised.
- **MoE comparison** — Head-to-head with Mixture of Experts at equal parameter
  budget.

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
        return {'results': [...]}

    def analyze(self, results):
        print("My findings:", ...)

    def plot(self, results, out_path):
        ...
```

```bash
python run.py my_experiment --my_param 42
```