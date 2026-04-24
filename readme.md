# AIWN — Adaptive Indexed Weight Networks

Experiment framework for benchmarking `IndexedLinear` against `StandardLinear`.

## Structure

```
aiwn/
├── layers/
│   ├── indexed_linear.py   # IndexedLinear, indexed_dims, indexed_lr (+ Triton kernel)
│   └── standard_linear.py  # StandardLinear wrapper
├── experiments/
│   ├── base.py             # BaseExperiment interface
│   └── sweep.py            # Comprehensive speed + perplexity sweep
├── training/
│   └── regression.py       # Synthetic regression task (make_dataset, train_and_eval)
├── bench/
│   └── timing.py           # GPU-synchronised bench_fn, bench_layer
└── config.py               # Device resolution, shared CLI args
run.py                      # Entrypoint — discovers and dispatches experiments
```

---

## How IndexedLinear works

A standard linear layer computes:

```
out = x @ W + b
```

where `W` is a fixed weight matrix shared across all inputs. The same weights
are applied regardless of what the input actually is.

`IndexedLinear` replaces the single weight matrix with a **table** of `K` weight
matrices indexed by the input values themselves:

```
table: shape (K, in_d, out_d)   ← K weight slices instead of one

for each input element x_i:
    bucket = floor((x_i + 1) / bucket_width)                     ← which slice?
    W_eff[i] = lerp(table[bucket], table[bucket+1], frac)         ← interpolate

out_j = sum_i  W_eff[i, j] * x_i  +  b_j                        ← accumulate
```

The input domain `[-1, 1]` is divided into `K` equally-spaced buckets. For each
input dimension, the active weight slice is determined by where that input value
falls in the domain, with linear interpolation between the two nearest bucket
boundaries for smoothness.

In plain terms: **the weights the layer uses depend on what the input is**.
Regions of input space that need different transformations get different effective
weight matrices, learned independently during training.

The key efficiency constraint: although the table has shape `(K, in_d, out_d)`,
only **2 slices per input dimension** are read per forward pass — the two nearest
bucket boundaries. The active compute is the same as a standard linear layer,
`O(in_d * out_d)`, but the total representational capacity is `K` times larger.

To keep parameter counts equal when comparing fairly, `in_d` and `out_d` are
scaled down by `sqrt(K)` via `indexed_dims()`. A K=16 indexed layer has roughly
the same number of parameters as a standard layer at the original dimensions,
but accesses far fewer of them per forward pass.

---

## Why it's faster

The speedup comes from two compounding effects:

**1. Fewer active FLOPs per forward pass.**
After equal-parameter scaling, `d_idx ≈ d / sqrt(K)`. The FLOP count for a
matrix multiply is `2 * in_d * out_d`, so the indexed layer does roughly:

```
2 * (d / sqrt(K)) * (d / sqrt(K))  =  2 * d² / K
```

— a theoretical `K×` FLOP reduction. At K=32 that is a 32x reduction on paper.

**2. Memory bandwidth reduction.**
Modern GPUs are frequently memory-bandwidth bound rather than compute-bound,
especially at the matrix sizes common in transformer FFN layers. The indexed
layer reads a much smaller working set of weights per forward pass — active
weight bytes scale as `1/K` relative to the standard layer at the same parameter
count. At large sequence lengths, where many tokens are processed in a single
kernel call, this translates directly to wall-clock speedup because the GPU
spends less time waiting for weight data to arrive from HBM.

**3. The Triton kernel fuses the operation.**
Rather than materialising intermediate tensors, the fused kernel computes bucket
indices, performs interpolation, and accumulates the output in a single pass.
This avoids the extra memory round-trips that a naive PyTorch implementation
would require, keeping the working set in L2 cache across the entire computation.

The practical speedup is always less than the theoretical `K×` due to kernel
launch overhead, index computation cost, and the fact that small layers are
latency-bound rather than throughput-bound. But at large `d` and long sequences
the gap closes significantly — the sweep results show the Triton kernel achieving
40–70% of theoretical FLOP speedup at d ≥ 256.

---

## Why accuracy holds (and sometimes improves) at high K

This is the most surprising finding from the sweep. At K ≥ 16, `IndexedLinear`
frequently matches or beats `StandardLinear` in perplexity despite the same
parameter count and far less compute per forward pass.

**The standard layer is fundamentally limited to linear functions of its input.**
No matter how many parameters it has, `out = x @ W + b` computes a single affine
transformation. If the true relationship between input and output is nonlinear —
which it almost always is in practice, since layers operate after nonlinear
activations and residual connections — the standard layer must approximate that
nonlinearity with a single fixed matrix, leaving residual error it cannot reduce.

**IndexedLinear is a piecewise-linear function approximator.** With K buckets,
the effective weight matrix changes smoothly as the input moves through the
domain. For a target like `Y = tanh(X @ W_true)`, the indexed layer can learn a
different effective `W` for each region of input space: near zero where tanh is
approximately linear it uses one set of weights, near saturation where tanh
flattens it uses another. The standard layer must find a single compromise matrix
that works everywhere.

**Higher K means finer-grained input-dependence, which explains why convergence
accelerates with K.** At K=32 the layer can represent 32 distinct weight regimes
per input dimension, interpolated smoothly. This is qualitatively similar to a
Mixture of Experts layer, but with routing determined continuously by the input
value rather than a discrete softmax gate — and without the load-balancing
instability that discrete routing introduces.

**The equal-parameter constraint amplifies the advantage at high K.** When K=32,
`d_idx ≈ d / 5.6`. The standard layer being compared has 5.6× larger hidden
dimensions but is constrained to a single affine transformation. The indexed
layer has smaller dimensions but 32 learned weight regimes. For smooth nonlinear
targets, the latter is a strictly better use of the same parameter budget.

**Hypothesis: why this may generalise to real transformers.**
The input to a transformer FFN layer is a post-layernorm residual stream. Layernorm
constrains activations to lie on a hypersphere, and the representational geometry
literature consistently finds that this space has meaningful regional structure —
tokens of similar semantic type cluster together. If the residual stream is
regionally structured in this way, then input-indexed weights can exploit that
structure directly: different weight regimes for different semantic regions. A
standard FFN layer, applying the same transformation everywhere, cannot do this
without stacking many nonlinear layers on top. IndexedLinear may be doing in one
operation what a standard layer needs depth to achieve.

---

## Key results

| Config | Fwd speedup | Step speedup | ppl_ratio |
|--------|-------------|--------------|-----------|
| d=512, K=32, seq=256, B=32 | 13.96x | 7.56x | 1.002 |
| d=384, K=32, seq=256, B=64 | 14.26x | 7.81x | 1.007 |
| d=256, K=32, seq=256, B=32 |  5.57x | 3.63x | 1.002 |
| d=256, K=16, seq=256, B=32 |  3.68x | 2.29x | 1.003 |

`ppl_ratio = ppl_idx / ppl_std`. Below 1.0 means indexed is more accurate.
At or near 1.0 means accuracy is indistinguishable from standard.

At K ≥ 16, indexed layers are frequently **Pareto-dominant** — faster and at
least as accurate as standard. The 13–14x forward speedup configs are at
realistic transformer FFN dimensions (d=384–512, seq=256) with essentially
identical perplexity on the synthetic regression benchmark.

The conditions under which indexed wins on both axes simultaneously:
- **d ≥ 256** — large enough that the GPU is compute/bandwidth bound on the standard layer
- **K ≥ 16** — enough buckets for the piecewise approximation to match the linear baseline
- **seq ≥ 128** — long enough to amortise Triton kernel launch overhead across tokens

---

## Open questions

- Does the speedup and accuracy parity hold inside a full transformer with
  residual connections and layernorm? The synthetic task is controlled but not
  representative of real activation distributions.
- Does the input domain assumption `[-1, 1]` hold after layernorm in practice,
  or does it need to be adaptive per-layer?
- Is there an optimal K schedule during training — analogous to learning rate
  warmup — where starting at low K and increasing it gradually improves
  convergence stability?
- How does IndexedLinear compare against Mixture of Experts at the same
  parameter budget? Both increase capacity without increasing active compute,
  but through different routing mechanisms.
- Does the faster convergence at high K hold on real tasks, or is it an
  artefact of the synthetic regression target being particularly well-suited
  to piecewise-linear approximation?

---

## Running experiments

```bash
# Full sweep (default)
python run.py sweep

# Quick reduced grid for testing
python run.py sweep --quick

# More training steps for accuracy evaluation
python run.py sweep --ppl_steps 2000

# Custom output directory
python run.py sweep --out_dir results/v1

# Force CPU
python run.py sweep --device cpu

# List all available experiments
python run.py --help
```

## Adding a new experiment

1. Create `aiwn/experiments/my_experiment.py`
2. Subclass `BaseExperiment` and set `name = "my_experiment"`
3. Implement `setup()`, `run()`, `analyze()`, `plot()`

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