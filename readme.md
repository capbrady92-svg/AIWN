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

## Running experiments

```bash
# Full sweep (default)
python run.py sweep

# Quick reduced grid for testing
python run.py sweep --quick

# More training for accuracy evaluation
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
        # your experiment logic here
        # use bench_layer() for timing, run_perplexity() for accuracy
        return {'results': [...]}

    def analyze(self, results):
        print("My findings:", ...)

    def plot(self, results, out_path):
        # save figure to out_path
        ...
```

Then run with:
```bash
python run.py my_experiment --my_param 42
```

## Key results so far

| Config | Fwd speedup | Step speedup | ppl_ratio |
|--------|-------------|--------------|-----------|
| d=512, K=32, seq=256, B=32 | 13.96x | 7.56x | 1.002 |
| d=384, K=32, seq=256, B=64 | 14.26x | 7.81x | 1.007 |
| d=256, K=32, seq=256, B=32 | 5.57x  | 3.63x | 1.002 |

At K≥16, indexed layers are frequently **faster AND more accurate** than
standard (ppl_ratio < 1.0), because the input-dependent weighting acts as
a nonlinear function approximator that fits the tanh target better than a
fixed weight matrix.