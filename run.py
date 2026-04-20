#!/usr/bin/env python
"""
AIWN experiment runner.

Usage
-----
  python run.py sweep                        # full sweep
  python run.py sweep --quick                # reduced grid
  python run.py sweep --ppl_steps 2000       # more training
  python run.py sweep --device cpu           # force CPU
  python run.py sweep --out_dir results/v1   # custom output dir

Adding a new experiment
-----------------------
  1. Create  aiwn/experiments/my_exp.py
  2. Subclass BaseExperiment, set  name = "my_exp"
  3. Implement setup(), run(), analyze(), plot()
  4. Run with:  python run.py my_exp

The runner discovers all BaseExperiment subclasses in aiwn/experiments/
automatically — no registration needed.
"""

import argparse
import importlib
import inspect
import sys
from pathlib import Path

# Make sure the repo root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from aiwn.experiments.base import BaseExperiment
from aiwn.config import resolve_device, print_device_info, add_common_args


def discover_experiments() -> dict[str, type]:
    """
    Import every module in aiwn/experiments/ and collect BaseExperiment subclasses.
    Returns {name: cls} mapping.
    """
    exp_dir = Path(__file__).parent / 'aiwn' / 'experiments'
    registry = {}
    for path in sorted(exp_dir.glob('*.py')):
        if path.stem.startswith('_'):
            continue
        module_name = f"aiwn.experiments.{path.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"  Warning: could not import {module_name}: {e}")
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(obj, BaseExperiment)
                    and obj is not BaseExperiment
                    and obj.name):
                registry[obj.name] = obj
    return registry


def main():
    registry = discover_experiments()

    if not registry:
        print("No experiments found in aiwn/experiments/")
        sys.exit(1)

    # ── Top-level parser: just picks the experiment name ──────────────────────
    top = argparse.ArgumentParser(
        description="AIWN experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available experiments:\n" +
               "\n".join(f"  {name}" for name in sorted(registry)),
    )
    top.add_argument('experiment', choices=sorted(registry),
                     help='Which experiment to run')

    # Parse just the experiment name first so we can build the full parser
    top_args, remaining = top.parse_known_args()
    cls = registry[top_args.experiment]

    # ── Full parser: common args + experiment-specific args ───────────────────
    full = argparse.ArgumentParser(
        description=f"AIWN — {top_args.experiment}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(full)
    cls.add_args(full)
    args = full.parse_args(remaining)

    # ── Resolve device and print info ─────────────────────────────────────────
    device = resolve_device(args.device)
    print_device_info(device)

    # ── Run ───────────────────────────────────────────────────────────────────
    exp = cls()
    exp.setup(args, device)

    results = exp.run()

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    exp.save(results, out_dir)

    # ── Analyze ───────────────────────────────────────────────────────────────
    exp.analyze(results)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = out_dir / f"{exp.name}_results.png"
    try:
        exp.plot(results, plot_path)
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == '__main__':
    main()