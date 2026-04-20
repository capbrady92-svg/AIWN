"""
BaseExperiment — interface every experiment must implement.

To add a new experiment:
  1. Create experiments/my_experiment.py
  2. Subclass BaseExperiment
  3. Implement setup(), run(), analyze(), plot()
  4. run.py will discover and dispatch it automatically

Example
-------
    from aiwn.experiments.base import BaseExperiment

    class MyExperiment(BaseExperiment):
        name = "my_experiment"

        def setup(self, args):
            ...

        def run(self):
            ...
            return results

        def analyze(self, results):
            ...

        def plot(self, results, out_path):
            ...
"""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path


class BaseExperiment(ABC):
    # Subclasses set this to the CLI name, e.g. "sweep" or "convergence"
    name: str = ""

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Add experiment-specific CLI arguments to `parser`.
        Called by run.py before parse_args().  Override to add custom flags.
        """
        pass

    @abstractmethod
    def setup(self, args: argparse.Namespace, device):
        """
        Initialise the experiment from parsed args and the resolved device.
        Store anything needed by run() as instance attributes here.
        """

    @abstractmethod
    def run(self) -> dict:
        """
        Execute the experiment.
        Returns a results dict (must be torch.save-able).
        """

    @abstractmethod
    def analyze(self, results: dict):
        """Print KEY FINDINGS to stdout."""

    @abstractmethod
    def plot(self, results: dict, out_path: Path):
        """Save plots to out_path (a .png path)."""

    def save(self, results: dict, out_dir: Path):
        """
        Default save: write sweep_results.pt and sweep_results.csv.
        Experiments can override for custom formats.
        """
        import torch, csv
        out_dir.mkdir(parents=True, exist_ok=True)
        pt_path = out_dir / f"{self.name}_results.pt"
        torch.save(results, pt_path)
        print(f"Saved {pt_path}")

        # Flat CSV — skip any list-valued fields
        rows = results.get('results', [])
        if rows:
            flat_fields = [k for k, v in rows[0].items() if not isinstance(v, list)]
            csv_path = out_dir / f"{self.name}_results.csv"
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=flat_fields)
                w.writeheader()
                w.writerows({k: r[k] for k in flat_fields} for r in rows)
            print(f"Saved {csv_path}")