"""
Housing Test Experiment — IndexedLinear vs Standard on California Housing.

8 purely continuous features, regression task, strong regional structure.
Latitude/longitude alone create obvious regional patterns — Bay Area, LA,
Central Valley, rural areas have fundamentally different price relationships.

A single global linear layer cannot represent this regional variation.
IndexedLinear with K local weight regimes should capture it directly.

Hypothesis:
  Standard Linear(8, 1) — one global fit — will have high MSE because
  it cannot represent the different price dynamics across regions.

  IndexedLinear(8, 1, K) — K regional fits — should achieve lower MSE
  by specializing each bucket to a different part of the feature space.

Usage:
    python run.py housing_test
    python run.py housing_test --K 32
    python run.py housing_test --K 64 --epochs 50
    python run.py housing_test --model_type all
"""

import argparse
from pathlib import Path

import torch

from aiwn.experiments.base import BaseExperiment
from aiwn.training.housing import (
    load_housing, StandardModel, IndexedModelV1,
    IndexedModelV2, train_and_eval
)


class HousingTestExperiment(BaseExperiment):
    name = "housing_test"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        g = parser.add_argument_group("housing_test")
        g.add_argument('--K',            type=int,   default=32)
        g.add_argument('--epochs',       type=int,   default=30)
        g.add_argument('--batch_size',   type=int,   default=256)
        g.add_argument('--lr',           type=float, default=1e-3)
        g.add_argument('--weight_decay', type=float, default=1e-4)
        g.add_argument('--model_type',   default='both',
                       choices=['standard', 'v1', 'v2', 'both', 'all'])
        g.add_argument('--val_split',    type=float, default=0.1)
        g.add_argument('--test_split',   type=float, default=0.1)
        g.add_argument('--seed',         type=int,   default=42)

    def setup(self, args, device):
        self.device = device
        self.args   = args
        torch.manual_seed(args.seed)

    def run(self) -> dict:
        args   = self.args
        device = self.device

        print(f"\nLoading California Housing...")
        train_data, val_data, test_data, n_features, y_min, y_max = load_housing(
            device=device, val_split=args.val_split,
            test_split=args.test_split, seed=args.seed)

        self.y_min = y_min
        self.y_max = y_max
        results = {}

        if args.model_type in ('standard', 'both', 'all'):
            print(f"\nBuilding standard model...")
            model = StandardModel(n_features=n_features).to(device)
            results['standard'] = train_and_eval(
                model=model, train_data=train_data,
                val_data=val_data, test_data=test_data,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, weight_decay=args.weight_decay,
                device=device, label="Standard Linear(8→1)")
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if args.model_type in ('v1', 'all'):
            print(f"\nBuilding V1 model (K={args.K}, no CDF)...")
            model = IndexedModelV1(n_features=n_features, K=args.K).to(device)
            results['v1'] = train_and_eval(
                model=model, train_data=train_data,
                val_data=val_data, test_data=test_data,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, weight_decay=args.weight_decay,
                device=device, label=f"IndexedLinear V1 (K={args.K})")
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if args.model_type in ('v2', 'both', 'all'):
            for K in ([args.K] if args.model_type != 'all' else [16, 32, 64, 128]):
                print(f"\nBuilding V2 model (K={K}, Gaussian CDF)...")
                model = IndexedModelV2(n_features=n_features, K=K).to(device)
                results[f'v2_K{K}'] = train_and_eval(
                    model=model, train_data=train_data,
                    val_data=val_data, test_data=test_data,
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, weight_decay=args.weight_decay,
                    device=device, label=f"IndexedLinear V2 (K={K})")
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        return results

    def analyze(self, results: dict):
        y_range = self.y_max - self.y_min

        print(f"\n{'='*65}")
        print("CALIFORNIA HOUSING TEST RESULTS")
        print(f"{'='*65}")
        print(f"  (target normalized to [0,1], original range "
              f"${self.y_min*100:.0f}k–${self.y_max*100:.0f}k)")

        std_r = results.get('standard')

        print(f"\n  {'Model':<32} {'Test MSE':>9} {'Test MAE':>9} "
              f"{'vs Std':>9} {'ms/step':>8}")
        print(f"  {'-'*69}")

        for key, r in results.items():
            vs_std = ""
            if std_r and key != 'standard':
                # MSE improvement — lower is better
                pct = (std_r['test_mse'] - r['test_mse']) / std_r['test_mse'] * 100
                vs_std = f"{pct:+.1f}%"
            print(f"  {r['label']:<32} "
                  f"{r['test_mse']:>9.4f} "
                  f"{r['test_mae']:>9.4f} "
                  f"{vs_std:>9} "
                  f"{r['avg_step_ms']:>7.3f}ms")

        if std_r:
            best = min(
                ((k, v) for k, v in results.items() if k != 'standard'),
                key=lambda x: x[1]['test_mse'],
                default=(None, None))
            if best[0]:
                pct = (std_r['test_mse'] - best[1]['test_mse']) / \
                       std_r['test_mse'] * 100
                print(f"\n  Best indexed MSE improvement: {pct:.1f}%")
                if pct > 10:
                    print(f"  ✓ Strong regional structure captured by indexed layer.")
                elif pct > 0:
                    print(f"  ✓ Modest improvement — some regional structure captured.")
                else:
                    print(f"  ✗ Standard layer competitive — try more K or epochs.")

    def plot(self, results: dict, out_path: Path):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0d0d0d')
        palette = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f',
                   '#9b59b6', '#1abc9c']

        for ax in axes:
            ax.set_facecolor('#111')
            ax.tick_params(colors='#aaa')
            ax.spines[:].set_color('#333')

        for i, (key, r) in enumerate(results.items()):
            color  = palette[i % len(palette)]
            epochs = [c[0] for c in r['curve']]
            mses   = [c[2] for c in r['curve']]
            losses = [c[1] for c in r['curve']]
            axes[0].plot(epochs, mses,   'o-', color=color,
                         lw=2, ms=4, label=r['label'])
            axes[1].plot(epochs, losses, 'o-', color=color,
                         lw=2, ms=4, label=r['label'])

        axes[0].set_xlabel("Epoch", color='#aaa')
        axes[0].set_ylabel("Val MSE", color='#aaa')
        axes[0].set_title("Validation MSE — California Housing\n(lower is better)",
                          color='white')
        axes[0].legend(facecolor='#111', labelcolor='white', fontsize=7)

        axes[1].set_xlabel("Epoch", color='#aaa')
        axes[1].set_ylabel("Train Loss", color='#aaa')
        axes[1].set_title("Training Loss", color='white')
        axes[1].legend(facecolor='#111', labelcolor='white', fontsize=7)

        fig.suptitle(
            "AIWN Housing Test — 8 continuous features, regression\n"
            "Hypothesis: regional price structure captured by indexed layer",
            color='white', fontsize=11)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
        print(f"Saved {out_path}")