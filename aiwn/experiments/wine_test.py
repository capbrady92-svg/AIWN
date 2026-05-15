"""
Wine Quality Test Experiment.

11 continuous chemical features, regression task.
Tests single layer and MLP variants of standard vs indexed.

Usage:
    python run.py wine_test
    python run.py wine_test --K 32 --hidden 64
    python run.py wine_test --model_type all
"""

import argparse
from pathlib import Path

import torch

from aiwn.experiments.base import BaseExperiment
from aiwn.training.wine import (
    load_wine, StandardModel, IndexedModel, train_and_eval
)


class WineTestExperiment(BaseExperiment):
    name = "wine_test"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        g = parser.add_argument_group("wine_test")
        g.add_argument('--K',            type=int,   default=32)
        g.add_argument('--hidden',       type=int,   default=64,
                       help='Hidden dim (0 = single layer)')
        g.add_argument('--epochs',       type=int,   default=30)
        g.add_argument('--batch_size',   type=int,   default=256)
        g.add_argument('--lr',           type=float, default=1e-3)
        g.add_argument('--weight_decay', type=float, default=1e-4)
        g.add_argument('--model_type',   default='both',
                       choices=['standard', 'indexed', 'both', 'all'])
        g.add_argument('--seed',         type=int,   default=42)

    def setup(self, args, device):
        self.device = device
        self.args   = args
        torch.manual_seed(args.seed)

    def run(self) -> dict:
        args, device = self.args, self.device

        print(f"\nLoading Wine Quality dataset...")
        train_data, val_data, test_data, n_features, y_min, y_max = load_wine(
            device=device, seed=args.seed)
        self.y_min, self.y_max = y_min, y_max

        results = {}
        configs = []

        if args.model_type in ('standard', 'both', 'all'):
            # Single layer standard
            configs.append(('std_single', 'standard', 0))
            if args.hidden:
                # MLP standard
                configs.append(('std_mlp', 'standard', args.hidden))

        if args.model_type in ('indexed', 'both', 'all'):
            # Single layer indexed
            configs.append(('idx_single', 'indexed', 0))
            if args.hidden:
                # MLP indexed
                configs.append(('idx_mlp', 'indexed', args.hidden))

        for key, mtype, hidden in configs:
            label_h = f"→{hidden}→" if hidden else "→"
            if mtype == 'standard':
                label = f"Standard ({n_features}{label_h}1)"
                model = StandardModel(n_features, hidden).to(device)
            else:
                label = f"Indexed K={args.K} ({n_features}{label_h}1)"
                model = IndexedModel(n_features, args.K, hidden).to(device)

            results[key] = train_and_eval(
                model=model, train_data=train_data,
                val_data=val_data, test_data=test_data,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, weight_decay=args.weight_decay,
                device=device, label=label)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return results

    def analyze(self, results: dict):
        print(f"\n{'='*60}")
        print("WINE QUALITY TEST RESULTS")
        print(f"{'='*60}")
        print(f"  (target normalized [0,1], original range "
              f"{self.y_min:.0f}–{self.y_max:.0f})")

        # Find standard baselines
        std_single = results.get('std_single')
        std_mlp    = results.get('std_mlp')

        print(f"\n  {'Model':<35} {'Test MSE':>9} {'Test MAE':>9} "
              f"{'vs Std':>9} {'ms/step':>8}")
        print(f"  {'-'*72}")

        for key, r in results.items():
            # Compare indexed against matching standard
            baseline = std_mlp if 'mlp' in key and std_mlp else std_single
            vs_std = ""
            if baseline and 'idx' in key:
                pct = (baseline['test_mse'] - r['test_mse']) / \
                       baseline['test_mse'] * 100
                vs_std = f"{pct:+.1f}%"
            print(f"  {r['label']:<35} "
                  f"{r['test_mse']:>9.4f} "
                  f"{r['test_mae']:>9.4f} "
                  f"{vs_std:>9} "
                  f"{r['avg_step_ms']:>7.3f}ms")

        # Key comparison — MLP if available
        if std_mlp and 'idx_mlp' in results:
            idx_mlp = results['idx_mlp']
            pct = (std_mlp['test_mse'] - idx_mlp['test_mse']) / \
                   std_mlp['test_mse'] * 100
            print(f"\n  MLP comparison: indexed {'better' if pct>0 else 'worse'} "
                  f"by {abs(pct):.1f}% MSE")
            if pct > 5:
                print(f"  ✓ Strong regional structure captured in hidden layer.")
            elif pct > 0:
                print(f"  ✓ Modest improvement — some regional structure.")
            else:
                print(f"  ✗ Standard MLP competitive at this K/hidden setting.")

    def plot(self, results: dict, out_path: Path):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0d0d0d')
        palette = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']

        for ax in axes:
            ax.set_facecolor('#111')
            ax.tick_params(colors='#aaa')
            ax.spines[:].set_color('#333')

        for i, (key, r) in enumerate(results.items()):
            c = palette[i % len(palette)]
            ep  = [x[0] for x in r['curve']]
            mse = [x[2] for x in r['curve']]
            ls  = [x[1] for x in r['curve']]
            axes[0].plot(ep, mse, 'o-', color=c, lw=2, ms=4, label=r['label'])
            axes[1].plot(ep, ls,  'o-', color=c, lw=2, ms=4, label=r['label'])

        axes[0].set_xlabel("Epoch", color='#aaa')
        axes[0].set_ylabel("Val MSE", color='#aaa')
        axes[0].set_title("Validation MSE — Wine Quality", color='white')
        axes[0].legend(facecolor='#111', labelcolor='white', fontsize=7)
        axes[1].set_xlabel("Epoch", color='#aaa')
        axes[1].set_ylabel("Train Loss", color='#aaa')
        axes[1].set_title("Training Loss", color='white')
        axes[1].legend(facecolor='#111', labelcolor='white', fontsize=7)

        fig.suptitle(
            f"AIWN Wine Test — K={self.args.K}, hidden={self.args.hidden}\n"
            "11 continuous features, regression",
            color='white', fontsize=11)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
        print(f"Saved {out_path}")