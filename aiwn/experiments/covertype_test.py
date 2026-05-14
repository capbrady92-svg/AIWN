"""
Covertype Test Experiment — IndexedLinear vs StandardLinear on regional data.

Tests whether IndexedLinear's piecewise structure outperforms a standard
linear layer on a real dataset with strong regional structure.

Covertype predicts forest cover type (7 classes) from 54 cartographic
features. Different elevation bands, soil types, and wilderness areas
have fundamentally different feature-to-class mappings — exactly the
kind of regional structure IndexedLinear is designed to exploit.

Hypothesis:
  Standard Linear(54, 7) — one global decision boundary — will plateau
  at lower accuracy because it structurally cannot represent the regional
  variation in the data.

  IndexedLinear(54, 7, K) — K local decision boundaries — should achieve
  higher accuracy by specializing each bucket to a different input region.

Usage:
    python run.py covertype_test
    python run.py covertype_test --K 32
    python run.py covertype_test --K 64
    python run.py covertype_test --K 128
    python run.py covertype_test --epochs 30
"""

import argparse
from pathlib import Path

import torch

from aiwn.experiments.base import BaseExperiment
from aiwn.training.covertype import (
    load_covertype, StandardModel, IndexedModel, IndexedModelV2, train_and_eval
)


class CovertypeTestExperiment(BaseExperiment):
    name = "covertype_test"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        g = parser.add_argument_group("covertype_test")
        g.add_argument('--K',            type=int,   default=32,
                       help='Number of buckets for IndexedLinear')
        g.add_argument('--epochs',       type=int,   default=20)
        g.add_argument('--batch_size',   type=int,   default=1024)
        g.add_argument('--lr',           type=float, default=1e-3)
        g.add_argument('--weight_decay', type=float, default=1e-4)
        g.add_argument('--model_type',   default='both',
                       choices=['standard', 'indexed', 'indexed_v2', 'both', 'all'])
        g.add_argument('--entropy_weight', type=float, default=0.01,
                       help='CDF entropy regularization weight for V2')
        g.add_argument('--val_split',    type=float, default=0.1)
        g.add_argument('--test_split',   type=float, default=0.1)
        g.add_argument('--seed',         type=int,   default=42)

    def setup(self, args: argparse.Namespace, device: torch.device):
        self.device = device
        self.args   = args
        torch.manual_seed(args.seed)

    def run(self) -> dict:
        args   = self.args
        device = self.device

        print(f"\nLoading Covertype dataset...")
        train_data, val_data, test_data, n_features, n_classes = load_covertype(
            device     = device,
            val_split  = args.val_split,
            test_split = args.test_split,
            seed       = args.seed,
        )

        results = {}

        if args.model_type in ('standard', 'both'):
            print(f"\nBuilding standard model...")
            model_std = StandardModel(
                n_features=n_features, n_classes=n_classes).to(device)
            results['standard'] = train_and_eval(
                model        = model_std,
                train_data   = train_data,
                val_data     = val_data,
                test_data    = test_data,
                epochs       = args.epochs,
                batch_size   = args.batch_size,
                lr           = args.lr,
                weight_decay = args.weight_decay,
                device       = device,
                label        = f"Standard Linear(54→7)",
            )
            del model_std
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if args.model_type in ('indexed', 'both'):
            print(f"\nBuilding indexed model (K={args.K})...")
            model_idx = IndexedModel(
                n_features=n_features, n_classes=n_classes, K=args.K).to(device)
            results['indexed'] = train_and_eval(
                model        = model_idx,
                train_data   = train_data,
                val_data     = val_data,
                test_data    = test_data,
                epochs       = args.epochs,
                batch_size   = args.batch_size,
                lr           = args.lr,
                weight_decay = args.weight_decay,
                device       = device,
                label        = f"Indexed Linear(54→7, K={args.K})",
            )
            del model_idx
            if device.type == 'cuda':
                torch.cuda.empty_cache()


        if args.model_type in ('indexed_v2', 'all'):
            print(f"\nBuilding indexed V2 model (K={args.K}, entropy_weight={args.entropy_weight})...")
            model_v2 = IndexedModelV2(
                n_features=n_features, n_classes=n_classes,
                K=args.K, entropy_weight=args.entropy_weight).to(device)
            results['indexed_v2'] = train_and_eval(
                model        = model_v2,
                train_data   = train_data,
                val_data     = val_data,
                test_data    = test_data,
                epochs       = args.epochs,
                batch_size   = args.batch_size,
                lr           = args.lr,
                weight_decay = args.weight_decay,
                device       = device,
                label        = f"Indexed V2 CDF (K={args.K})",
            )
            del model_v2
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        return results

    def analyze(self, results: dict):
        print(f"\n{'='*60}")
        print("COVERTYPE TEST RESULTS")
        print(f"{'='*60}")

        if 'standard' in results and 'indexed' in results:
            std_r = results['standard']
            idx_r = results['indexed']

            acc_ratio    = idx_r['test_acc'] / std_r['test_acc']
            step_speedup = std_r['avg_step_ms'] / idx_r['avg_step_ms']

            print(f"\n  {'Metric':<25} {'Standard':>12} {'Indexed':>12} {'Ratio':>8}")
            print(f"  {'-'*59}")
            print(f"  {'Best val accuracy':<25} "
                  f"{std_r['best_val_acc']*100:>11.2f}% "
                  f"{idx_r['best_val_acc']*100:>11.2f}% "
                  f"{idx_r['best_val_acc']/std_r['best_val_acc']:>7.3f}x")
            print(f"  {'Test accuracy':<25} "
                  f"{std_r['test_acc']*100:>11.2f}% "
                  f"{idx_r['test_acc']*100:>11.2f}% "
                  f"{acc_ratio:>7.3f}x")
            print(f"  {'Avg step ms':<25} "
                  f"{std_r['avg_step_ms']:>12.3f} "
                  f"{idx_r['avg_step_ms']:>12.3f} "
                  f"{step_speedup:>7.2f}x")

            print(f"\n  acc_ratio:    {acc_ratio:.4f} "
                  f"({'indexed better ✓' if acc_ratio > 1 else 'standard better'})")
            print(f"  step_speedup: {step_speedup:.2f}x "
                  f"({'indexed faster ✓' if step_speedup > 1 else 'standard faster'})")

            # Interpret the result
            print(f"\n  Interpretation:")
            if acc_ratio > 1.05:
                print(f"  ✓ IndexedLinear outperforms standard by "
                      f"{(acc_ratio-1)*100:.1f}% — regional structure confirmed.")
            elif acc_ratio > 1.0:
                print(f"  ✓ IndexedLinear marginally better — "
                      f"some regional structure captured.")
            else:
                print(f"  ✗ Standard better — regional structure not captured "
                      f"at K={self.args.K}, or task doesn't benefit from "
                      f"piecewise approximation at this scale.")

        else:
            for key, r in results.items():
                print(f"\n  {r['label']}")
                print(f"    Test acc  : {r['test_acc']*100:.2f}%")
                print(f"    Step time : {r['avg_step_ms']:.3f}ms")

    def plot(self, results: dict, out_path: Path):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0d0d0d')
        CGRN = '#2ecc71'; CBLU = '#3498db'

        for ax in axes:
            ax.set_facecolor('#111')
            ax.tick_params(colors='#aaa')
            ax.spines[:].set_color('#333')

        colors = {'standard': CBLU, 'indexed': CGRN}

        # Val accuracy over epochs
        for key, r in results.items():
            epochs = [c[0] for c in r['curve']]
            accs   = [c[2] * 100 for c in r['curve']]
            axes[0].plot(epochs, accs, 'o-', color=colors[key],
                         lw=2.5, ms=5, label=r['label'])
        axes[0].set_xlabel("Epoch", color='#aaa')
        axes[0].set_ylabel("Val Accuracy (%)", color='#aaa')
        axes[0].set_title("Validation Accuracy — Covertype", color='white')
        axes[0].legend(facecolor='#111', labelcolor='white', fontsize=8)

        # Training loss over epochs
        for key, r in results.items():
            epochs = [c[0] for c in r['curve']]
            losses = [c[1] for c in r['curve']]
            axes[1].plot(epochs, losses, 'o-', color=colors[key],
                         lw=2.5, ms=5, label=r['label'])
        axes[1].set_xlabel("Epoch", color='#aaa')
        axes[1].set_ylabel("Train Loss", color='#aaa')
        axes[1].set_title("Training Loss — Covertype", color='white')
        axes[1].legend(facecolor='#111', labelcolor='white', fontsize=8)

        K = self.args.K
        fig.suptitle(
            f"AIWN Covertype Test — K={K} — {self.device}\n"
            f"Hypothesis: IndexedLinear captures regional structure "
            f"that standard linear cannot",
            color='white', fontsize=11)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
        print(f"Saved {out_path}")