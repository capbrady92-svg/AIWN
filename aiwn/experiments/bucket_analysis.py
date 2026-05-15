"""
Bucket Utilization Analysis — which buckets are actually being used?

Hooks into IndexedLinear and IndexedLinearV2 to track which buckets
receive gradient updates during a forward pass.

Usage:
    python run.py bucket_analysis --K 32
    python run.py bucket_analysis --K 128

Drop into aiwn/experiments/bucket_analysis.py
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from aiwn.experiments.base import BaseExperiment
from aiwn.layers import IndexedLinear
from aiwn.training.covertype import load_covertype

try:
    from aiwn.layers.indexed_linear_v2 import IndexedLinearV2
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False


def get_bucket_counts(layer, x: torch.Tensor, K: int) -> torch.Tensor:
    """
    Run a forward pass and count how many samples hit each bucket
    across all input dimensions.

    Returns bucket_counts: (K,) tensor of hit counts
    """
    with torch.no_grad():
        # Handle both IndexedLinear (in_dim) and IndexedLinearV2 (in_d)
        in_d = getattr(layer, 'in_d', None) or getattr(layer, 'in_dim', None) or layer.linear.in_dim
        xf = x.reshape(-1, in_d)
        if isinstance(layer, IndexedLinearV2) and hasattr(layer, 'cdf'):
            xf = layer.cdf(xf)

        bw  = 2.0 / K
        bk  = ((xf + 1) / bw).long().clamp(0, K - 1)  # (N, in_d)
        # Count hits per bucket across all samples and dimensions
        counts = torch.bincount(bk.reshape(-1), minlength=K).float()
        return counts


class BucketAnalysisExperiment(BaseExperiment):
    name = "bucket_analysis"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        g = parser.add_argument_group("bucket_analysis")
        g.add_argument('--K',            type=int,   default=32)
        g.add_argument('--n_samples',    type=int,   default=10000,
                       help='Number of samples to analyze')
        g.add_argument('--entropy_weight', type=float, default=0.01)
        g.add_argument('--dataset',      default='covertype',
                       choices=['covertype', 'housing'])
        g.add_argument('--seed',         type=int,   default=42)

    def setup(self, args, device):
        self.device = device
        self.args   = args
        torch.manual_seed(args.seed)

    def run(self) -> dict:
        args   = self.args
        device = self.device

        if args.dataset == 'housing':
            from aiwn.training.housing import load_housing
            print(f"\nLoading California Housing...")
            train_data, val_data, test_data, n_features, _, _ = load_housing(
                device=device, seed=args.seed)
            n_classes = 1
        else:
            print(f"\nLoading Covertype...")
            train_data, val_data, test_data, n_features, n_classes = load_covertype(
                device=device, seed=args.seed)

        # Use a subset for analysis
        x = train_data[0][:args.n_samples]
        print(f"Analyzing {len(x):,} samples, K={args.K}")

        results = {}

        # ── Raw input (no layer) ──────────────────────────────────────────────
        bw  = 2.0 / args.K
        bk  = ((x + 1) / bw).long().clamp(0, args.K - 1)
        raw_counts = torch.bincount(bk.reshape(-1), minlength=args.K).float()
        results['raw'] = raw_counts.cpu()

        # ── IndexedLinear (no CDF) ────────────────────────────────────────────
        layer_v1 = IndexedLinear(n_features, n_classes, args.K).to(device)
        v1_counts = get_bucket_counts(layer_v1, x, args.K)
        results['indexed_v1'] = v1_counts.cpu()

        # ── IndexedLinearV2 (with CDF, untrained) ────────────────────────────
        if V2_AVAILABLE:
            layer_v2_untrained = IndexedLinearV2(
                n_features, n_classes, args.K).to(device)
            # No warmup needed - Gaussian CDF has no running stats
            v2_untrained_counts = get_bucket_counts(
                layer_v2_untrained, x, args.K)
            results['indexed_v2_untrained'] = v2_untrained_counts.cpu()

            # ── IndexedLinearV2 (with CDF, trained) ──────────────────────────
            print(f"\nTraining V2 layer for 10 epochs to learn CDF...")
            layer_v2_trained = IndexedLinearV2(
                n_features, n_classes, args.K).to(device)
            optimizer = torch.optim.AdamW(
                layer_v2_trained.parameters(), lr=1e-3)
            train_x, train_y = train_data

            for epoch in range(10):
                perm = torch.randperm(len(train_x), device=device)
                for i in range(0, len(train_x), 1024):
                    idx  = perm[i:i+1024]
                    xb, yb = train_x[idx], train_y[idx]
                    pred = layer_v2_trained(xb)
                    if yb.dtype == torch.float32:
                        loss = F.mse_loss(pred.squeeze(-1), yb)
                    else:
                        loss = F.cross_entropy(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            v2_trained_counts = get_bucket_counts(
                layer_v2_trained, x, args.K)
            results['indexed_v2_trained'] = v2_trained_counts.cpu()

        return results

    def analyze(self, results: dict):
        K = self.args.K
        print(f"\n{'='*65}")
        print(f"BUCKET UTILIZATION ANALYSIS (K={K})")
        print(f"{'='*65}")

        for name, counts in results.items():
            total      = counts.sum().item()
            n_active   = (counts > 0).sum().item()
            n_dominant = (counts > counts.mean() * 2).sum().item()
            max_count  = counts.max().item()
            min_count  = counts.min().item()

            # Entropy of bucket distribution
            probs   = counts / total
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
            max_ent = math.log(K)
            ent_pct = entropy / max_ent * 100

            print(f"\n  {name}:")
            print(f"    Active buckets    : {n_active}/{K} "
                  f"({n_active/K*100:.0f}%)")
            print(f"    Dominant buckets  : {n_dominant} "
                  f"(>2x mean traffic)")
            print(f"    Max/Min hits      : {max_count:.0f} / {min_count:.0f}")
            print(f"    Distribution entropy: {entropy:.3f} / {max_ent:.3f} "
                  f"= {ent_pct:.1f}% of maximum")

            # Show top 5 and bottom 5 buckets
            top5    = counts.topk(5).indices.tolist()
            bot5    = counts.topk(5, largest=False).indices.tolist()
            print(f"    Top 5 buckets     : {top5} "
                  f"(counts: {[int(counts[i]) for i in top5]})")
            print(f"    Bottom 5 buckets  : {bot5} "
                  f"(counts: {[int(counts[i]) for i in bot5]})")

    def plot(self, results: dict, out_path: Path):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plot")
            return

        K = self.args.K
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        if n == 1:
            axes = [axes]
        fig.patch.set_facecolor('#0d0d0d')

        colors = {
            'raw':                  '#3498db',
            'indexed_v1':           '#e74c3c',
            'indexed_v2_untrained': '#f1c40f',
            'indexed_v2_trained':   '#2ecc71',
        }
        titles = {
            'raw':                  'Raw Input\n(no transform)',
            'indexed_v1':           'IndexedLinear V1\n(no CDF)',
            'indexed_v2_untrained': 'IndexedLinear V2\n(CDF untrained)',
            'indexed_v2_trained':   'IndexedLinear V2\n(CDF trained)',
        }

        for ax, (name, counts) in zip(axes, results.items()):
            ax.set_facecolor('#111')
            ax.tick_params(colors='#aaa')
            ax.spines[:].set_color('#333')

            probs   = (counts / counts.sum()).numpy()

            ax.bar(range(K), probs, color=colors.get(name, '#aaa'), alpha=0.8)
            ax.axhline(1/K, color='white', ls='--', lw=1.5,
                       label=f'Uniform (1/K)')
            ax.set_xlabel("Bucket index", color='#aaa')
            ax.set_ylabel("Fraction of hits", color='#aaa')
            ax.set_title(titles.get(name, name), color='white', fontsize=10)
            ax.legend(facecolor='#111', labelcolor='white', fontsize=8)

        fig.suptitle(f"Bucket Utilization — K={K} — Covertype Dataset",
                     color='white', fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
        print(f"Saved {out_path}")