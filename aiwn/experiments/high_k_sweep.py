"""
High-K Experiment — testing AIWN at K=64, 128, 256, 512

Tests whether the sqrt(K) dimension scaling assumption breaks down at high K,
and whether the piecewise-linear resolution hypothesis holds:
  - More buckets = finer regional specialization
  - Each bucket covers a simpler function → needs less width
  - Speedup and accuracy may both improve beyond K=32

Usage:
  python run.py high_k
  python run.py high_k --fixed_dim   (bypasses sqrt(K) scaling — keeps d_idx = d_std)

Key questions:
  1. Does speedup continue scaling past K=32, or does kernel overhead dominate?
  2. Does accuracy hold or improve at K=512 despite small d_idx?
  3. If --fixed_dim: does accuracy improve dramatically with more buckets at full width?
"""

import argparse
import math
from itertools import product
from pathlib import Path

import numpy as np
import torch

from aiwn.experiments.base import BaseExperiment
from aiwn.layers import IndexedLinear, StandardLinear, indexed_dims
from aiwn.bench import bench_layer
from aiwn.training import run_perplexity


class HighKExperiment(BaseExperiment):
    name = "high_k"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        g = parser.add_argument_group("high_k")
        g.add_argument('--fixed_dim', action='store_true',
                       help='Keep d_idx = d_std regardless of K (test resolution hypothesis)')
        g.add_argument('--n_bench',        type=int,   default=200)
        g.add_argument('--n_warm',         type=int,   default=50)
        g.add_argument('--ppl_steps',      type=int,   default=500)
        g.add_argument('--ppl_lr',         type=float, default=1e-3)
        g.add_argument('--ppl_batch',      type=int,   default=256)
        g.add_argument('--ppl_n_data',     type=int,   default=4096)
        g.add_argument('--ppl_ckpt_every', type=int,   default=50)

    def setup(self, args, device):
        self.device = device
        self.args = args

        # High-K values — the interesting unknowns
        self.K_VALUES    = [512]   # 32 as baseline for comparison
        # Large d and batch — where AIWN dominates (above crossover regime)
        self.D_MODELS    = [512]
        self.BATCH_SIZES = [128]
        self.SEQ_LENS    = [4096]
        self.N_BENCH     = args.n_bench
        self.N_WARM      = args.n_warm
        self.PPL_STEPS   = args.ppl_steps
        self.N_HEADS     = 4
        self.fixed_dim   = args.fixed_dim

    def _get_dims(self, d_std, K):
        """Return (d_idx, ff_idx, h_dn).
        If --fixed_dim: bypass sqrt(K) scaling, keep d_idx = d_std.
        This tests the hypothesis that higher K doesn't need smaller d.
        """
        if self.fixed_dim:
            # No dimension reduction — test pure resolution scaling
            return d_std, 4 * d_std, 4 * d_std
        else:
            return indexed_dims(d_std, 4 * d_std, K, self.N_HEADS)

    def run(self) -> dict:
        mode = "FIXED_DIM (resolution hypothesis)" if self.fixed_dim else "SQRT_K scaling (standard)"
        print(f"\nHigh-K Experiment — mode: {mode}")
        print(f"K values : {self.K_VALUES}")
        print(f"d_models : {self.D_MODELS}")
        print(f"Batches  : {self.BATCH_SIZES}")
        print(f"Seqs     : {self.SEQ_LENS}")

        if self.fixed_dim:
            print("\nNOTE: fixed_dim mode — d_idx = d_std regardless of K.")
            print("Parameter count is NOT matched to standard. This tests expressiveness only.")
        else:
            print("\nNOTE: standard sqrt(K) scaling — parameter counts matched.")
            print("At K=512, d_idx ≈ d/22. Testing if resolution compensates for smaller width.")

        print("\n" + "=" * 110)
        print(f"{'B':>4} {'d':>4} {'K':>4} {'seq':>4} {'d_idx':>6} | "
              f"{'fwd_std':>8} {'fwd_idx':>8} {'speedup':>8} | "
              f"{'stp_std':>8} {'stp_idx':>8} {'stp_spd':>8} | "
              f"{'mse_std':>8} {'mse_idx':>8} {'ppl_r':>6}")
        print("-" * 110)

        ppl_cache = {}
        results   = []

        for B, d_std, K, seq in product(
                self.BATCH_SIZES, self.D_MODELS, self.K_VALUES, self.SEQ_LENS):

            d_idx, ff_idx, h_dn = self._get_dims(d_std, K)
            N = B * seq

            try:
                layer_std = StandardLinear(d_std, 4 * d_std).to(self.device)
                layer_idx = IndexedLinear(h_dn, d_idx, K).to(self.device)
            except Exception as e:
                print(f"  SKIP (layer init) B={B} d={d_std} K={K}: {e}")
                continue

            # Perplexity — cached per (d, K, fixed_dim)
            cache_key = (d_std, K, self.fixed_dim)
            if cache_key not in ppl_cache:
                try:
                    if self.fixed_dim:
                        # Custom run_perplexity with fixed dims
                        ppl_cache[cache_key] = run_perplexity(
                            d_std           = d_std,
                            K               = K,
                            device          = self.device,
                            steps           = self.PPL_STEPS,
                            lr              = self.args.ppl_lr,
                            batch_size      = self.args.ppl_batch,
                            n_data          = self.args.ppl_n_data,
                            ckpt_every      = self.args.ppl_ckpt_every,
                            indexed_dims_fn = lambda d, ff, k, nh=4: (d, ff, ff),
                        )
                    else:
                        ppl_cache[cache_key] = run_perplexity(
                            d_std           = d_std,
                            K               = K,
                            device          = self.device,
                            steps           = self.PPL_STEPS,
                            lr              = self.args.ppl_lr,
                            batch_size      = self.args.ppl_batch,
                            n_data          = self.args.ppl_n_data,
                            ckpt_every      = self.args.ppl_ckpt_every,
                            indexed_dims_fn = indexed_dims,
                        )
                except Exception as e:
                    print(f"  SKIP (ppl) d={d_std} K={K}: {e}")
                    continue

            try:
                r = bench_layer(
                    layer_std, layer_idx, N, d_std, 4 * d_std, K,
                    self.N_WARM, self.N_BENCH, self.device)
            except Exception as e:
                print(f"  SKIP (bench) B={B} d={d_std} K={K} seq={seq}: {e}")
                continue

            pm = ppl_cache[cache_key]
            pareto_win = (r['fwd_speedup'] >= 3.0 and pm['ppl_ratio'] <= 1.01)

            r.update({
                'B': B, 'd_std': d_std, 'd_idx': d_idx, 'K': K, 'seq': seq,
                'N': N, 'fixed_dim': self.fixed_dim,
                'params_std': layer_std.lin.weight.numel() + layer_std.lin.bias.numel(),
                'params_idx': layer_idx.table.numel()      + layer_idx.bias.numel(),
                'ppl_ratio':   pm['ppl_ratio'],
                'val_mse_std': pm['val_mse_std'],
                'val_mse_idx': pm['val_mse_idx'],
                'pareto_win':  pareto_win,
            })
            results.append(r)

            flag = ' ★' if pareto_win else ''
            print(f"{B:4d} {d_std:4d} {K:4d} {seq:4d} {d_idx:6d} | "
                  f"{r['fwd_std_ms']:7.3f}ms {r['fwd_idx_ms']:7.3f}ms "
                  f"{r['fwd_speedup']:7.2f}x | "
                  f"{r['step_std_ms']:7.3f}ms {r['step_idx_ms']:7.3f}ms "
                  f"{r['step_speedup']:7.2f}x | "
                  f"{pm['val_mse_std']:8.5f} {pm['val_mse_idx']:8.5f} "
                  f"{pm['ppl_ratio']:6.3f}{flag}")

        return {'results': results, 'ppl_cache': ppl_cache}

    def analyze(self, results: dict):
        rows = results['results']
        ppl_cache = results['ppl_cache']

        print("\n" + "=" * 70)
        print("HIGH-K FINDINGS")
        print("=" * 70)

        # K scaling per d_model
        for d in self.D_MODELS:
            print(f"\nK scaling — d={d}, B=32, seq=256:")
            print(f"  {'K':>4} {'d_idx':>6} {'fwd_spd':>9} {'stp_spd':>9} "
                  f"{'ppl_r':>8} {'params_ratio':>13} {'pareto':>8}")
            sub = [r for r in rows if r['d_std']==d and r['B']==32 and r['seq']==256]
            for r in sorted(sub, key=lambda x: x['K']):
                pratio = r['params_idx'] / r['params_std']
                flag = '★' if r['pareto_win'] else ''
                print(f"  {r['K']:>4} {r['d_idx']:>6} {r['fwd_speedup']:>8.2f}x "
                      f"{r['step_speedup']:>8.2f}x {r['ppl_ratio']:>8.4f} "
                      f"{pratio:>12.3f}x {flag:>8}")

        # Best configs overall
        pareto = [r for r in rows if r['pareto_win']]
        print(f"\nPareto wins (fwd ≥3x AND ppl ≤1.01): {len(pareto)}")
        if pareto:
            best = sorted(pareto, key=lambda x: -x['fwd_speedup'])[:10]
            print(f"  {'B':>4} {'d':>4} {'K':>4} {'seq':>4} {'d_idx':>6} "
                  f"{'fwd_spd':>9} {'ppl_r':>8}")
            for r in best:
                print(f"  {r['B']:>4} {r['d_std']:>4} {r['K']:>4} {r['seq']:>4} "
                      f"{r['d_idx']:>6} {r['fwd_speedup']:>8.2f}x {r['ppl_ratio']:>8.4f}")

        # Key question: does K=512 beat K=32?
        print("\nK=32 vs K=512 head-to-head (B=64, seq=256):")
        print(f"  {'d':>4} {'K':>4} {'fwd_spd':>9} {'ppl_r':>8} {'verdict':>12}")
        for d in self.D_MODELS:
            for K in [32, 512]:
                sub = [r for r in rows if r['d_std']==d and r['K']==K
                       and r['B']==64 and r['seq']==256]
                if sub:
                    r = sub[0]
                    verdict = '★ BETTER' if r['pareto_win'] else ('fast' if r['fwd_speedup']>1 else 'slower')
                    print(f"  {d:>4} {K:>4} {r['fwd_speedup']:>8.2f}x "
                          f"{r['ppl_ratio']:>8.4f} {verdict:>12}")

    def plot(self, results: dict, out_path: Path):
        rows = results['results']
        ppl_cache = results['ppl_cache']

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plot")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor('#0d0d0d')
        CGRN = '#2ecc71'; CRED = '#e74c3c'; CBLU = '#3498db'; CYEL = '#f1c40f'
        colors_d = [CBLU, CGRN, CYEL]

        for ax in axes:
            ax.set_facecolor('#111')
            ax.tick_params(colors='#aaa')
            ax.spines[:].set_color('#333')

        # Plot 1: K vs forward speedup
        for i, d in enumerate(self.D_MODELS):
            sub = sorted([(r['K'], r['fwd_speedup']) for r in rows
                          if r['d_std']==d and r['B']==64 and r['seq']==256])
            if sub:
                ks, spds = zip(*sub)
                axes[0].plot(ks, spds, 'o-', color=colors_d[i], lw=2.5, ms=8, label=f'd={d}')
        axes[0].axhline(1.0, color=CRED, ls='--', lw=1.5, label='break-even')
        axes[0].set_xlabel("K", color='#aaa')
        axes[0].set_ylabel("Forward speedup", color='#aaa')
        axes[0].set_title("K vs Forward Speedup\n(B=64, seq=256)", color='white')
        axes[0].legend(facecolor='#111', labelcolor='white')

        # Plot 2: K vs ppl_ratio
        for i, d in enumerate(self.D_MODELS):
            pts = sorted([(K, ppl_cache[(d, K, self.fixed_dim)]['ppl_ratio'])
                          for K in self.K_VALUES if (d, K, self.fixed_dim) in ppl_cache])
            if pts:
                ks, ratios = zip(*pts)
                axes[1].plot(ks, ratios, 'o-', color=colors_d[i], lw=2.5, ms=8, label=f'd={d}')
        axes[1].axhline(1.0, color=CRED, ls='--', lw=1.5, label='break-even')
        axes[1].set_xlabel("K", color='#aaa')
        axes[1].set_ylabel("ppl_ratio (idx/std)", color='#aaa')
        axes[1].set_title("K vs Perplexity Ratio\n(<1.0 = indexed better)", color='white')
        axes[1].legend(facecolor='#111', labelcolor='white')

        # Plot 3: K vs d_idx (show dimension collapse)
        for i, d in enumerate(self.D_MODELS):
            sub = sorted([(r['K'], r['d_idx']) for r in rows if r['d_std']==d])
            if sub:
                ks, dims = zip(*sub)
                axes[2].plot(ks, dims, 'o-', color=colors_d[i], lw=2.5, ms=8, label=f'd={d}')
                axes[2].axhline(d, color=colors_d[i], ls=':', lw=1, alpha=0.4)
        axes[2].set_xlabel("K", color='#aaa')
        axes[2].set_ylabel("d_idx (actual hidden dim)", color='#aaa')
        axes[2].set_title("K vs d_idx\n(dotted = original d_std)", color='white')
        axes[2].legend(facecolor='#111', labelcolor='white')

        mode = "fixed_dim" if self.fixed_dim else "sqrt_K_scaling"
        fig.suptitle(f"AIWN High-K Experiment — {mode} — {self.device}",
                     color='white', fontsize=13)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
        print(f"Saved {out_path}")