"""
Sweep experiment — comprehensive indexed vs standard benchmark.

Tests every combination of:
  batch_size × d_model × K × seq_len

Measuring speed (fwd ms, step ms, speedup) and accuracy (perplexity via
synthetic regression, computed once per (d, K) pair and cached).

Usage via run.py:
  python run.py sweep
  python run.py sweep --quick
  python run.py sweep --ppl_steps 2000 --n_bench 500
"""

import argparse
import csv
import math
from itertools import product
from pathlib import Path

import numpy as np
import torch

from aiwn.experiments.base import BaseExperiment
from aiwn.layers import IndexedLinear, StandardLinear, indexed_dims
from aiwn.bench import bench_layer
from aiwn.training import run_perplexity


class SweepExperiment(BaseExperiment):
    name = "sweep"

    # ── CLI args ───────────────────────────────────────────────────────────────
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        g = parser.add_argument_group("sweep")
        g.add_argument('--quick',          action='store_true',
                       help='Reduced grid for quick testing')
        g.add_argument('--n_bench',        type=int,   default=200,
                       help='Timed iterations per config')
        g.add_argument('--n_warm',         type=int,   default=50,
                       help='Warmup iterations per config')
        g.add_argument('--ppl_steps',      type=int,   default=500,
                       help='Optimiser steps for perplexity eval per (d,K)')
        g.add_argument('--ppl_lr',         type=float, default=1e-3)
        g.add_argument('--ppl_batch',      type=int,   default=256)
        g.add_argument('--ppl_n_data',     type=int,   default=4096)
        g.add_argument('--ppl_ckpt_every', type=int,   default=50)

    # ── Setup ──────────────────────────────────────────────────────────────────
    def setup(self, args: argparse.Namespace, device: torch.device):
        self.device = device
        self.args   = args

        if args.quick:
            self.BATCH_SIZES = [1, 8, 32, 128]
            self.D_MODELS    = [32, 64, 128, 256]
            self.K_VALUES    = [4, 8, 16]
            self.SEQ_LENS    = [48]
            self.N_BENCH     = 50
            self.N_WARM      = 10
            self.PPL_STEPS   = 200
        else:
            self.BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            self.D_MODELS    = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
            self.K_VALUES    = [4, 8, 16, 32]   # K=2 dropped — too few buckets
            self.SEQ_LENS    = [16, 48, 128, 256]
            self.N_BENCH     = args.n_bench
            self.N_WARM      = args.n_warm
            self.PPL_STEPS   = args.ppl_steps

        self.N_HEADS = 4

    # ── Run ────────────────────────────────────────────────────────────────────
    def run(self) -> dict:
        n_total = (len(self.BATCH_SIZES) * len(self.D_MODELS)
                   * len(self.K_VALUES)  * len(self.SEQ_LENS))
        n_ppl   = len(self.D_MODELS) * len(self.K_VALUES)

        print(f"\nConfigs  : {n_total}  "
              f"({self.N_WARM} warmup + {self.N_BENCH} timed each)")
        print(f"Ppl pairs: {n_ppl} unique (d,K) x {self.PPL_STEPS} steps "
              f"(computed on first encounter)")
        print(f"K values : {self.K_VALUES}  (K=2 excluded — pathological capacity)")

        print("\n" + "=" * 108)
        print(f"{'B':>4} {'d':>4} {'K':>3} {'seq':>4} | "
              f"{'fwd_std':>8} {'fwd_idx':>8} {'fwd_spd':>8} | "
              f"{'stp_std':>8} {'stp_idx':>8} {'stp_spd':>8} | "
              f"{'mse_std':>8} {'mse_idx':>8} {'ppl_r':>6} | {'faster?':>8}")
        print(f"{'':54}  {'lower=better':>23} {'<1=idx':>6}")
        print("-" * 108)

        ppl_cache = {}
        results   = []
        done      = 0

        for B, d_std, K, seq in product(
                self.BATCH_SIZES, self.D_MODELS, self.K_VALUES, self.SEQ_LENS):

            d_idx, ff_idx, h_dn = indexed_dims(d_std, 4 * d_std, K, self.N_HEADS)
            layer_std = StandardLinear(d_std, 4 * d_std).to(self.device)
            layer_idx = IndexedLinear(d_idx, ff_idx, K).to(self.device)
            N = B * seq

            # Perplexity: once per (d,K), cached for all B/seq combos
            if (d_std, K) not in ppl_cache:
                ppl_cache[(d_std, K)] = run_perplexity(
                    d_std        = d_std,
                    K            = K,
                    device       = self.device,
                    steps        = self.PPL_STEPS,
                    lr           = self.args.ppl_lr,
                    batch_size   = self.args.ppl_batch,
                    n_data       = self.args.ppl_n_data,
                    ckpt_every   = self.args.ppl_ckpt_every,
                    indexed_dims_fn = indexed_dims,
                )

            try:
                r = bench_layer(
                    layer_std, layer_idx, N, d_std, 4 * d_std, K,
                    self.N_WARM, self.N_BENCH, self.device)
            except Exception as e:
                print(f"  SKIP B={B} d={d_std} K={K} seq={seq}: {e}")
                continue

            pm = ppl_cache[(d_std, K)]
            pareto_win = (r['fwd_speedup'] >= 3.0 and pm['ppl_ratio'] <= 1.01)

            r.update({
                'B': B, 'd_std': d_std, 'd_idx': d_idx, 'K': K, 'seq': seq,
                'h_dn': h_dn, 'N': N,
                'params_std':  layer_std.lin.weight.numel() + layer_std.lin.bias.numel(),
                'params_idx':  layer_idx.table.numel()      + layer_idx.bias.numel(),
                'backend':     'triton' if _triton_ok() else 'eager',
                'ppl_std':     pm['ppl_std'],
                'ppl_idx':     pm['ppl_idx'],
                'ppl_ratio':   pm['ppl_ratio'],
                'val_mse_std': pm['val_mse_std'],
                'val_mse_idx': pm['val_mse_idx'],
                'pareto_win':  pareto_win,
            })
            results.append(r)
            done += 1

            faster = (('✓ FWD'  if r['idx_faster_fwd']  else '') or
                      ('✓ STEP' if r['idx_faster_step'] else '') or '✗')
            flag = ' ★' if pareto_win else ''

            if done % 5 == 1 or r['idx_faster_fwd'] or r['idx_faster_step']:
                print(f"{B:4d} {d_std:4d} {K:3d} {seq:4d} | "
                      f"{r['fwd_std_ms']:7.3f}ms {r['fwd_idx_ms']:7.3f}ms "
                      f"{r['fwd_speedup']:7.2f}x | "
                      f"{r['step_std_ms']:7.3f}ms {r['step_idx_ms']:7.3f}ms "
                      f"{r['step_speedup']:7.2f}x | "
                      f"{pm['val_mse_std']:8.5f} {pm['val_mse_idx']:8.5f} "
                      f"{pm['ppl_ratio']:6.3f} | {faster:>8}{flag}")
            if done % 20 == 0:
                print(f"  [{done}/{n_total}]")

        return {'results': results, 'ppl_cache': ppl_cache}

    # ── Analyze ────────────────────────────────────────────────────────────────
    def analyze(self, results: dict):
        rows      = results['results']
        ppl_cache = results['ppl_cache']

        k_target   = 16  if 16  in self.K_VALUES    else self.K_VALUES[-1]
        b_target   = 32  if 32  in self.BATCH_SIZES else self.BATCH_SIZES[-1]
        d_target   = min(64, max(self.D_MODELS))
        d_k        = 128 if 128 in self.D_MODELS    else self.D_MODELS[-1]
        seq_target = 48  if 48  in self.SEQ_LENS    else self.SEQ_LENS[0]

        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        # 1. Batch crossover
        print(f"\n1. Batch size crossover (d={d_target}, K={k_target}, seq={seq_target}):")
        sub = [r for r in rows if r['d_std']==d_target
               and r['K']==k_target and r['seq']==seq_target]
        if sub:
            print(f"  {'B':>6} {'fwd_speedup':>12} {'step_speedup':>12} {'faster?':>10}")
            for r in sorted(sub, key=lambda x: x['B']):
                print(f"  {r['B']:>6} {r['fwd_speedup']:>11.2f}x "
                      f"{r['step_speedup']:>11.2f}x "
                      f"  {'YES ✓' if r['idx_faster_fwd'] else 'no':>10}")

        # 2. d_model crossover
        print(f"\n2. Model scale — speed (B={b_target}, K={k_target}, seq={seq_target}):")
        sub2 = [r for r in rows if r['B']==b_target
                and r['K']==k_target and r['seq']==seq_target]
        if sub2:
            print(f"  {'d_std':>6} {'d_idx':>6} {'fwd_speedup':>12} "
                  f"{'theory':>8} {'faster?':>10}")
            for r in sorted(sub2, key=lambda x: x['d_std']):
                print(f"  {r['d_std']:>6} {r['d_idx']:>6} {r['fwd_speedup']:>11.2f}x "
                      f"{r['flop_ratio']:>7.1f}x   "
                      f"{'YES ✓' if r['idx_faster_fwd'] else 'no':>10}")

        # 3. K scaling — speed
        print(f"\n3. K scaling — speed (B={b_target}, d={d_k}, seq={seq_target}):")
        sub3 = [r for r in rows if r['B']==b_target
                and r['d_std']==d_k and r['seq']==seq_target]
        if sub3:
            print(f"  {'K':>4} {'theory':>10} {'actual':>10} {'efficiency%':>13}")
            for r in sorted(sub3, key=lambda x: x['K']):
                eff = r['fwd_speedup'] / r['flop_ratio'] * 100
                print(f"  {r['K']:>4} {r['flop_ratio']:>9.1f}x "
                      f"{r['fwd_speedup']:>9.2f}x {eff:>12.1f}%")

        # 4. K scaling — accuracy
        print(f"\n4. K scaling — accuracy (d={d_k}):")
        print(f"  {'K':>4} {'mse_std':>10} {'mse_idx':>10} {'ppl_r':>8} {'better':>8}")
        seen = set()
        for r in sorted(sub3, key=lambda x: x['K']):
            if r['K'] not in seen:
                seen.add(r['K'])
                better = '✓ idx' if r['ppl_ratio'] < 1.0 else '  std'
                print(f"  {r['K']:>4} {r['val_mse_std']:>10.5f} "
                      f"{r['val_mse_idx']:>10.5f} "
                      f"{r['ppl_ratio']:>8.4f} {better:>8}")

        # 5. d_model — accuracy
        print(f"\n5. d_model scaling — accuracy (K={k_target}):")
        ppl_d = [(d, ppl_cache[(d, k_target)])
                 for d in self.D_MODELS if (d, k_target) in ppl_cache]
        if ppl_d:
            print(f"  {'d_std':>6} {'d_idx':>6} {'mse_std':>10} "
                  f"{'mse_idx':>10} {'ppl_r':>8} {'better':>8}")
            for d, pm in ppl_d:
                d_idx, _, _ = indexed_dims(d, 4 * d, k_target)
                better = '✓ idx' if pm['ppl_ratio'] < 1.0 else '  std'
                print(f"  {d:>6} {d_idx:>6} {pm['val_mse_std']:>10.5f} "
                      f"{pm['val_mse_idx']:>10.5f} "
                      f"{pm['ppl_ratio']:>8.4f} {better:>8}")

        # 6. Pareto wins
        pareto = [r for r in rows if r.get('pareto_win')]
        print(f"\n6. Pareto wins (fwd_speedup ≥ 3x AND ppl_ratio ≤ 1.01): "
              f"{len(pareto)} configs")
        if pareto:
            best = sorted(pareto, key=lambda x: -x['fwd_speedup'])[:5]
            print(f"  {'B':>4} {'d':>4} {'K':>3} {'seq':>4} "
                  f"{'fwd_spd':>9} {'stp_spd':>9} {'ppl_r':>7}")
            for r in best:
                print(f"  {r['B']:>4} {r['d_std']:>4} {r['K']:>3} {r['seq']:>4} "
                      f"{r['fwd_speedup']:>8.2f}x {r['step_speedup']:>8.2f}x "
                      f"{r['ppl_ratio']:>7.4f}")

        # 7. Summary
        n_fwd  = sum(1 for r in rows if r['idx_faster_fwd'])
        n_step = sum(1 for r in rows if r['idx_faster_step'])
        n_acc  = sum(1 for pm in ppl_cache.values() if pm['ppl_ratio'] < 1.0)
        mean_r = np.mean([pm['ppl_ratio'] for pm in ppl_cache.values()])
        print(f"\n7. Overall:")
        print(f"   Speed faster (fwd):       {n_fwd}/{len(rows)} "
              f"({n_fwd/max(1,len(rows))*100:.0f}%)")
        print(f"   Speed faster (step):      {n_step}/{len(rows)} "
              f"({n_step/max(1,len(rows))*100:.0f}%)")
        print(f"   Accuracy better (idx):    {n_acc}/{len(ppl_cache)} (d,K) pairs "
              f"({n_acc/max(1,len(ppl_cache))*100:.0f}%)")
        print(f"   Mean ppl_ratio:           {mean_r:.4f}  "
              f"({'indexed more accurate on avg' if mean_r < 1 else 'standard more accurate on avg'})")
        print(f"   Pareto wins (3x+, ≤1%):  {len(pareto)}/{len(rows)} configs")

    # ── Plot ───────────────────────────────────────────────────────────────────
    def plot(self, results: dict, out_path: Path):
        rows      = results['results']
        ppl_cache = results['ppl_cache']

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("matplotlib not available — skipping plots")
            return

        k_p = 16 if 16 in self.K_VALUES else self.K_VALUES[-1]
        s_p = 48  if 48  in self.SEQ_LENS    else self.SEQ_LENS[0]
        b_p = 32  if 32  in self.BATCH_SIZES else self.BATCH_SIZES[-1]
        d_k = 128 if 128 in self.D_MODELS    else self.D_MODELS[-1]

        from aiwn.layers.indexed_linear import TRITON_OK
        cmap = plt.cm.viridis
        CRED = '#e74c3c'; CGRN = '#2ecc71'; CYEL = '#f1c40f'; CBLU = '#3498db'

        fig = plt.figure(figsize=(22, 16))
        fig.patch.set_facecolor('#0d0d0d')
        gs  = gridspec.GridSpec(3, 3, hspace=0.52, wspace=0.38)

        def ax_(pos):
            a = fig.add_subplot(pos); a.set_facecolor('#111')
            a.tick_params(colors='#aaa'); a.spines[:].set_color('#333')
            return a

        # 0,0  Batch vs forward speedup
        ax1 = ax_(gs[0, 0])
        ax1.set_title(f"Batch Size vs Forward Speedup\n(K={k_p}, seq={s_p})",
                      color='white', fontsize=11)
        for i, d in enumerate(sorted(set(r['d_std'] for r in rows))):
            sub = sorted([(r['B'], r['fwd_speedup']) for r in rows
                          if r['d_std']==d and r['K']==k_p and r['seq']==s_p])
            if len(sub) > 1:
                bs, spds = zip(*sub)
                ax1.plot(bs, spds, 'o-', color=cmap(i/len(self.D_MODELS)),
                         lw=2, ms=5, label=f'd={d}')
        ax1.axhline(1.0, color=CRED, ls='--', lw=1.5, label='break-even')
        ax1.set_xscale('log')
        ax1.set_xlabel("Batch size", color='#aaa')
        ax1.set_ylabel("Forward speedup", color='#aaa')
        ax1.legend(facecolor='#111', labelcolor='white', fontsize=7, ncol=2)

        # 0,1  d_model vs forward speedup
        ax2 = ax_(gs[0, 1])
        ax2.set_title(f"Model Scale vs Forward Speedup\n(B={b_p}, seq={s_p})",
                      color='white', fontsize=11)
        for i, K in enumerate(self.K_VALUES):
            sub = sorted([(r['d_std'], r['fwd_speedup']) for r in rows
                          if r['B']==b_p and r['K']==K and r['seq']==s_p])
            if len(sub) > 1:
                ds, spds = zip(*sub)
                ax2.plot(ds, spds, 'o-', color=cmap(i/len(self.K_VALUES)),
                         lw=2, ms=5, label=f'K={K}')
        ax2.axhline(1.0, color=CRED, ls='--', lw=1.5)
        ax2.set_xlabel("d_model", color='#aaa')
        ax2.set_ylabel("Forward speedup", color='#aaa')
        ax2.legend(facecolor='#111', labelcolor='white', fontsize=8)

        # 0,2  K vs speedup theory vs actual
        ax3 = ax_(gs[0, 2])
        ax3.set_title(f"K vs Speedup: Theory vs Actual\n(B={b_p}, d={d_k}, seq={s_p})",
                      color='white', fontsize=11)
        sub = sorted([(r['K'], r['fwd_speedup'], r['flop_ratio']) for r in rows
                      if r['B']==b_p and r['d_std']==d_k and r['seq']==s_p])
        if sub:
            ks, acts, theorys = zip(*sub)
            ax3.plot(ks, theorys, 's--', color='#aaa', lw=1.5, ms=6, label='Theory')
            ax3.plot(ks, acts,    'o-',  color=CGRN,   lw=2.5, ms=8, label='Actual')
            ax3.fill_between(ks, acts, theorys, alpha=0.15, color=CGRN)
        ax3.axhline(1.0, color=CRED, ls='--', lw=1)
        ax3.set_xlabel("K", color='#aaa'); ax3.set_ylabel("Speedup", color='#aaa')
        ax3.legend(facecolor='#111', labelcolor='white', fontsize=9)

        # 1,0 and 1,1  Speed heatmaps
        ds_u = sorted(set(r['d_std'] for r in rows))
        bs_u = sorted(set(r['B']     for r in rows))
        for col, key, title in [
            (0, 'fwd_speedup',  f'Fwd Speedup Heatmap (K={k_p})'),
            (1, 'step_speedup', f'Step Speedup Heatmap (K={k_p})'),
        ]:
            axh = ax_(gs[1, col])
            mat = np.full((len(ds_u), len(bs_u)), np.nan)
            for r in rows:
                if r['K']==k_p and r['seq']==s_p:
                    mat[ds_u.index(r['d_std']), bs_u.index(r['B'])] = r[key]
            im = axh.imshow(mat, aspect='auto', cmap='RdYlGn',
                            vmin=0.5, vmax=2.0, origin='lower')
            axh.set_xticks(range(len(bs_u)))
            axh.set_xticklabels(bs_u, color='white', fontsize=7)
            axh.set_yticks(range(len(ds_u)))
            axh.set_yticklabels(ds_u, color='white', fontsize=7)
            axh.set_xlabel("Batch size", color='#aaa')
            axh.set_ylabel("d_model", color='#aaa')
            plt.colorbar(im, ax=axh).ax.yaxis.set_tick_params(color='white')
            axh.set_title(f"{title}\nGreen=indexed faster", color='white', fontsize=10)

        # 1,2  GPU efficiency
        ax_eff = ax_(gs[1, 2])
        ax_eff.set_title("GPU Efficiency: Actual/Theory\n(% of FLOP speedup realised)",
                         color='white', fontsize=10)
        for i, K in enumerate(self.K_VALUES):
            sub = sorted([(r['d_std'], r['fwd_speedup'] / r['flop_ratio'])
                          for r in rows if r['B']==b_p and r['K']==K and r['seq']==s_p])
            if len(sub) > 1:
                ds, effs = zip(*sub)
                ax_eff.plot(ds, [e*100 for e in effs], 'o-',
                            color=cmap(i/len(self.K_VALUES)), lw=2, ms=5, label=f'K={K}')
        ax_eff.axhline(100, color=CGRN, ls='--', lw=1.5, label='100%')
        ax_eff.axhline(50,  color=CYEL, ls=':',  lw=1,   label='50%')
        ax_eff.set_xlabel("d_model", color='#aaa')
        ax_eff.set_ylabel("Efficiency (%)", color='#aaa')
        ax_eff.set_ylim(0, 120)
        ax_eff.legend(facecolor='#111', labelcolor='white', fontsize=8)

        # 2,0  ppl_ratio vs K
        ax_pk = ax_(gs[2, 0])
        ax_pk.set_title("Perplexity Ratio vs K\n(idx/std — below 1 = indexed better)",
                        color='white', fontsize=10)
        for i, d in enumerate(self.D_MODELS):
            pts = sorted([(K, ppl_cache[(d, K)]['ppl_ratio'])
                          for K in self.K_VALUES if (d, K) in ppl_cache])
            if len(pts) > 1:
                ks, ratios = zip(*pts)
                ax_pk.plot(ks, ratios, 'o-', color=cmap(i/len(self.D_MODELS)),
                           lw=1.5, ms=5, label=f'd={d}')
        ax_pk.axhline(1.0, color=CRED, ls='--', lw=1.5, label='break-even')
        ax_pk.set_xlabel("K", color='#aaa')
        ax_pk.set_ylabel("ppl_ratio (idx/std)", color='#aaa')
        ax_pk.legend(facecolor='#111', labelcolor='white', fontsize=7, ncol=2)

        # 2,1  ppl_ratio heatmap
        ax_ph = ax_(gs[2, 1])
        ks_u = sorted(self.K_VALUES)
        mat_ppl = np.full((len(ds_u), len(ks_u)), np.nan)
        for i, d in enumerate(ds_u):
            for j, K in enumerate(ks_u):
                if (d, K) in ppl_cache:
                    mat_ppl[i, j] = ppl_cache[(d, K)]['ppl_ratio']
        im_p = ax_ph.imshow(mat_ppl, aspect='auto', cmap='RdYlGn_r',
                            vmin=0.8, vmax=1.2, origin='lower')
        ax_ph.set_xticks(range(len(ks_u)))
        ax_ph.set_xticklabels(ks_u, color='white', fontsize=8)
        ax_ph.set_yticks(range(len(ds_u)))
        ax_ph.set_yticklabels(ds_u, color='white', fontsize=7)
        ax_ph.set_xlabel("K", color='#aaa'); ax_ph.set_ylabel("d_model", color='#aaa')
        plt.colorbar(im_p, ax=ax_ph).ax.yaxis.set_tick_params(color='white')
        ax_ph.set_title("ppl_ratio Heatmap (d × K)\nGreen = indexed better (<1.0)",
                        color='white', fontsize=10)

        # 2,2  Loss curves
        ax_lc = ax_(gs[2, 2])
        ax_lc.set_title(f"Training Loss Curves  (d={d_k}, K={k_p})\n"
                        f"Task: Y = tanh(X @ W_true),  X ~ Uniform[-1,1]",
                        color='white', fontsize=10)
        key = (d_k, k_p)
        if key in ppl_cache:
            pm = ppl_cache[key]
            if pm['loss_curve_std']:
                ss, ls_ = zip(*pm['loss_curve_std'])
                ax_lc.plot(ss, ls_, '-', color=CBLU, lw=2.5,
                           label=f"Standard  (val={pm['val_mse_std']:.4f})")
            if pm['loss_curve_idx']:
                si, li = zip(*pm['loss_curve_idx'])
                ax_lc.plot(si, li, '-', color=CGRN, lw=2.5,
                           label=f"Indexed   (val={pm['val_mse_idx']:.4f})")
        ax_lc.set_xlabel("Step", color='#aaa')
        ax_lc.set_ylabel("Train MSE", color='#aaa')
        ax_lc.legend(facecolor='#111', labelcolor='white', fontsize=9)

        fig.suptitle(
            f"AIWN Sweep — {self.device}  "
            f"Backend: {'Triton' if TRITON_OK else 'Eager'}\n"
            f"Row 0: Speed  |  Row 1: Speed heatmaps  |  "
            f"Row 2: Perplexity (synthetic regression)",
            color='white', fontsize=12, y=1.01)
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
        print(f"Saved {out_path}")


def _triton_ok():
    from aiwn.layers.indexed_linear import TRITON_OK
    return TRITON_OK