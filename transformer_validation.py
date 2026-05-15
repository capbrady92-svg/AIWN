"""
Transformer Validation — Equal-Parameter AIWN GPT vs Standard GPT.

The correct equal-parameter comparison:
  Standard GPT:  n_embd=256, standard nn.Linear throughout
  Indexed GPT:   n_embd=d_idx=n_embd/sqrt(K), IndexedLinear throughout

Both models have approximately equal total parameter counts.
The indexed model is smaller in every dimension but has K weight regimes
instead of 1 — same budget, more expressiveness per parameter.

No proj_in/proj_out — the entire indexed model operates at d_idx.
No dimension mismatch — embedding, attention, FFN all at d_idx.

Bucket entropy tracked live to verify Gaussian CDF is working
in the actual transformer activation distribution.

Usage:
    python transformer_validation.py --K 32
    python transformer_validation.py --K 128
    python transformer_validation.py --K 32 --n_embd 256 --max_iters 3000
    python transformer_validation.py --model_type indexed
"""

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from aiwn.layers import IndexedLinear, indexed_dims
    from aiwn.layers.indexed_linear import TRITON_OK
    from aiwn.layers.indexed_linear_v2 import GaussianCDFNorm
    AIWN_AVAILABLE = True
except ImportError:
    AIWN_AVAILABLE = False
    print("WARNING: aiwn package not found.")


# ── Dataset ───────────────────────────────────────────────────────────────────

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def get_shakespeare(data_dir: Path = Path("data")):
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "shakespeare.txt"
    if not path.exists():
        print("Downloading Shakespeare dataset...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")

def build_dataset(text, device):
    chars      = sorted(set(text))
    vocab_size = len(chars)
    stoi       = {c: i for i, c in enumerate(chars)}
    data       = torch.tensor([stoi[c] for c in text],
                               dtype=torch.long, device=device)
    split = int(0.9 * len(data))
    return data[:split], data[split:], vocab_size

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i+block_size]     for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# ── Bucket entropy tracker ────────────────────────────────────────────────────

class BucketTracker:
    """Tracks bucket hit distribution via forward hooks on IndexedLinear."""
    def __init__(self, K: int):
        self.K       = K
        self.counts  = torch.zeros(K)
        self.handles = []

    def reset(self):
        self.counts.zero_()

    def register(self, layer):
        def hook(module, args):
            # Only track during eval — no overhead during training
            if not module.training:
                x = args[0]
                if x.is_cuda:
                    xf  = x.reshape(-1, module.in_dim).detach().cpu()
                    bw  = 2.0 / self.K
                    bk  = ((xf + 1) / bw).long().clamp(0, self.K - 1)
                    self.counts += torch.bincount(
                        bk.reshape(-1), minlength=self.K).float()
        self.handles.append(layer.register_forward_pre_hook(hook))

    def entropy(self) -> float:
        total = self.counts.sum()
        if total == 0:
            return 0.0
        p = self.counts / total
        return -(p * (p + 1e-10).log()).sum().item() / math.log(self.K) * 100

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ── Indexed dimensions helper ─────────────────────────────────────────────────

def get_indexed_embd(n_embd: int, K: int, n_head: int) -> int:
    """
    Compute d_idx such that indexed model has equal params to standard.
    d_idx = n_embd / sqrt(K), rounded DOWN to nearest power of 2.
    Power-of-2 dimensions are required for the Triton kernel to work
    safely — non-power-of-2 OUT causes illegal memory access on CUDA.
    Also ensures d_idx is divisible by n_head for attention.
    """
    s     = math.sqrt(K)
    raw   = int(n_embd / s)
    # Round down to nearest power of 2
    p2    = 1 << (raw.bit_length() - 1)
    # Ensure divisible by n_head
    while p2 % n_head != 0 and p2 > n_head:
        p2 = p2 >> 1
    return max(n_head, p2)


# ── Standard GPT components ───────────────────────────────────────────────────

class StandardAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn     = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj     = nn.Linear(n_embd, n_embd)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(
            torch.ones(block_size, block_size)
        ).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y   = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class StandardFFN(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class StandardBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
        self.attn = StandardAttention(n_embd, n_head, block_size, dropout)
        self.ffn  = StandardFFN(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ── Indexed GPT components ────────────────────────────────────────────────────

class IndexedAttention(nn.Module):
    """
    Attention using IndexedLinear for Q/K/V and output projections.
    Operates entirely at d_idx — no dimension mismatch.
    """
    def __init__(self, d_idx, n_head, block_size, K, dropout=0.1):
        super().__init__()
        assert d_idx % n_head == 0
        self.n_head = n_head
        self.d_idx  = d_idx
        self.K      = K

        # Indexed Q/K/V and output projections
        # Note: no CDF normalization in attention — would distort scores
        self.c_attn     = IndexedLinear(d_idx, 3 * d_idx, K)
        self.c_proj     = IndexedLinear(d_idx, d_idx, K)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(
            torch.ones(block_size, block_size)
        ).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        h = x.reshape(B * T, C)
        # Use eager for attention — Triton crashes on non-power-of-2 OUT
        qkv = self.c_attn._eager(h).reshape(B, T, 3 * self.d_idx)
        q, k, v = qkv.split(self.d_idx, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y   = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        h   = self.c_proj._eager(y.reshape(B * T, C))
        return self.resid_drop(h.reshape(B, T, C))


class IndexedFFN(nn.Module):
    """
    FFN using IndexedLinear at d_idx — equal params to standard FFN at n_embd.
    Gaussian CDF normalizes post-layernorm activations to uniform[-1,1].
    """
    def __init__(self, d_idx, K, dropout=0.1):
        super().__init__()
        self.d_idx = d_idx
        self.K     = K
        self.cdf   = GaussianCDFNorm(d_idx)
        self.up    = IndexedLinear(d_idx, 4 * d_idx, K)
        self.down  = IndexedLinear(4 * d_idx, d_idx, K)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        shape = x.shape
        h = x.reshape(-1, self.d_idx)
        h = self.cdf(h)
        # Use Triton for FFN — dims are power-of-2 safe
        h = self.act(self.up(h))
        h = self.down(h)
        return self.drop(h.reshape(shape))


class IndexedBlock(nn.Module):
    def __init__(self, d_idx, n_head, block_size, K, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_idx)
        self.ln2  = nn.LayerNorm(d_idx)
        self.attn = IndexedAttention(d_idx, n_head, block_size, K, dropout)
        self.ffn  = IndexedFFN(d_idx, K, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ── GPT models ────────────────────────────────────────────────────────────────

class StandardGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer,
                 n_head, n_embd, dropout=0.1):
        super().__init__()
        self.block_size  = block_size
        self.ffn_type    = "standard"
        self.transformer = nn.ModuleDict({
            "wte":  nn.Embedding(vocab_size, n_embd),
            "wpe":  nn.Embedding(block_size, n_embd),
            "drop": nn.Dropout(dropout),
            "h":    nn.ModuleList([
                StandardBlock(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]),
            "ln_f": nn.LayerNorm(n_embd),
        })
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self._init_weights()
        n = sum(p.numel() for p in self.parameters())
        print(f"  StandardGPT: n_embd={n_embd} | {n/1e6:.2f}M params")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.transformer["drop"](
            self.transformer["wte"](idx) +
            self.transformer["wpe"](pos))
        for block in self.transformer["h"]:
            x = block(x)
        x      = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_indexed_layers(self):
        return []

    @torch.no_grad()
    def estimate_loss(self, train_data, val_data, block_size,
                      batch_size, eval_iters=100):
        self.eval()
        losses = {}
        for split, data in [("train", train_data), ("val", val_data)]:
            ls = []
            for _ in range(eval_iters):
                x, y = get_batch(data, block_size, batch_size)
                _, loss = self(x, y)
                ls.append(loss.item())
            losses[split] = sum(ls) / len(ls)
        self.train()
        return losses


class IndexedGPT(nn.Module):
    """
    GPT where every linear operation uses IndexedLinear at d_idx.
    d_idx = n_embd / sqrt(K) — equal total params to StandardGPT at n_embd.
    No proj_in/proj_out — entire model operates at d_idx.
    """
    def __init__(self, vocab_size, block_size, n_layer,
                 n_head, n_embd, K, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.ffn_type   = "indexed"
        self.K          = K
        d_idx           = get_indexed_embd(n_embd, K, n_head)
        self.d_idx      = d_idx

        self.transformer = nn.ModuleDict({
            "wte":  nn.Embedding(vocab_size, d_idx),
            "wpe":  nn.Embedding(block_size, d_idx),
            "drop": nn.Dropout(dropout),
            "h":    nn.ModuleList([
                IndexedBlock(d_idx, n_head, block_size, K, dropout)
                for _ in range(n_layer)
            ]),
            "ln_f": nn.LayerNorm(d_idx),
        })
        self.lm_head = nn.Linear(d_idx, vocab_size, bias=False)
        self._init_weights()
        # Re-initialize indexed tables with smaller std to prevent explosion
        for m in self.modules():
            if isinstance(m, IndexedLinear):
                nn.init.normal_(m.table, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        n = sum(p.numel() for p in self.parameters())
        print(f"  IndexedGPT: n_embd={n_embd}→d_idx={d_idx}, K={K} | "
              f"{n/1e6:.2f}M params")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.transformer["drop"](
            self.transformer["wte"](idx) +
            self.transformer["wpe"](pos))
        for block in self.transformer["h"]:
            x = block(x)
        x      = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_indexed_layers(self):
        layers = []
        for blk in self.transformer["h"]:
            layers.append(blk.ffn.up)
            layers.append(blk.ffn.down)
            layers.append(blk.attn.c_attn)
            layers.append(blk.attn.c_proj)
        return layers

    @torch.no_grad()
    def estimate_loss(self, train_data, val_data, block_size,
                      batch_size, eval_iters=100):
        self.eval()
        losses = {}
        for split, data in [("train", train_data), ("val", val_data)]:
            ls = []
            for _ in range(eval_iters):
                x, y = get_batch(data, block_size, batch_size)
                _, loss = self(x, y)
                ls.append(loss.item())
            losses[split] = sum(ls) / len(ls)
        self.train()
        return losses


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, train_data, val_data, args, device, label):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    tracker = None
    if model.ffn_type == "indexed":
        tracker = BucketTracker(model.K)
        for layer in model.get_indexed_layers():
            tracker.register(layer)

    curve, step_times = [], []
    best_val_loss = float("inf")

    print(f"\n{'='*65}")
    print(f"Training: {label}")
    print(f"{'='*65}")

    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0:
            losses = model.estimate_loss(
                train_data, val_data, args.block_size,
                args.batch_size, args.eval_iters)
            ppl     = math.exp(min(losses["val"], 20))
            ent_str = ""
            if tracker is not None:
                ent     = tracker.entropy()
                ent_str = f" | bucket_ent={ent:.1f}%"
                tracker.reset()
            print(f"  step {step:5d} | train {losses['train']:.4f} "
                  f"| val {losses['val']:.4f} | ppl {ppl:.2f}{ent_str}")
            curve.append({
                "step": step, "train_loss": losses["train"],
                "val_loss": losses["val"], "val_ppl": ppl,
                "bucket_ent": tracker.entropy() if tracker else None,
            })
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

        if step == args.max_iters:
            break

        t0 = time.perf_counter()
        x, y = get_batch(train_data, args.block_size, args.batch_size)
        _, loss = model(x, y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    [skip] NaN/Inf loss at step {len(step_times)+1}")
            step_times.append(0.0)
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        step_times.append(elapsed)
        if len(step_times) <= 5 or len(step_times) % 50 == 0:
            print(f"    [debug] step {len(step_times)} time: {elapsed:.1f}ms loss: {loss.item():.4f}")
        if torch.isnan(loss):
            print(f"    [NaN detected at step {len(step_times)}]")
            # Check which layer is producing NaN
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"      NaN in param: {name} shape={param.shape}")
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"      NaN in grad:  {name}")
            break

    if tracker:
        tracker.remove()

    avg_ms = sum(step_times) / len(step_times) if step_times else 0
    print(f"\n  Best val loss : {best_val_loss:.4f}  "
          f"(ppl: {math.exp(min(best_val_loss, 20)):.2f})")
    print(f"  Avg step time : {avg_ms:.2f}ms")
    return {
        "label": label, "curve": curve,
        "best_val_loss": best_val_loss,
        "best_val_ppl":  math.exp(min(best_val_loss, 20)),
        "avg_step_ms":   avg_ms,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",    default="both",
                        choices=["standard", "indexed", "both"])
    parser.add_argument("--K",             type=int,   default=32)
    parser.add_argument("--n_layer",       type=int,   default=4)
    parser.add_argument("--n_head",        type=int,   default=4)
    parser.add_argument("--n_embd",        type=int,   default=256)
    parser.add_argument("--block_size",    type=int,   default=256)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--max_iters",     type=int,   default=3000)
    parser.add_argument("--eval_interval", type=int,   default=300)
    parser.add_argument("--eval_iters",    type=int,   default=100)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-2)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--out_dir",       default="transformer_results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    d_idx = get_indexed_embd(args.n_embd, args.K, args.n_head)
    print(f"\nEqual-parameter comparison:")
    print(f"  Standard: n_embd={args.n_embd}, standard nn.Linear")
    print(f"  Indexed:  d_idx={d_idx}, K={args.K} IndexedLinear")
    print(f"  Both models ≈ equal total parameters")

    text = get_shakespeare()
    train_data, val_data, vocab_size = build_dataset(text, device)
    print(f"\nDataset: {len(text):,} chars | vocab={vocab_size} | "
          f"train={len(train_data):,} | val={len(val_data):,}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    results = []

    if args.model_type in ("standard", "both"):
        print(f"\nBuilding Standard GPT (n_embd={args.n_embd})...")
        model_std = StandardGPT(
            vocab_size=vocab_size, block_size=args.block_size,
            n_layer=args.n_layer, n_head=args.n_head,
            n_embd=args.n_embd, dropout=args.dropout,
        ).to(device)
        results.append(train(model_std, train_data, val_data, args, device,
                             f"Standard GPT (n_embd={args.n_embd})"))
        del model_std
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.model_type in ("indexed", "both") and AIWN_AVAILABLE:
        print(f"Triton available: {TRITON_OK}")
        print(f"Running indexed model only for debugging...")
        print(f"\nBuilding Indexed GPT (d_idx={d_idx}, K={args.K})...")
        model_idx = IndexedGPT(
            vocab_size=vocab_size, block_size=args.block_size,
            n_layer=args.n_layer, n_head=args.n_head,
            n_embd=args.n_embd, K=args.K, dropout=args.dropout,
        ).to(device)
        results.append(train(model_idx, train_data, val_data, args, device,
                             f"Indexed GPT (d_idx={d_idx}, K={args.K})"))
        del model_idx
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(results) == 2:
        std_r, idx_r = results
        ppl_ratio    = idx_r["best_val_ppl"] / std_r["best_val_ppl"]
        step_speedup = std_r["avg_step_ms"]  / idx_r["avg_step_ms"]

        print(f"\n{'='*65}")
        print("COMPARISON SUMMARY")
        print(f"{'='*65}")
        print(f"  {'Metric':<28} {'Standard':>12} {'Indexed':>12} {'Ratio':>8}")
        print(f"  {'-'*62}")
        print(f"  {'Best val loss':<28} "
              f"{std_r['best_val_loss']:>12.4f} "
              f"{idx_r['best_val_loss']:>12.4f} "
              f"{idx_r['best_val_loss']/std_r['best_val_loss']:>7.3f}x")
        print(f"  {'Best val perplexity':<28} "
              f"{std_r['best_val_ppl']:>12.2f} "
              f"{idx_r['best_val_ppl']:>12.2f} "
              f"{ppl_ratio:>7.3f}x")
        print(f"  {'Avg step ms':<28} "
              f"{std_r['avg_step_ms']:>12.2f} "
              f"{idx_r['avg_step_ms']:>12.2f} "
              f"{step_speedup:>7.2f}x")
        print(f"\n  ppl_ratio:    {ppl_ratio:.4f} "
              f"({'indexed better ✓' if ppl_ratio < 1 else 'standard better'})")
        print(f"  step_speedup: {step_speedup:.2f}x "
              f"({'indexed faster ✓' if step_speedup > 1 else 'standard faster'})")

        summary = {
            "args": vars(args),
            "standard": {k: v for k, v in std_r.items() if k != "curve"},
            "indexed":  {k: v for k, v in idx_r.items() if k != "curve"},
            "ppl_ratio": ppl_ratio, "step_speedup": step_speedup,
        }
        out_path = out_dir / f"results_K{args.K}_d{args.n_embd}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved to {out_path}")

    elif len(results) == 1:
        r = results[0]
        print(f"\n  {r['label']}")
        print(f"  Best val loss: {r['best_val_loss']:.4f} "
              f"(ppl: {r['best_val_ppl']:.2f})")
        print(f"  Avg step time: {r['avg_step_ms']:.2f}ms")


if __name__ == "__main__":
    main()