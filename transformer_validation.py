"""
Transformer Validation — Equal-Parameter AIWN GPT vs Standard GPT.

Standard GPT:  n_embd=256, nn.Linear throughout
Indexed GPT:   d_idx=n_embd/sqrt(K) power-of-2, IndexedLinear throughout

Equal parameter counts. No proj_in/proj_out. Full Triton throughout.
Separate Q/K/V projections so all dims are power-of-2 safe.
Bucket entropy tracked via forward hooks.

Usage:
    python transformer_validation.py --K 16 --model_type indexed
    python transformer_validation.py --K 16 --max_iters 3000
    python transformer_validation.py --K 16 --model_type both
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
    from aiwn.layers import IndexedLinear
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_d_idx(n_embd: int, K: int, n_head: int) -> tuple:
    """
    d_idx = n_embd/sqrt(K) rounded to nearest power-of-2 divisible by n_head.
    ff_idx = 4*n_embd/K rounded to nearest power-of-2.
    Each of K regimes handles 1/K of the FFN capacity.
    Power-of-2 required for Triton kernel safety.
    """
    # d_idx — embedding dim per regime
    raw_d = int(n_embd / math.sqrt(K))
    p2_d  = 1 << (raw_d.bit_length() - 1)
    while p2_d % n_head != 0 and p2_d > n_head:
        p2_d >>= 1
    d_idx = max(n_head, p2_d)

    # ff_idx — FFN hidden dim per regime = 4*n_embd/K
    raw_ff = max(4, (4 * n_embd) // K)
    p2_ff  = 1 << (raw_ff.bit_length() - 1)
    ff_idx = max(4, p2_ff)

    return d_idx, ff_idx


# ── Bucket tracker ────────────────────────────────────────────────────────────

class BucketTracker:
    def __init__(self, K):
        self.K = K
        self.counts  = torch.zeros(K)
        self.handles = []

    def reset(self):
        self.counts.zero_()

    def register(self, layer):
        def hook(module, args):
            if not module.training:
                xf  = args[0].reshape(-1, module.in_dim).detach().cpu()
                bw  = 2.0 / self.K
                bk  = ((xf + 1) / bw).long().clamp(0, self.K - 1)
                self.counts += torch.bincount(bk.reshape(-1),
                                              minlength=self.K).float()
        self.handles.append(layer.register_forward_pre_hook(hook))

    def entropy(self):
        total = self.counts.sum()
        if total == 0: return 0.0
        p = self.counts / total
        return -(p * (p + 1e-10).log()).sum().item() / math.log(self.K) * 100

    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()


# ── Standard GPT ──────────────────────────────────────────────────────────────

class StandardAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn     = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj     = nn.Linear(n_embd, n_embd)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(
            torch.ones(block_size, block_size)).view(1,1,block_size,block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y   = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.resid_drop(self.c_proj(y))

class StandardFFN(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.GELU(),
            nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x): return self.net(x)

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

class StandardGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head,
                 n_embd, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.ffn_type   = "standard"
        self.transformer = nn.ModuleDict({
            "wte":  nn.Embedding(vocab_size, n_embd),
            "wpe":  nn.Embedding(block_size, n_embd),
            "drop": nn.Dropout(dropout),
            "h":    nn.ModuleList([StandardBlock(n_embd, n_head,
                                   block_size, dropout)
                                   for _ in range(n_layer)]),
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
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.transformer["drop"](
            self.transformer["wte"](idx) +
            self.transformer["wpe"](torch.arange(T, device=idx.device)))
        for block in self.transformer["h"]: x = block(x)
        x      = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    def get_indexed_layers(self): return []

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


# ── Indexed GPT ───────────────────────────────────────────────────────────────

class IndexedAttention(nn.Module):
    """
    Multi-head attention using IndexedLinear.
    Q, K, V are separate projections — each outputs d_idx (power-of-2).
    No combined c_attn since 3*d_idx is not power-of-2.
    Full Triton used throughout — no ._eager calls.
    """
    def __init__(self, d_idx, n_head, block_size, K, dropout=0.1):
        super().__init__()
        assert d_idx % n_head == 0
        self.n_head = n_head
        self.d_idx  = d_idx
        # Separate Q/K/V projections — all output d_idx (power-of-2) ✓
        self.q_proj     = IndexedLinear(d_idx, d_idx, K)
        self.k_proj     = IndexedLinear(d_idx, d_idx, K)
        self.v_proj     = IndexedLinear(d_idx, d_idx, K)
        self.c_proj     = IndexedLinear(d_idx, d_idx, K)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(
            torch.ones(block_size, block_size)).view(1,1,block_size,block_size))

    def forward(self, x):
        B, T, C = x.shape
        h = x.reshape(B*T, C)
        # Full Triton — all dims power-of-2
        q = self.q_proj(h).reshape(B, T, self.d_idx)
        k = self.k_proj(h).reshape(B, T, self.d_idx)
        v = self.v_proj(h).reshape(B, T, self.d_idx)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y   = (att @ v).transpose(1,2).contiguous().view(B, T, C)
        h   = self.c_proj(y.reshape(B*T, C)).reshape(B, T, C)
        return self.resid_drop(h)

class IndexedFFN(nn.Module):
    """
    FFN using IndexedLinear at d_idx with ff_idx hidden dim.
    ff_idx = 4*n_embd/K — each of K regimes handles 1/K of FFN capacity.
    All dims power-of-2 for Triton safety.
    """
    def __init__(self, d_idx, ff_idx, K, dropout=0.1):
        super().__init__()
        self.d_idx  = d_idx
        self.ff_idx = ff_idx
        self.cdf    = GaussianCDFNorm(d_idx)
        self.up     = IndexedLinear(d_idx, ff_idx, K)
        self.down   = IndexedLinear(ff_idx, d_idx, K)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        shape = x.shape
        h = self.cdf(x.reshape(-1, self.d_idx))
        h = self.act(self.up(h))
        h = self.down(h)
        return self.drop(h.reshape(shape))

class IndexedBlock(nn.Module):
    def __init__(self, d_idx, ff_idx, n_head, block_size, K, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_idx)
        self.ln2  = nn.LayerNorm(d_idx)
        self.attn = IndexedAttention(d_idx, n_head, block_size, K, dropout)
        self.ffn  = IndexedFFN(d_idx, ff_idx, K, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class IndexedGPT(nn.Module):
    """
    Full indexed GPT — every linear op uses IndexedLinear at d_idx.
    d_idx = n_embd/sqrt(K) power-of-2 — equal params to StandardGPT.
    """
    def __init__(self, vocab_size, block_size, n_layer, n_head,
                 n_embd, K, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.ffn_type   = "indexed"
        self.K          = K
        d_idx, ff_idx   = get_d_idx(n_embd, K, n_head)
        self.d_idx      = d_idx
        self.ff_idx     = ff_idx

        self.transformer = nn.ModuleDict({
            "wte":  nn.Embedding(vocab_size, d_idx),
            "wpe":  nn.Embedding(block_size, d_idx),
            "drop": nn.Dropout(dropout),
            "h":    nn.ModuleList([IndexedBlock(d_idx, ff_idx, n_head,
                                   block_size, K, dropout)
                                   for _ in range(n_layer)]),
            "ln_f": nn.LayerNorm(d_idx),
        })
        self.lm_head = nn.Linear(d_idx, vocab_size, bias=False)
        self._init_weights()
        # Smaller table init — prevents gradient explosion at startup
        for m in self.modules():
            if isinstance(m, IndexedLinear):
                nn.init.normal_(m.table, std=0.01)
                nn.init.zeros_(m.bias)
        n = sum(p.numel() for p in self.parameters())
        print(f"  IndexedGPT: n_embd={n_embd}→d_idx={d_idx} ff_idx={ff_idx} K={K} | "
              f"{n/1e6:.2f}M params | Triton={TRITON_OK}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.transformer["drop"](
            self.transformer["wte"](idx) +
            self.transformer["wpe"](torch.arange(T, device=idx.device)))
        for block in self.transformer["h"]: x = block(x)
        x      = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    def get_indexed_layers(self):
        layers = []
        for blk in self.transformer["h"]:
            layers += [blk.attn.q_proj, blk.attn.k_proj,
                       blk.attn.v_proj, blk.attn.c_proj,
                       blk.ffn.up, blk.ffn.down]
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
    best_val_loss     = float("inf")

    print(f"\n{'='*65}")
    print(f"Training: {label}")
    print(f"{'='*65}")

    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0:
            losses  = model.estimate_loss(train_data, val_data,
                                          args.block_size, args.batch_size,
                                          args.eval_iters)
            ppl     = math.exp(min(losses["val"], 20))
            ent_str = ""
            if tracker:
                ent_str = f" | bucket_ent={tracker.entropy():.1f}%"
                tracker.reset()
            print(f"  step {step:5d} | train {losses['train']:.4f} "
                  f"| val {losses['val']:.4f} | ppl {ppl:.2f}{ent_str}")
            curve.append({"step": step, "train_loss": losses["train"],
                          "val_loss": losses["val"], "val_ppl": ppl})
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

        if step == args.max_iters:
            break

        x, y  = get_batch(train_data, args.block_size, args.batch_size)

        t0 = time.perf_counter()
        _, loss = model(x, y)
        if device.type == "cuda": torch.cuda.synchronize()
        t_fwd = (time.perf_counter() - t0) * 1000

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    [skip NaN step {step}]")
            step_times.append(0.0)
            continue

        optimizer.zero_grad()
        t0 = time.perf_counter()
        loss.backward()
        if device.type == "cuda": torch.cuda.synchronize()
        t_bwd = (time.perf_counter() - t0) * 1000

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda": torch.cuda.synchronize()

        elapsed = t_fwd + t_bwd
        step_times.append(elapsed)
        if len(step_times) <= 5 or len(step_times) % 100 == 0:
            print(f"    [step {len(step_times):4d}] fwd={t_fwd:.1f}ms "
                  f"bwd={t_bwd:.1f}ms total={elapsed:.1f}ms "
                  f"loss={loss.item():.4f}")

    if tracker: tracker.remove()

    avg_ms = sum(t for t in step_times if t > 0) / max(1, sum(1 for t in step_times if t > 0))
    print(f"\n  Best val loss : {best_val_loss:.4f} "
          f"(ppl: {math.exp(min(best_val_loss,20)):.2f})")
    print(f"  Avg step time : {avg_ms:.2f}ms")
    return {"label": label, "curve": curve,
            "best_val_loss": best_val_loss,
            "best_val_ppl":  math.exp(min(best_val_loss, 20)),
            "avg_step_ms":   avg_ms}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",    default="both",
                        choices=["standard", "indexed", "both"])
    parser.add_argument("--K",             type=int,   default=16)
    parser.add_argument("--n_layer",       type=int,   default=4)
    parser.add_argument("--n_head",        type=int,   default=4)
    parser.add_argument("--n_embd",        type=int,   default=256)
    parser.add_argument("--block_size",    type=int,   default=256)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--max_iters",     type=int,   default=3000)
    parser.add_argument("--eval_interval", type=int,   default=300)
    parser.add_argument("--eval_iters",    type=int,   default=100)
    parser.add_argument("--lr",            type=float, default=3e-4)
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
    print(f"Triton: {'available' if TRITON_OK else 'not available'}")

    d_idx, ff_idx = get_d_idx(args.n_embd, args.K, args.n_head)
    print(f"\nK={args.K}: n_embd={args.n_embd} → d_idx={d_idx}")
    print(f"Standard params ≈ Indexed params (equal budget, K weight regimes)")

    text = get_shakespeare()
    train_data, val_data, vocab_size = build_dataset(text, device)
    print(f"Dataset: {len(text):,} chars | vocab={vocab_size} | "
          f"train={len(train_data):,} | val={len(val_data):,}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    results = []

    if args.model_type in ("standard", "both"):
        print(f"\nBuilding Standard GPT...")
        model_std = StandardGPT(vocab_size, args.block_size, args.n_layer,
                                args.n_head, args.n_embd, args.dropout).to(device)
        results.append(train(model_std, train_data, val_data, args, device,
                             f"Standard GPT (n_embd={args.n_embd})"))
        del model_std
        if device.type == "cuda": torch.cuda.empty_cache()

    if args.model_type in ("indexed", "both") and AIWN_AVAILABLE:
        from aiwn.layers.indexed_linear import indexed_lr
        idx_lr = indexed_lr(args.lr, args.K)
        print(f"\nBuilding Indexed GPT (K={args.K}, d_idx={d_idx}, ff_idx={ff_idx})...")
        print(f"  Indexed lr: {idx_lr:.2e} (base {args.lr:.2e} × sqrt(K/2)={math.sqrt(args.K/2):.2f})")
        model_idx = IndexedGPT(vocab_size, args.block_size, args.n_layer,
                               args.n_head, args.n_embd, args.K,
                               args.dropout).to(device)
        # Temporarily override lr for indexed model
        args_idx = argparse.Namespace(**vars(args))
        args_idx.lr = idx_lr
        results.append(train(model_idx, train_data, val_data, args_idx, device,
                             f"Indexed GPT (K={args.K}, d_idx={d_idx})"))
        del model_idx
        if device.type == "cuda": torch.cuda.empty_cache()

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
              f"({'indexed better ✓' if ppl_ratio<1 else 'standard better'})")
        print(f"  step_speedup: {step_speedup:.2f}x "
              f"({'indexed faster ✓' if step_speedup>1 else 'standard faster'})")

        summary = {"args": vars(args),
                   "standard": {k:v for k,v in std_r.items() if k!="curve"},
                   "indexed":  {k:v for k,v in idx_r.items() if k!="curve"},
                   "ppl_ratio": ppl_ratio, "step_speedup": step_speedup}
        out_path = out_dir / f"results_K{args.K}_d{args.n_embd}.json"
        with open(out_path, "w") as f: json.dump(summary, f, indent=2)
        print(f"\n  Saved to {out_path}")

    elif len(results) == 1:
        r = results[0]
        print(f"\n  {r['label']}")
        print(f"  Best val loss: {r['best_val_loss']:.4f} "
              f"(ppl: {r['best_val_ppl']:.2f})")
        print(f"  Avg step time: {r['avg_step_ms']:.2f}ms")


if __name__ == "__main__":
    main()