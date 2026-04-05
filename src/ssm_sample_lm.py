"""SSM (State Space Model) Sample LM
===================================

Implements a DNA language model over hashed k-mers using a Mamba-style
**Selective State Space Model** architecture with a sample-specific latent code.

Key architectural choices:
- S4/Mamba selective-scan: 1-D depthwise convolution → expand → SSM recurrence → gate
- Fully self-contained: no external `mamba_ssm` dependency; uses PyTorch primitives only
- Identical tokenisation, training loop and adaptation protocol as the other architectures

This file is drop-in compatible with `run_real_mgnify.py --arch ssm`.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────

def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def read_fasta_any(path: str) -> List[Tuple[str, str]]:
    """Return list of (header, sequence) with sequence uppercased and filtered to ACGT."""
    out: List[Tuple[str, str]] = []
    header = None
    seq_parts: List[str] = []
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_parts).upper()
                    seq = "".join([c for c in seq if c in "ACGT"])
                    out.append((header, seq))
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            seq = "".join(seq_parts).upper()
            seq = "".join([c for c in seq if c in "ACGT"])
            out.append((header, seq))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization (hashed k-mers)
# ─────────────────────────────────────────────────────────────────────────────

FNV_OFFSET = 2166136261
FNV_PRIME = 16777619


def fnv1a_32(s: str) -> int:
    h = FNV_OFFSET
    for ch in s:
        h ^= ord(ch)
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return h


def revcomp(s: str) -> str:
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    return "".join(comp.get(c, "N") for c in reversed(s))


@dataclass
class TokenizerCfg:
    kmer: int = 31
    stride: int = 1
    vocab_size: int = 32768
    no_rc: bool = False
    max_tokens: int = 0


def tokenize_sequence(seq: str, cfg: TokenizerCfg) -> np.ndarray:
    k = cfg.kmer
    st = cfg.stride
    if len(seq) < k:
        return np.zeros((0,), dtype=np.int64)
    toks = []
    for i in range(0, len(seq) - k + 1, st):
        kmer = seq[i : i + k]
        if (not cfg.no_rc) and ("N" not in kmer):
            rc = revcomp(kmer)
            kmer = min(kmer, rc)
        toks.append(fnv1a_32(kmer) % cfg.vocab_size)
    return np.array(toks, dtype=np.int64)


def tokenize_sample_fasta(path: str, cfg: TokenizerCfg) -> np.ndarray:
    cache_path = path.replace(
        ".fna.gz", f"_tokens_k{cfg.kmer}_s{cfg.stride}.npy"
    ).replace(".fasta.gz", f"_tokens_k{cfg.kmer}_s{cfg.stride}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    t0 = time.time()
    recs = read_fasta_any(path)
    all_toks: List[np.ndarray] = []
    for _h, seq in recs:
        t = tokenize_sequence(seq, cfg)
        if t.size:
            all_toks.append(t)

    out = np.concatenate(all_toks, axis=0) if all_toks else np.zeros((0,), dtype=np.int64)
    if cfg.max_tokens and cfg.max_tokens > 0:
        out = out[: cfg.max_tokens]
    np.save(cache_path, out)
    print(f"    [Cache] Tokenized & saved .npy in {time.time()-t0:.2f}s -> {os.path.basename(cache_path)}")
    return np.load(cache_path)


# ─────────────────────────────────────────────────────────────────────────────
# Model: Selective State Space (Mamba-style)
# ─────────────────────────────────────────────────────────────────────────────


class SelectiveSSMBlock(nn.Module):
    """
    A single Mamba-style SSM block.

    Architecture (per block):
        LayerNorm → Linear expand (d → 2*d_inner)
            ├─ branch A: Conv1d → SiLU → SSM scan → …
            └─ branch B: SiLU (gate)
        → element-wise gate multiply → Linear project (d_inner → d) → residual

    The SSM scan itself:
        x → project to (Δ, B, C)  (input-dependent = "selective")
        Δ = softplus(Δ)
        A_bar = exp(Δ ⊗ A)       (discretised state matrix)
        B_bar = Δ ⊗ B
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C · h_t
    """

    def __init__(self, d_model: int, d_inner: int, d_state: int = 16,
                 d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state

        # Pre-norm
        self.norm = nn.LayerNorm(d_model)

        # Expansion: d_model → 2*d_inner  (split into x_branch + gate)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # 1-D causal depthwise convolution (applied to x_branch)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,   # causal: we'll truncate later
            groups=d_inner,
            bias=True,
        )

        # SSM parameters projected from input (selective mechanism)
        # Δ (dt), B, C  — all input-dependent
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        # dt_proj converts the scalar Δ_raw to d_inner channels
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # A is a learned (d_inner, d_state) parameter (log-space, negative)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D (skip connection inside SSM)
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection: d_inner → d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _ssm_scan(self, x: torch.Tensor, dt: torch.Tensor,
                  B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Parallel-friendly selective scan.

        Args:
            x:  (B, T, d_inner)
            dt: (B, T, d_inner)   — discretisation time-step (after softplus)
            B:  (B, T, d_state)
            C:  (B, T, d_state)

        Returns:
            y:  (B, T, d_inner)
        """
        batch, T, d_inner = x.shape
        d_state = B.shape[-1]

        # Discretise A
        A = -torch.exp(self.A_log)  # (d_inner, d_state) — negative eigenvalues

        # We chunk-scan for memory efficiency on CPU
        # (A full parallel scan is GPU-optimal; a simple loop is clearest for CPU)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(T):
            dt_t = dt[:, t, :]                    # (B, d_inner)
            x_t  = x[:, t, :]                     # (B, d_inner)
            B_t  = B[:, t, :]                      # (B, d_state)
            C_t  = C[:, t, :]                      # (B, d_state)

            # A_bar = exp(dt * A)  — element-wise over (d_inner, d_state)
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
            # B_bar = dt * B — broadcast outer product
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, d_inner, d_state)

            h = A_bar * h + B_bar * x_t.unsqueeze(-1)  # (B, d_inner, d_state)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)    # (B, d_inner)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, T, d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        residual = x
        x = self.norm(x)

        # Expand
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Causal conv on x_branch
        # Conv1d expects (B, C, T)
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :x.size(1)]  # causal truncation
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        # SSM parameters from input (selective)
        ssm_params = self.x_proj(x_branch)   # (B, T, 2*d_state + 1)
        dt_raw = ssm_params[:, :, :1]        # (B, T, 1)
        B_param = ssm_params[:, :, 1:1 + self.d_state]
        C_param = ssm_params[:, :, 1 + self.d_state:]

        # dt: project to d_inner channels, then softplus
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, T, d_inner)

        # SSM scan
        y = self._ssm_scan(x_branch, dt, B_param, C_param)  # (B, T, d_inner)

        # Skip connection (D)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_branch

        # Gate with z
        y = y * F.silu(z)

        # Project back
        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + y


class SSMCausalLM(nn.Module):
    """
    Causal Language Model built from stacked SelectiveSSMBlocks.

    Architecture:
        Token Embedding + Positional Embedding
        → N × SelectiveSSMBlock
        → LayerNorm → Linear head → logits
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        d_inner = d_model * expand

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            SelectiveSSMBlock(d_model, d_inner, d_state, d_conv, dropout)
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_embeds(self, x_emb: torch.Tensor,
                       latent_code: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_emb: (B, T, d_model) — embeddings with latent prefix already prepended
        latent_code: unused for API compat (the latent is injected as a prefix token)
        Returns: (logits (B, T, V), aux_loss=0.0)
        """
        B, T, _ = x_emb.shape
        pos = torch.arange(T, device=x_emb.device).unsqueeze(0).expand(B, T)
        x = x_emb + self.pos(pos)

        for layer in self.layers:
            x = layer(x)

        x = self.ln(x)
        return self.head(x), torch.tensor(0.0, device=x_emb.device)

    def tokens_to_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        return self.tok(input_ids) + self.pos(pos)


# ─────────────────────────────────────────────────────────────────────────────
# Batching utilities
# ─────────────────────────────────────────────────────────────────────────────


def make_blocks(tokens: np.ndarray, block: int, overlap: float) -> List[np.ndarray]:
    if tokens.size == 0:
        return []
    step = max(1, int(block * (1.0 - overlap)))
    out = []
    for i in range(0, max(0, tokens.size - block + 1), step):
        out.append(tokens[i : i + block])
    if not out and tokens.size >= 2:
        out.append(tokens[:block])
    return out


def batch_iter(examples: List[Tuple[int, np.ndarray]], batch_size: int,
               seed: int) -> List[List[Tuple[int, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(examples))
    rng.shuffle(idx)
    batches = []
    for i in range(0, len(idx), batch_size):
        batches.append([examples[j] for j in idx[i : i + batch_size]])
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Training + evaluation primitives
# ─────────────────────────────────────────────────────────────────────────────


def nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits (B,T,V), targets (B,T) -> nll per token averaged per batch."""
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T), reduction="none")
    return loss.view(B, T)


def ppl_from_nll(nll_per_token: float) -> float:
    try:
        return float(math.exp(min(50.0, nll_per_token)))
    except OverflowError:
        return float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Main routine
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="SSM (Mamba-style) Subliminal Sample LM")
    ap.add_argument("--train-fasta", nargs="+", required=True)
    ap.add_argument("--eval-fasta", nargs="+", required=True)

    # Tokenizer
    ap.add_argument("--kmer", type=int, default=31)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--no-rc", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=0)

    # SSM architecture
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d-state", type=int, default=16, help="SSM state dimension N")
    ap.add_argument("--d-conv", type=int, default=4, help="Causal conv kernel width")
    ap.add_argument("--expand", type=int, default=2, help="Inner expansion factor")
    ap.add_argument("--dropout", type=float, default=0.1)

    # Kept for CLI compat with runner (ignored by SSM)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--moe-loss-weight", type=float, default=0.01)

    # Latent code
    ap.add_argument("--code-dim", type=int, default=64)
    ap.add_argument("--code-l2", type=float, default=1e-3)
    ap.add_argument("--code-dropout", type=float, default=0.1)

    # Training
    ap.add_argument("--train-block", type=int, default=128)
    ap.add_argument("--train-overlap", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pred-k", type=int, default=1)

    # Eval
    ap.add_argument("--embed-block", type=int, default=256)
    ap.add_argument("--embed-step", type=int, default=256)
    ap.add_argument("--adapt-steps", type=int, default=50)
    ap.add_argument("--adapt-lr", type=float, default=0.1)

    ap.add_argument("--save", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")

    args = ap.parse_args()

    # ── Expand globs ──────────────────────────────────────────────────────────
    def expand_globs(paths: List[str]) -> List[str]:
        expanded: List[str] = []
        for p in paths:
            if any(ch in p for ch in ("*", "?", "[")):
                matches = sorted(glob.glob(p))
                expanded.extend(matches if matches else [p])
            else:
                expanded.append(p)
        return expanded

    args.train_fasta = expand_globs(args.train_fasta)
    args.eval_fasta = expand_globs(args.eval_fasta)

    os.makedirs(args.save, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("   SSM (Selective State Space Model / Mamba-style) Sample LM")
    print("=" * 70)
    device = torch.device(args.device)
    print(f"[Device] {device}")

    tok_cfg = TokenizerCfg(
        kmer=args.kmer, stride=args.stride,
        vocab_size=args.vocab_size, no_rc=args.no_rc, max_tokens=args.max_tokens,
    )

    def sample_id_from_path(p: str) -> str:
        base = os.path.basename(p)
        for suf in [".fna.gz", ".fa.gz", ".fasta.gz", ".fna", ".fa", ".fasta"]:
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        return base

    train_paths = args.train_fasta
    eval_paths = args.eval_fasta
    train_ids = [sample_id_from_path(p) for p in train_paths]
    eval_ids = [sample_id_from_path(p) for p in eval_paths]

    # ── Tokenize ──────────────────────────────────────────────────────────────
    t_wall_start = time.time()
    print(f"\n[Tokenize] train_samples={len(train_paths)} eval_samples={len(eval_paths)}")
    train_tokens: Dict[str, np.ndarray] = {}
    eval_tokens: Dict[str, np.ndarray] = {}

    t_tok_start = time.time()
    for p, sid in zip(train_paths, train_ids):
        t = tokenize_sample_fasta(p, tok_cfg)
        train_tokens[sid] = t
        print(f"  - {sid}: tokens={t.size}")

    for p, sid in zip(eval_paths, eval_ids):
        t = tokenize_sample_fasta(p, tok_cfg)
        eval_tokens[sid] = t
        split = "train" if sid in train_tokens else "eval"
        print(f"  - {sid} ({split}): tokens={t.size}")
    t_tok_end = time.time()
    print(f"\n⏱  Tokenisation: {t_tok_end - t_tok_start:.2f}s")

    # ── Build training examples ───────────────────────────────────────────────
    sample_index = {sid: i for i, sid in enumerate(train_ids)}
    examples: List[Tuple[int, np.ndarray]] = []
    for sid in train_ids:
        blocks = make_blocks(train_tokens[sid], args.train_block, args.train_overlap)
        for b in blocks:
            if b.size >= 2:
                examples.append((sample_index[sid], b))

    if len(examples) == 0:
        raise SystemExit("No training blocks produced; check FASTA content.")

    print(f"[Data] {len(examples)} training blocks (block={args.train_block}, overlap={args.train_overlap})")

    # ── Model ─────────────────────────────────────────────────────────────────
    d_model = args.d_model
    code_dim = args.code_dim if args.code_dim > 0 else d_model

    lm = SSMCausalLM(
        args.vocab_size,
        d_model=d_model,
        n_layers=args.layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        max_len=max(2048, args.train_block + 2),
    ).to(device)

    n_params = sum(p.numel() for p in lm.parameters())
    print(f"[Model] SSM  layers={args.layers}  d_model={d_model}  d_state={args.d_state}  "
          f"d_conv={args.d_conv}  expand={args.expand}")
    print(f"[Model] Total LM params: {n_params:,}")

    code_table = nn.Embedding(len(train_ids), code_dim).to(device)
    code_proj = nn.Linear(code_dim, d_model, bias=False).to(device) if code_dim != d_model else None
    code_do = nn.Dropout(args.code_dropout)

    params = list(lm.parameters()) + list(code_table.parameters())
    if code_proj is not None:
        params += list(code_proj.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr)

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  TRAINING  —  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    print(f"{'─'*50}")
    t_train_start = time.time()
    lm.train()

    for epoch in range(1, args.epochs + 1):
        t_ep_start = time.time()
        batches = batch_iter(examples, args.batch, seed=args.seed + epoch)
        losses = []
        for batch in batches:
            sidx = torch.tensor([s for s, _ in batch], dtype=torch.long, device=device)
            x = torch.tensor(np.stack([b for _, b in batch], axis=0), dtype=torch.long, device=device)

            # prefix embedding
            code = code_table(sidx)
            if code_proj is not None:
                code = code_proj(code)
            code = code_do(code).unsqueeze(1)  # (B,1,d)
            tok_emb = lm.tok(x)                # (B,T,d)
            x_emb = torch.cat([code, tok_emb], dim=1)  # (B,T+1,d)

            logits, aux = lm.forward_embeds(x_emb, latent_code=code.squeeze(1))

            nll_sum = 0.0
            for k in range(1, args.pred_k + 1):
                logits_use = logits[:, :-k, :]
                targets = x[:, k - 1 :]
                nll_sum += nll_from_logits(logits_use, targets).mean()
            nll = nll_sum / args.pred_k

            l2 = code_table.weight.pow(2).sum(dim=1).mean()
            loss = nll + args.code_l2 * l2

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))

        t_ep_end = time.time()
        print(f"  [Epoch {epoch}/{args.epochs}] loss={np.mean(losses):.4f}  "
              f"({t_ep_end - t_ep_start:.1f}s)")

    t_train_end = time.time()
    print(f"\n⏱  Training total: {t_train_end - t_train_start:.2f}s")
    lm.eval()

    # ── Save config ───────────────────────────────────────────────────────────
    cfg_out = dict(vars(args))
    cfg_out.update({"train_ids": train_ids, "eval_ids": eval_ids,
                    "architecture": "ssm_mamba", "n_params": n_params})
    with open(os.path.join(args.save, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)

    # Extract train codes
    with torch.no_grad():
        train_codes = code_table.weight.detach().cpu().numpy().astype(np.float32)

    # ── Helper: score and embed ───────────────────────────────────────────────
    def score_and_pool(tokens: np.ndarray, code_vec_d: torch.Tensor) -> Tuple[float, np.ndarray]:
        blocks = make_blocks(tokens, args.train_block, args.train_overlap)
        if not blocks:
            return float("inf"), np.zeros((d_model,), dtype=np.float32)
        all_h = []
        all_nll = []
        for b in blocks:
            x = torch.tensor(b[None, :], dtype=torch.long, device=device)
            with torch.no_grad():
                tok_emb = lm.tok(x)
                x_emb = torch.cat([code_vec_d[None, None, :], tok_emb], dim=1)
                B_, T1, _ = x_emb.shape
                pos = torch.arange(T1, device=device).unsqueeze(0).expand(B_, T1)
                h = x_emb + lm.pos(pos)
                for layer in lm.layers:
                    h = layer(h)
                h = lm.ln(h)
                logits = lm.head(h)
                logits_use = logits[:, :-1, :]
                nll_tok = nll_from_logits(logits_use, x).squeeze(0)
                all_nll.append(nll_tok.detach().cpu().numpy())
                # Pool hidden states (excluding prefix)
                h_tok = h[:, 1:, :].squeeze(0).detach().cpu().numpy()
                all_h.append(h_tok)
        nll_all = np.concatenate(all_nll, axis=0)
        nll_mean = float(nll_all.mean())
        ppl = ppl_from_nll(nll_mean)
        h_all = np.concatenate(all_h, axis=0)
        pooled = h_all.mean(axis=0).astype(np.float32)
        return ppl, pooled

    # Mean code
    with torch.no_grad():
        mean_code = torch.tensor(train_codes.mean(axis=0), dtype=torch.float32, device=device)
        mean_code_d = code_proj(mean_code) if code_proj is not None else mean_code

    # ── Evaluate + adapt ──────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  EVALUATION + ADAPTATION  —  adapt_steps={args.adapt_steps}")
    print(f"{'─'*50}")

    rows = []
    latent_out = []
    pooled_out = []
    sample_out_ids = []

    t_eval_start = time.time()
    for sid in eval_ids:
        t_sample_start = time.time()
        tokens = eval_tokens[sid]
        split = "train" if sid in sample_index else "eval"

        # base: use mean train code
        ppl_base, pooled_base = score_and_pool(tokens, mean_code_d)

        # init code
        if split == "train":
            with torch.no_grad():
                c0 = torch.tensor(train_codes[sample_index[sid]], dtype=torch.float32, device=device)
        else:
            with torch.no_grad():
                c0 = torch.tensor(train_codes.mean(axis=0), dtype=torch.float32, device=device)

        # Adapt
        c = c0.clone().detach().requires_grad_(True)
        opt_c = torch.optim.Adam([c], lr=args.adapt_lr)

        for p in lm.parameters():
            p.requires_grad_(False)
        if code_proj is not None:
            for p in code_proj.parameters():
                p.requires_grad_(False)

        best = float("inf")
        best_c = None

        all_adapt_blocks = make_blocks(tokens, args.train_block, args.train_overlap)
        MAX_ADAPT_BLOCKS = 256

        for step in range(1, args.adapt_steps + 1):
            opt_c.zero_grad(set_to_none=True)
            c_d = code_proj(c) if code_proj is not None else c
            if not all_adapt_blocks:
                break
            if len(all_adapt_blocks) > MAX_ADAPT_BLOCKS:
                rng_adapt = np.random.default_rng(args.seed + step)
                idxs = rng_adapt.choice(len(all_adapt_blocks), MAX_ADAPT_BLOCKS, replace=False)
                blocks = [all_adapt_blocks[i] for i in idxs]
            else:
                blocks = all_adapt_blocks

            nll_sum = torch.tensor(0.0, device=device)
            n_blocks = 0
            for b in blocks:
                x = torch.tensor(b[None, :], dtype=torch.long, device=device)
                tok_emb = lm.tok(x)
                x_emb = torch.cat([c_d[None, None, :], tok_emb], dim=1)
                logits, _ = lm.forward_embeds(x_emb, latent_code=c_d.unsqueeze(0))

                nll_s = 0.0
                for k in range(1, args.pred_k + 1):
                    logits_use = logits[:, :-k, :]
                    targets = x[:, k - 1 :]
                    nll_s += float(nll_from_logits(logits_use, targets).mean())
                nll_sum = nll_sum + torch.tensor(nll_s / args.pred_k, device=device)
                n_blocks += 1

            nll_mean = nll_sum / max(1, n_blocks)
            loss = nll_mean + args.code_l2 * c.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([c], 1.0)
            opt_c.step()

            with torch.no_grad():
                cur = float(nll_mean.detach().cpu())
                if cur < best:
                    best = cur
                    best_c = c.detach().clone()

        if best_c is None:
            best_c = c0.detach().clone()

        with torch.no_grad():
            best_c_d = code_proj(best_c) if code_proj is not None else best_c
            ppl_adapt, pooled_adapt = score_and_pool(tokens, best_c_d)

        delta = float(ppl_base - ppl_adapt)
        t_sample_end = time.time()

        rows.append({
            "sample_id": sid,
            "split": split,
            "tokens": int(tokens.size),
            "ppl_base": float(ppl_base),
            "ppl_adapt": float(ppl_adapt),
            "delta_ppl": delta,
            "time_s": round(t_sample_end - t_sample_start, 2),
        })

        latent_out.append(best_c.detach().cpu().numpy().astype(np.float32))
        pooled_out.append(pooled_adapt.astype(np.float32))
        sample_out_ids.append(sid)

        print(f"  [Eval] {sid}  split={split}  ppl_base={ppl_base:.3f}  "
              f"ppl_adapt={ppl_adapt:.3f}  Δ={delta:.3f}  ({t_sample_end - t_sample_start:.1f}s)")

    t_eval_end = time.time()
    print(f"\n⏱  Evaluation total: {t_eval_end - t_eval_start:.2f}s")

    latent_arr = np.stack(latent_out, axis=0)
    pooled_arr = np.stack(pooled_out, axis=0)

    # ── Nearest-neighbors ─────────────────────────────────────────────────────
    def nn_report(emb: np.ndarray) -> Dict[str, Tuple[str, float]]:
        if emb.shape[0] < 2:
            return {}
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn.fit(emb)
        dist, idx = nn.kneighbors(emb)
        out = {}
        for i, sid in enumerate(sample_out_ids):
            j = idx[i, 1]
            out[sid] = (sample_out_ids[j], float(dist[i, 1]))
        return out

    nn_lat = nn_report(latent_arr)
    nn_pool = nn_report(pooled_arr)

    for r in rows:
        sid = r["sample_id"]
        if sid in nn_lat:
            r["nn1_latent"] = nn_lat[sid][0]
            r["nn1_latent_dist"] = nn_lat[sid][1]
        if sid in nn_pool:
            r["nn1_pooled"] = nn_pool[sid][0]
            r["nn1_pooled_dist"] = nn_pool[sid][1]

    # PCA
    pca = PCA(n_components=2, random_state=args.seed)
    p2 = pca.fit_transform(pooled_arr)

    # ── Save artifacts ────────────────────────────────────────────────────────
    import csv

    summary_path = os.path.join(args.save, "samples_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(args.save, "sample_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sample_out_ids) + "\n")

    np.save(os.path.join(args.save, "embeddings_latent.npy"), latent_arr)
    np.save(os.path.join(args.save, "embeddings_pooled.npy"), pooled_arr)

    pca_path = os.path.join(args.save, "samples_pca2.csv")
    with open(pca_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "pc1", "pc2"])
        for sid, (a, b) in zip(sample_out_ids, p2):
            w.writerow([sid, float(a), float(b)])

    torch.save({
        "lm_state": lm.state_dict(),
        "code_table": code_table.state_dict(),
        "code_proj": (code_proj.state_dict() if code_proj is not None else None),
        "args": cfg_out,
    }, os.path.join(args.save, "model.pt"))

    # ── Timing summary ────────────────────────────────────────────────────────
    t_wall_end = time.time()
    timing = {
        "tokenisation_s": round(t_tok_end - t_tok_start, 2),
        "training_s": round(t_train_end - t_train_start, 2),
        "evaluation_s": round(t_eval_end - t_eval_start, 2),
        "total_wall_s": round(t_wall_end - t_wall_start, 2),
    }
    with open(os.path.join(args.save, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\n{'='*70}")
    print("  TIMING SUMMARY")
    print(f"{'='*70}")
    print(f"  Tokenisation : {timing['tokenisation_s']:>8.2f}s")
    print(f"  Training     : {timing['training_s']:>8.2f}s")
    print(f"  Evaluation   : {timing['evaluation_s']:>8.2f}s")
    print(f"  ─────────────────────────────")
    print(f"  Total wall   : {timing['total_wall_s']:>8.2f}s")
    print(f"{'='*70}")

    print(f"\n[OK] wrote: {summary_path}")
    print(f"[OK] wrote: {pca_path}")
    print(f"[OK] wrote: timing.json")
    print("[OK] wrote embeddings: embeddings_latent.npy + embeddings_pooled.npy")
    print(f"[OK] wrote model: {os.path.join(args.save, 'model.pt')}")

    # Leaderboard
    print("\n=== Top samples by ppl_adapt (higher = more novel even after adaptation) ===")
    rows_sorted = sorted(rows, key=lambda x: x["ppl_adapt"], reverse=True)
    for r in rows_sorted:
        sid = r["sample_id"]
        nn1 = r.get("nn1_latent", "-")
        d1 = r.get("nn1_latent_dist", float("nan"))
        print(f"{sid:12s} split={r['split']:5s} ppl_adapt={r['ppl_adapt']:.3f} "
              f"delta={r['delta_ppl']:.3f} nn1_lat={nn1} ({d1:.4f})  [{r['time_s']:.1f}s]")


if __name__ == "__main__":
    main()
