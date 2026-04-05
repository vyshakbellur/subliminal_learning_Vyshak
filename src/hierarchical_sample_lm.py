"""Subliminal Sample LM

Implements a DNA language model over hashed k-mers with a *sample-specific latent code*.

Key idea ("subliminal learning"):
- Train an LM (next-token prediction) while also learning a small embedding vector per training sample.
- At inference on a new sample, freeze the LM and *adapt* only the latent code to fit the sample.

Outputs:
- Latent-code embeddings (train + inferred)
- Pooled contextual embeddings
- Per-sample perplexity before and after adaptation (novelty + adaptability)

This is intentionally lightweight and CPU-friendly.
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


# -----------------------------
# IO
# -----------------------------

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


# -----------------------------
# Tokenization (hashed k-mers)
# -----------------------------

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
    import time
    cache_path = path.replace(".fna.gz", f"_tokens_k{cfg.kmer}_s{cfg.stride}.npy").replace(".fasta.gz", f"_tokens_k{cfg.kmer}_s{cfg.stride}.npy")
    if os.path.exists(cache_path):
        out = np.load(cache_path)
        if cfg.max_tokens and cfg.max_tokens > 0:
            out = out[:cfg.max_tokens]
        return out

    t0 = time.time()
    recs = read_fasta_any(path)
    all_toks: List[np.ndarray] = []
    for _h, seq in recs:
        t = tokenize_sequence(seq, cfg)
        if t.size:
            all_toks.append(t)
    
    out = np.concatenate(all_toks, axis=0) if all_toks else np.zeros((0,), dtype=np.int64)
    if cfg.max_tokens and cfg.max_tokens > 0:
        out = out[:cfg.max_tokens]
    np.save(cache_path, out)
    print(f"    [Cache] Tokenized & saved .npy in {time.time()-t0:.2f}s -> {os.path.basename(cache_path)}")
    
    # Reload explicitly directly into full RAM space for computational speed
    return np.load(cache_path)


# -----------------------------
# Model: Hierarchical Agents
# -----------------------------

class MoEFeedForward(nn.Module):
    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.latent_router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor, latent_code: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        
        route_logits = self.router(x_flat) # (B*T, num_experts)
        if latent_code is not None:
            # latent_code is (B, D)
            lat_route = self.latent_router(latent_code) # (B, num_experts)
            lat_route = lat_route.unsqueeze(1).expand(B, T, self.num_experts) # (B, T, num_experts)
            route_logits = route_logits + lat_route.reshape(B * T, self.num_experts)
            
        routing_probs = F.softmax(route_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True) # normalize
        
        out = torch.zeros_like(x_flat)
        # process experts
        for i, expert in enumerate(self.experts):
            idx, nth_expert = torch.where(selected_experts == i)
            if idx.numel() > 0:
                expert_out = expert(x_flat[idx])
                w = routing_weights[idx, nth_expert].unsqueeze(-1)
                out[idx] += w * expert_out

        # Balance loss computing
        prob_mean = routing_probs.mean(dim=0)
        zeros = torch.zeros_like(route_logits)
        zeros.scatter_(1, selected_experts, 1.0)
        frac_mean = zeros.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(prob_mean * frac_mean)
        
        return out.view(B, T, D), balance_loss


class LatentMoETransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoEFeedForward(d_model, num_experts, top_k, dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, latent_code: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        
        h = self.norm2(x)
        moe_out, bal_loss = self.moe(h, latent_code=latent_code)
        x = x + moe_out
        return x, bal_loss


class LocalAgentEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layers: int, dropout: float, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            LatentMoETransformerLayer(d_model, n_heads, dropout, num_experts, top_k)
            for _ in range(layers)
        ])
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor):
        bal_loss = 0.0
        for layer in self.layers:
            x, bal = layer(x, mask=causal_mask, latent_code=None)
            bal_loss = bal_loss + bal
        return x, bal_loss

class GlobalPatternEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layers: int, dropout: float, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            LatentMoETransformerLayer(d_model, n_heads, dropout, num_experts, top_k)
            for _ in range(layers)
        ])
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor, latent_code: torch.Tensor = None):
        bal_loss = 0.0
        for layer in self.layers:
            x, bal = layer(x, mask=causal_mask, latent_code=latent_code)
            bal_loss = bal_loss + bal
        return x, bal_loss


class TinyCausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1, max_len: int = 2048, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Use simple map-reduce chunks
        self.chunk_size = 16 

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        
        self.local_agent = LocalAgentEncoder(d_model, n_heads, n_layers, dropout, num_experts, top_k)
        self.global_agent = GlobalPatternEncoder(d_model, n_heads, max(1, n_layers//2), dropout, num_experts, top_k)
        
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_embeds(self, x_emb: torch.Tensor, latent_code: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x_emb.shape
        pos = torch.arange(T, device=x_emb.device).unsqueeze(0).expand(B, T)
        x = x_emb + self.pos(pos)
        
        # 1. Local Processing (Agents extracting sequence knowledge)
        causal_local = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h_local, bal_local = self.local_agent(x, causal_local)
        
        # 2. Extract synopses
        idxs = list(range(self.chunk_size - 1, T, self.chunk_size))
        if not idxs or idxs[-1] != T - 1:
            idxs.append(T - 1)
            
        synopses = h_local[:, idxs, :] # (B, num_chunks, d_model)
        
        # 3. Global processing (Pattern finding across chunks)
        if latent_code is not None:
            # Shift the synopses downstream naturally, prefixed by latent code
            global_in = torch.cat([latent_code.unsqueeze(1), synopses], dim=1)
        else:
            global_in = synopses
            
        causal_global = torch.triu(torch.ones(global_in.size(1), global_in.size(1), device=x.device, dtype=torch.bool), diagonal=1)
        h_global, bal_global = self.global_agent(global_in, causal_global, latent_code=latent_code)
        
        if latent_code is not None:
            global_context = h_global[:, :-1, :]
        else:
            global_context = h_global
            
        # 4. Fuse global insight back into local stream
        out = h_local.clone()
        for i, idx in enumerate(idxs):
            start = 0 if i == 0 else idxs[i-1] + 1
            out[:, start:idx+1, :] += global_context[:, i:i+1, :]
            
        out = self.ln(out)
        return self.head(out), bal_local + bal_global

    def tokens_to_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        return self.tok(input_ids) + self.pos(pos)


# -----------------------------
# Batching utilities
# -----------------------------


def make_blocks(tokens: np.ndarray, block: int, overlap: float) -> List[np.ndarray]:
    if tokens.size == 0:
        return []
    step = max(1, int(block * (1.0 - overlap)))
    out = []
    for i in range(0, max(0, tokens.size - block + 1), step):
        out.append(tokens[i : i + block])
    if not out and tokens.size >= 2:
        # very short: still return one block (truncate)
        out.append(tokens[:block])
    return out


def batch_iter(examples: List[Tuple[int, np.ndarray]], batch_size: int, seed: int) -> List[List[Tuple[int, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(examples))
    rng.shuffle(idx)
    batches = []
    for i in range(0, len(idx), batch_size):
        batches.append([examples[j] for j in idx[i : i + batch_size]])
    return batches


# -----------------------------
# Training + evaluation primitives
# -----------------------------


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


# -----------------------------
# Main routine
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-fasta", nargs="+", required=True, help="Training sample FASTA/FASTA.GZ files")
    ap.add_argument("--eval-fasta", nargs="+", required=True, help="Evaluation sample FASTA/FASTA.GZ files")

    ap.add_argument("--kmer", type=int, default=31)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--no-rc", action="store_true")

    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--moe-loss-weight", type=float, default=0.01)

    ap.add_argument("--code-dim", type=int, default=64, help="Latent sample code dim (defaults to d_model)")
    ap.add_argument("--code-l2", type=float, default=1e-3)
    ap.add_argument("--kl-weight", type=float, default=0.01, help="VAE KL-Divergence weight (beta)")
    ap.add_argument("--code-dropout", type=float, default=0.1)

    ap.add_argument("--train-block", type=int, default=128)
    ap.add_argument("--train-overlap", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)

    ap.add_argument("--max-tokens", type=int, default=0, help="Hard truncation limit per genome")
    ap.add_argument("--pred-k", type=int, default=1, help="Multi-token prediction window")

    ap.add_argument("--embed-block", type=int, default=256)
    ap.add_argument("--embed-step", type=int, default=256, help="Not used (kept for compatibility)")

    ap.add_argument("--adapt-steps", type=int, default=50)
    ap.add_argument("--adapt-lr", type=float, default=0.1)

    ap.add_argument("--save", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")

    args = ap.parse_args()

    def expand_globs(paths: List[str]) -> List[str]:
        expanded: List[str] = []
        for p in paths:
            if any(ch in p for ch in ("*", "?", "[")):
                matches = sorted(glob.glob(p))
                if not matches:
                    # Keep the literal path so the error message is informative.
                    expanded.append(p)
                else:
                    expanded.extend(matches)
            else:
                expanded.append(p)
        return expanded

    args.train_fasta = expand_globs(args.train_fasta)
    args.eval_fasta = expand_globs(args.eval_fasta)

    os.makedirs(args.save, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("[subliminal] fixed-subliminal-sample-lm-0.3.0")
    device = torch.device(args.device)
    print(f"[Device] {device}")

    tok_cfg = TokenizerCfg(kmer=args.kmer, stride=args.stride, vocab_size=args.vocab_size, no_rc=args.no_rc, max_tokens=args.max_tokens)

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

    # Tokenize
    t0_total = time.time()
    print(f"[Tokenize] train_samples={len(train_paths)} eval_samples={len(eval_paths)}")
    train_tokens: Dict[str, np.ndarray] = {}
    eval_tokens: Dict[str, np.ndarray] = {}

    for p, sid in zip(train_paths, train_ids):
        t = tokenize_sample_fasta(p, tok_cfg)
        train_tokens[sid] = t
        print(f"  - {sid}: tokens={t.size}")

    for p, sid in zip(eval_paths, eval_ids):
        t = tokenize_sample_fasta(p, tok_cfg)
        eval_tokens[sid] = t
        split = "train" if sid in train_tokens else "eval"
        print(f"  - {sid} ({split}): tokens={t.size}")
        
    t1 = time.time()
    print(f"\n[Time Elapsed] Tokenization / Caching Phase completed in {t1 - t0_total:.2f} seconds.")

    # Build training examples (sample_index, token_block)
    sample_index = {sid: i for i, sid in enumerate(train_ids)}
    examples: List[Tuple[int, np.ndarray]] = []
    for sid in train_ids:
        blocks = make_blocks(train_tokens[sid], args.train_block, args.train_overlap)
        for b in blocks:
            if b.size >= 2:
                examples.append((sample_index[sid], b))

    if len(examples) == 0:
        raise SystemExit("No training blocks produced; check FASTA content.")

    # Model
    d_model = args.d_model
    code_dim = args.code_dim if args.code_dim > 0 else d_model

    lm = TinyCausalTransformer(
        args.vocab_size, 
        d_model=d_model, 
        n_layers=args.layers, 
        n_heads=args.heads, 
        dropout=args.dropout, 
        max_len=max(2048, args.train_block + 1),
        num_experts=args.num_experts,
        top_k=args.top_k
    ).to(device)
    # sample codes for training set (variational: mu, logvar)
    code_table = nn.Embedding(len(train_ids), 2 * code_dim).to(device)
    code_proj = nn.Linear(code_dim, d_model, bias=False).to(device) if code_dim != d_model else None
    code_do = nn.Dropout(args.code_dropout)

    params = list(lm.parameters()) + list(code_table.parameters())
    if code_proj is not None:
        params += list(code_proj.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr)

    # Training
    print("[Time] Beginning Training Phase...")
    t_train_start = time.time()
    lm.train()
    for epoch in range(1, args.epochs + 1):
        batches = batch_iter(examples, args.batch, seed=args.seed + epoch)
        losses = []
        for batch in batches:
            sidx = torch.tensor([s for s, _ in batch], dtype=torch.long, device=device)
            x = torch.tensor(np.stack([b for _, b in batch], axis=0), dtype=torch.long, device=device)

            # prefix embedding (Variational)
            latent_params = code_table(sidx) # (B, 2*code_dim)
            mu, logvar = latent_params.chunk(2, dim=-1)
            
            # Reparameterization
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            code = mu + eps * std if lm.training else mu 
            
            if code_proj is not None:
                code = code_proj(code)
            code = code_do(code).unsqueeze(1)  # (B,1,d)
            tok_emb = lm.tok(x)  # (B,T,d)
            x_emb = torch.cat([code, tok_emb], dim=1)  # (B,T+1,d)

            logits, bal_loss = lm.forward_embeds(x_emb, latent_code=code.squeeze(1))  # (B,T+1,V)
            
            nll_sum = 0.0
            for k in range(1, args.pred_k + 1):
                logits_use = logits[:, :-k, :]
                targets = x[:, k-1:]
                nll_sum += float(nll_from_logits(logits_use, targets).mean())
            nll = nll_sum / args.pred_k

            # KL Recovery + L2 + MoE Balancer
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            l2 = (code_table.weight.pow(2).sum(dim=1).mean())
            loss = nll + args.kl_weight * kl + args.code_l2 * l2 + args.moe_loss_weight * bal_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))

        print(f"[Train] epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f}")

    print(f"[Time Elapsed] Full Map-Reduce Training executed in {time.time() - t_train_start:.2f} seconds.")
    lm.eval()

    # Save config
    cfg_out = dict(vars(args))
    cfg_out.update({"train_ids": train_ids, "eval_ids": eval_ids})
    with open(os.path.join(args.save, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)

    # Extract train codes (mus)
    with torch.no_grad():
        latent_params = code_table.weight.detach()
        train_codes = latent_params.chunk(2, dim=-1)[0].cpu().numpy().astype(np.float32)
        if code_proj is not None:
            # keep original code space; projection is model internal
            pass

    # Helper to score and embed a sample given a code vector (d_model)
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
                # forward through encoder to get hidden states
                # reuse forward_embeds but we also want h; easiest: re-run parts
                B, T1, _ = x_emb.shape
                pos = torch.arange(T1, device=device).unsqueeze(0).expand(B, T1)
                h = x_emb + lm.pos(pos)
                causal = torch.triu(torch.ones(T1, T1, device=device, dtype=torch.bool), diagonal=1)
                
                logits, _ = lm.forward_embeds(x_emb, latent_code=code_vec_d.unsqueeze(0))
                logits_use = logits[:, :-1, :]
                nll_tok = nll_from_logits(logits_use, x).squeeze(0)  # (T,)
                all_nll.append(nll_tok.detach().cpu().numpy())
                
                h_tok = logits[:, -1, :].squeeze(0).detach().cpu().numpy()
                all_h.append(h_tok)
        nll_all = np.concatenate(all_nll, axis=0)
        nll_mean = float(nll_all.mean())
        ppl = ppl_from_nll(nll_mean)
        h_all = np.stack(all_h, axis=0)
        pooled = h_all.mean(axis=0).astype(np.float32)
        return ppl, pooled

    # Build a mean code in d_model space for base scoring
    with torch.no_grad():
        mean_code = torch.tensor(train_codes.mean(axis=0), dtype=torch.float32, device=device)
        if code_proj is not None:
            mean_code_d = code_proj(mean_code)
        else:
            mean_code_d = mean_code

    # Evaluate + adapt
    rows = []
    latent_out = []
    pooled_out = []
    sample_out_ids = []

    t_eval_start = time.time()
    for sid in eval_ids:
        tokens = eval_tokens[sid]
        split = "train" if sid in sample_index else "eval"

        # base: use mean train code
        ppl_base, pooled_base = score_and_pool(tokens, mean_code_d)

        # start code init
        if split == "train":
            with torch.no_grad():
                c0 = torch.tensor(train_codes[sample_index[sid]], dtype=torch.float32, device=device)
        else:
            with torch.no_grad():
                c0 = torch.tensor(train_codes.mean(axis=0), dtype=torch.float32, device=device)

        # Adapt by optimizing only code vector in *code space* (before projection)
        c = c0.clone().detach().requires_grad_(True)
        opt_c = torch.optim.Adam([c], lr=args.adapt_lr)

        # freeze LM
        for p in lm.parameters():
            p.requires_grad_(False)
        if code_proj is not None:
            for p in code_proj.parameters():
                p.requires_grad_(False)

        best = float("inf")
        best_c = None

        # Pre-compute blocks once; subsample per step to avoid OOM
        all_adapt_blocks = make_blocks(tokens, args.train_block, args.train_overlap)
        MAX_ADAPT_BLOCKS = 256  # cap to prevent GPU OOM on large samples

        for step in range(1, args.adapt_steps + 1):
            opt_c.zero_grad(set_to_none=True)
            # Project to d_model if needed
            c_d = code_proj(c) if code_proj is not None else c
            if not all_adapt_blocks:
                break
            # Subsample blocks if too many
            if len(all_adapt_blocks) > MAX_ADAPT_BLOCKS:
                rng_adapt = np.random.default_rng(args.seed + step)
                idxs = rng_adapt.choice(len(all_adapt_blocks), MAX_ADAPT_BLOCKS, replace=False)
                blocks = [all_adapt_blocks[i] for i in idxs]
            else:
                blocks = all_adapt_blocks
            # Accumulate NLL without keeping full graph — use running mean
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
                    targets = x[:, k-1:]
                    nll_s += float(nll_from_logits(logits_use, targets).mean())
                nll = nll_s / args.pred_k
                
                nll_sum = nll_sum + torch.tensor(nll, device=device)
                n_blocks += 1
            nll_mean = nll_sum / max(1, n_blocks)
            # regularize code
            loss = nll_mean + args.code_l2 * (c.pow(2).mean())
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

        rows.append({
            "sample_id": sid,
            "split": split,
            "tokens": int(tokens.size),
            "ppl_base": float(ppl_base),
            "ppl_adapt": float(ppl_adapt),
            "delta_ppl": delta,
        })

        latent_out.append(best_c.detach().cpu().numpy().astype(np.float32))
        pooled_out.append(pooled_adapt.astype(np.float32))
        sample_out_ids.append(sid)

        print(f"[Eval] {sid} split={split} ppl_base={ppl_base:.3f} ppl_adapt={ppl_adapt:.3f} delta={delta:.3f}")

    print(f"[Time Elapsed] Zero-Shot Adaptation executed in {time.time() - t_eval_start:.2f} seconds.")

    latent_arr = np.stack(latent_out, axis=0)
    pooled_arr = np.stack(pooled_out, axis=0)

    # Nearest neighbors on latent + pooled
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

    # PCA projection for quick plotting
    pca = PCA(n_components=2, random_state=args.seed)
    p2 = pca.fit_transform(pooled_arr)

    # Save artifacts
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

    print(f"[OK] wrote: {summary_path}")
    print(f"[OK] wrote: {pca_path}")
    print("[OK] wrote embeddings: embeddings_latent.npy + embeddings_pooled.npy")
    print(f"[OK] wrote model: {os.path.join(args.save, 'model.pt')}")

    # console leaderboard
    print("\n=== Top samples by ppl_adapt (higher = more novel even after adaptation) ===")
    rows_sorted = sorted(rows, key=lambda x: x["ppl_adapt"], reverse=True)
    for r in rows_sorted:
        sid = r["sample_id"]
        nn1 = r.get("nn1_latent", "-")
        d1 = r.get("nn1_latent_dist", float("nan"))
        print(f"{sid:12s} split={r['split']:5s} ppl_adapt={r['ppl_adapt']:.3f} delta={r['delta_ppl']:.3f} nn1_lat={nn1} ({d1:.4f})")


if __name__ == "__main__":
    main()
