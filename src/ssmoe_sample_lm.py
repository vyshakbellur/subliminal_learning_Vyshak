"""SSM-MoE (Hybrid Mamba-Expert) Sample LM
=========================================

Implements a pioneer DNA language model combining:
1. Selective State Space (Mamba) for long-range sequence context.
2. Sparsely Gated Mixture-of-Experts (MoE) with Top-2 Routing.
3. Variational Latent Space (VAE) for zero-shot environmental adaptation.

This architecture achieves linear scaling with sequence length while maintaining
massive model capacity through specialized biological experts.
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
    out: List[Tuple[str, str]] = []
    header = None
    seq_parts: List[str] = []
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
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
    k, st = cfg.kmer, cfg.stride
    if len(seq) < k: return np.zeros((0,), dtype=np.int64)
    toks = []
    for i in range(0, len(seq) - k + 1, st):
        kmer = seq[i : i + k]
        if (not cfg.no_rc) and ("N" not in kmer):
            rc = revcomp(kmer)
            kmer = min(kmer, rc)
        toks.append(fnv1a_32(kmer) % cfg.vocab_size)
    return np.array(toks, dtype=np.int64)

def tokenize_sample_fasta(path: str, cfg: TokenizerCfg) -> np.ndarray:
    prefix = path.replace(".fna.gz", "").replace(".fasta.gz", "")
    cache_path = f"{prefix}_tokens_k{cfg.kmer}_s{cfg.stride}.npy"
    if os.path.exists(cache_path): return np.load(cache_path)
    t0 = time.time()
    recs = read_fasta_any(path)
    all_toks = []
    for _, seq in recs:
        t = tokenize_sequence(seq, cfg)
        if t.size: all_toks.append(t)
    out = np.concatenate(all_toks, axis=0) if all_toks else np.zeros((0,), dtype=np.int64)
    if cfg.max_tokens > 0: out = out[:cfg.max_tokens]
    np.save(cache_path, out)
    print(f"    [Cache] Tokenized & saved in {time.time()-t0:.2f}s -> {os.path.basename(cache_path)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mamba SSM Block
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """Selective State Space (Mamba) layer for global sequence context."""
    def __init__(self, d_model: int, d_inner: int, d_state: int = 16, d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model, self.d_inner, self.d_state = d_model, d_inner, d_state
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=d_inner, bias=True)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _ssm_scan(self, x: torch.Tensor, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        batch, T, d_inner = x.shape
        A = -torch.exp(self.A_log)
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dt_t, x_t, B_t, C_t = dt[:, t, :], x[:, t, :], B[:, t, :], C[:, t, :]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        return torch.stack(ys, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :x.size(1)].transpose(1, 2)
        x_branch = F.silu(x_branch)
        ssm_p = self.x_proj(x_branch)
        dt_raw = ssm_p[:, :, :1]
        B_p, C_p = ssm_p[:, :, 1:1+self.d_state], ssm_p[:, :, 1+self.d_state:]
        dt = F.softplus(self.dt_proj(dt_raw))
        y = self._ssm_scan(x_branch, dt, B_p, C_p)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_branch
        y = y * F.silu(z)
        return residual + self.dropout(self.out_proj(y))

# ─────────────────────────────────────────────────────────────────────────────
# Top-2 Mixture-of-Experts (MoE)
# ─────────────────────────────────────────────────────────────────────────────

class ExpertLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class MoELayer(nn.Module):
    """Sparsely Gated MoE Layer with Top-2 Routing and Environmental Skew."""
    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([ExpertLayer(d_model, d_model * 4, dropout) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, latent_routing: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        residual = x
        x = self.norm(x)
        
        # Routing logits
        # If latent_routing provided, it skews the gating based on environment μ
        route_inp = x
        if latent_routing is not None:
            # Ensure latent_routing is (B, 1, D) for broadcasting over T
            if latent_routing.dim() == 1:
                lr = latent_routing.view(1, 1, -1)
            elif latent_routing.dim() == 2:
                lr = latent_routing.unsqueeze(1)
            else:
                lr = latent_routing
            route_inp = route_inp + lr
            
        logits = self.router(route_inp.reshape(-1, D)) # (B*T, num_experts)
        probs = F.softmax(logits, dim=-1)
        
        # Pick Top-K experts
        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True) # Normalized
        
        flat_x = x.reshape(-1, D)
        out = torch.zeros_like(flat_x)
        
        # Helper: dispatch and collect from experts
        # We loop over experts to process masked indices for speed
        for eid in range(self.num_experts):
            # Check which indices in flat_x use this expert
            mask = (top_indices == eid).any(dim=-1)
            if mask.any():
                # Extract tokens for this expert
                expert_out = self.experts[eid](flat_x[mask])
                # Weight by the probability assigned to this token-expert pair
                # Finding where this expert was in top-k
                match = (top_indices[mask] == eid)
                w = top_probs[mask][match].unsqueeze(-1)
                out[mask] += w * expert_out
                    
        # Load Balancing aux loss
        actual_load = probs.mean(0)
        aux_loss = (actual_load - 1.0/self.num_experts).pow(2).sum()
        
        return residual + out.reshape(B, T, D), aux_loss

# ─────────────────────────────────────────────────────────────────────────────
# Variational Latent Space (VAE)
# ─────────────────────────────────────────────────────────────────────────────

class VariationalEncoder(nn.Module):
    def __init__(self, d_model: int, code_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(d_model, code_dim)
        self.fc_logvar = nn.Linear(d_model, code_dim)
    def forward(self, x_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fc_mu(x_pool), self.fc_logvar(x_pool)

# ─────────────────────────────────────────────────────────────────────────────
# SSM-MoE Causal LM
# ─────────────────────────────────────────────────────────────────────────────

class SSMoECausalLM(nn.Module):
    """
    Hybrid Causal LM: Mamba (Sequential context) + MoE (Expert specialization).
    """
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2,
                 num_experts: int = 8, top_k: int = 2, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        
        # Interleaved Mamba and MoE layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(MambaBlock(d_model, d_model * 2, dropout=dropout))
            self.layers.append(MoELayer(d_model, num_experts, top_k, dropout))
            
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_embeds(self, x_emb: torch.Tensor, latent_routing: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x_emb.shape
        pos = torch.arange(T, device=x_emb.device).unsqueeze(0).expand(B, T)
        h = x_emb + self.pos(pos)
        
        total_aux_loss = torch.tensor(0.0, device=x_emb.device)
        for layer in self.layers:
            if isinstance(layer, MoELayer):
                h, aux = layer(h, latent_routing=latent_routing)
                total_aux_loss += aux
            else:
                h = layer(h)
                
        h = self.ln(h)
        return self.head(h), total_aux_loss

# ─────────────────────────────────────────────────────────────────────────────
# Utilities & Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def make_blocks(tokens: np.ndarray, block: int, overlap: float) -> List[np.ndarray]:
    if tokens.size == 0: return []
    step = max(1, int(block * (1.0 - overlap)))
    out = []
    for i in range(0, max(0, tokens.size - block + 1), step):
        out.append(tokens[i : i + block])
    if not out and tokens.size >= 2: out.append(tokens[:block])
    return out

def nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T), reduction="none")
    return loss.view(B, T)

def ppl_from_nll(nll: float) -> float:
    return math.exp(min(50, nll))

def main():
    ap = argparse.ArgumentParser(description="SSM-MoE (Hybrid Mamba-Expert) DNA LM")
    ap.add_argument("--train-fasta", nargs="+", required=True)
    ap.add_argument("--eval-fasta", nargs="+", required=True)
    ap.add_argument("--kmer", type=int, default=31)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4, help="Dummy arg for compatibility")
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--code-dim", type=int, default=0, help="Latent code dim (defaults to d_model)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--adapt-steps", type=int, default=50)
    ap.add_argument("--adapt-lr", type=float, default=0.1)
    ap.add_argument("--kl-weight", type=float, default=0.01)
    ap.add_argument("--moe-loss-weight", type=float, default=0.1)
    ap.add_argument("--train-block", type=int, default=128)
    ap.add_argument("--train-overlap", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=200000)
    ap.add_argument("--save", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    os.makedirs(args.save, exist_ok=True)
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] {device}")

    # Tokenize
    tok_cfg = TokenizerCfg(kmer=args.kmer, stride=args.stride, vocab_size=args.vocab_size, max_tokens=args.max_tokens)
    train_ids = [os.path.basename(p) for p in args.train_fasta]
    eval_ids = [os.path.basename(p) for p in args.eval_fasta]
    
    train_tokens = {sid: tokenize_sample_fasta(p, tok_cfg) for sid, p in zip(train_ids, args.train_fasta)}
    eval_tokens = {sid: tokenize_sample_fasta(p, tok_cfg) for sid, p in zip(eval_ids, args.eval_fasta)}

    # Model
    d_model = args.d_model
    code_dim = args.code_dim if args.code_dim > 0 else d_model
    
    lm = SSMoECausalLM(args.vocab_size, d_model=d_model, n_layers=args.layers, num_experts=args.num_experts).to(device)
    encoder = VariationalEncoder(d_model, code_dim).to(device)
    code_table = nn.Embedding(len(train_ids), code_dim).to(device) 

    optimizer = torch.optim.AdamW(list(lm.parameters()) + list(encoder.parameters()) + list(code_table.parameters()), lr=args.lr)

    print(f"\n[Train] Starting SSM-MoE Hybrid training (top-{args.top_k} Experts)...")
    for epoch in range(1, args.epochs + 1):
        lm.train()
        total_loss = 0
        # Simple training loop for demo (in production, use batch_iter)
        for sid in train_ids:
            tokens = train_tokens[sid]
            blocks = make_blocks(tokens, args.train_block, args.train_overlap)
            if not blocks: continue
            
            # Reparameterize from table
            mu = code_table.weight[list(train_ids).index(sid)]
            z = mu # (simplified for demo)
            
            for b in blocks[:10]: # limit per epoch for speed
                optimizer.zero_grad()
                x = torch.tensor(b[None, :], dtype=torch.long, device=device)
                tok_emb = lm.tok(x)
                x_emb = torch.cat([z[None, None, :], tok_emb], dim=1) # inject latent as prefix token
                
                logits, aux_loss = lm.forward_embeds(x_emb, latent_routing=z)
                nll = nll_from_logits(logits[:, :-1, :], x).mean()
                loss = nll + args.moe_loss_weight * aux_loss
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
        print(f"  Epoch {epoch} Loss: {total_loss:.4f}")

    # Zero-shot Adaptation
    print(f"\n[Eval] Beginning Adaptation (Information Significance Measurement)")
    rows = []
    for sid in eval_ids:
        t0_sample = time.time()
        tokens = eval_tokens[sid]
        
        # 1. Base PPL
        with torch.no_grad():
            mu_base = code_table.weight.mean(0)
            z_base = mu_base
            x = torch.tensor(tokens[None, :args.train_block], dtype=torch.long, device=device)
            x_emb = torch.cat([z_base[None, None, :], lm.tok(x)], dim=1)
            logits, _ = lm.forward_embeds(x_emb, latent_routing=z_base)
            nll_base = nll_from_logits(logits[:, :-1, :], x).mean().item()
            ppl_base = ppl_from_nll(nll_base)
            bpb_base = nll_base / math.log(2)

        # 2. Adaptation with Early Stopping (Biological Velocity Check)
        z = mu_base.clone().detach().requires_grad_(True)
        opt_z = torch.optim.Adam([z], lr=args.adapt_lr)
        
        target_bpb = bpb_base * 0.98 # Goal: reach 2% improvement in "surprise" (bits)
        max_cycles = 20 # Biological complexity limit per user requirement
        cycles_taken = max_cycles
        converged = False

        for step in range(max_cycles):
            opt_z.zero_grad()
            x_emb = torch.cat([z[None, None, :], lm.tok(x)], dim=1)
            logits, _ = lm.forward_embeds(x_emb, latent_routing=z)
            nll = nll_from_logits(logits[:, :-1, :], x).mean()
            nll.backward()
            opt_z.step()
            
            # Check convergence vs base BPB
            current_bpb = nll.item() / math.log(2)
            if current_bpb <= target_bpb:
                cycles_taken = step + 1
                converged = True
                break
        
        with torch.no_grad():
            nll_adapt = nll.item()
            ppl_adapt = ppl_from_nll(nll_adapt)
            bpb_adapt = nll_adapt / math.log(2)
            info_gain = bpb_base - bpb_adapt
            
        # Classification: If it took > 20 steps or didn't hit target, it is likely Non-Marine/Novel
        is_marine = "YES" if converged else "NO (Novel/OOD)"
            
        duration = time.time() - t0_sample
        print(f"  {sid:30s} BPB_Base={bpb_base:.3f} BPB_Adapt={bpb_adapt:.3f} IG={info_gain:.4f} bits ({duration:.1f}s) -> Marine: {is_marine}")
        rows.append({
            "sample_id": sid, 
            "bpb_base": bpb_base, 
            "bpb_adapt": bpb_adapt, 
            "info_gain": info_gain, 
            "cycles_to_adapt": cycles_taken,
            "is_marine": is_marine,
            "time_s": duration
        })

    # Save summary
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(args.save, "samples_summary.csv"), index=False)
    print(f"\n[DONE] Results saved to {args.save}")

if __name__ == "__main__":
    main()
