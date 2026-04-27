"""
Subliminal Sample LM - Enterprise Entry Point

Implements a DNA language model over hashed k-mers with a *sample-specific latent code*,
orchestrating modularized Hierarchical MoE Agents.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.data.dataset import batch_iter, make_blocks
from src.data.tokenizer import TokenizerCfg, tokenize_sample_fasta
from src.models.architectures import TinyCausalTransformer
from src.utils.metrics import nll_from_logits, ppl_from_nll

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
    ap.add_argument("--code-dim", type=int, default=64)
    ap.add_argument("--code-l2", type=float, default=1e-3)
    ap.add_argument("--code-dropout", type=float, default=0.1)
    ap.add_argument("--train-block", type=int, default=128)
    ap.add_argument("--train-overlap", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--pred-k", type=int, default=1)
    ap.add_argument("--embed-block", type=int, default=256)
    ap.add_argument("--embed-step", type=int, default=256)
    ap.add_argument("--adapt-steps", type=int, default=50)
    ap.add_argument("--adapt-lr", type=float, default=0.1)
    ap.add_argument("--save", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")

    args = ap.parse_args()

    def expand_globs(paths):
        expanded = []
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

    print("[subliminal] Hierarchical MoE Enterprise v1.0.0")
    device = torch.device(args.device)

    tok_cfg = TokenizerCfg(kmer=args.kmer, stride=args.stride, vocab_size=args.vocab_size, no_rc=args.no_rc, max_tokens=args.max_tokens)

    def sample_id_from_path(p: str) -> str:
        base = os.path.basename(p)
        for suf in [".fna.gz", ".fa.gz", ".fasta.gz", ".fna", ".fa", ".fasta"]:
            if base.endswith(suf):
                return base[: -len(suf)]
        return base

    train_ids = [sample_id_from_path(p) for p in args.train_fasta]
    eval_ids = [sample_id_from_path(p) for p in args.eval_fasta]

    train_tokens = {}
    eval_tokens = {}

    for p, sid in zip(args.train_fasta, train_ids):
        t = tokenize_sample_fasta(p, tok_cfg)
        train_tokens[sid] = t

    for p, sid in zip(args.eval_fasta, eval_ids):
        t = tokenize_sample_fasta(p, tok_cfg)
        eval_tokens[sid] = t

    sample_index = {sid: i for i, sid in enumerate(train_ids)}
    examples = []
    for sid in train_ids:
        blocks = make_blocks(train_tokens[sid], args.train_block, args.train_overlap)
        for b in blocks:
            if b.size >= 2:
                examples.append((sample_index[sid], b))

    if not examples:
        raise SystemExit("No training blocks produced; check FASTA content.")

    d_model = args.d_model
    code_dim = args.code_dim if args.code_dim > 0 else d_model

    lm = TinyCausalTransformer(args.vocab_size, d_model=d_model, n_layers=args.layers, n_heads=args.heads, dropout=args.dropout, max_len=max(2048, args.train_block + 1), num_experts=args.num_experts, top_k=args.top_k).to(device)
    code_table = nn.Embedding(len(train_ids), code_dim).to(device)
    code_proj = nn.Linear(code_dim, d_model, bias=False).to(device) if code_dim != d_model else None
    code_do = nn.Dropout(args.code_dropout)

    params = list(lm.parameters()) + list(code_table.parameters())
    if code_proj: params += list(code_proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    lm.train()
    for epoch in range(1, args.epochs + 1):
        batches = batch_iter(examples, args.batch, seed=args.seed + epoch)
        losses = []
        for batch in batches:
            sidx = torch.tensor([s for s, _ in batch], dtype=torch.long, device=device)
            x = torch.tensor(np.stack([b for _, b in batch], axis=0), dtype=torch.long, device=device)

            code = code_table(sidx)
            if code_proj: code = code_proj(code)
            code = code_do(code).unsqueeze(1)
            tok_emb = lm.tok(x)
            x_emb = torch.cat([code, tok_emb], dim=1)

            logits, bal_loss = lm.forward_embeds(x_emb, latent_code=code.squeeze(1))
            
            nll_sum = 0.0
            for k in range(1, args.pred_k + 1):
                nll_sum += nll_from_logits(logits[:, :-k, :], x[:, k-1:]).mean()
            nll = nll_sum / args.pred_k
            
            l2 = code_table.weight.pow(2).sum(dim=1).mean()
            loss = nll + args.code_l2 * l2 + args.moe_loss_weight * bal_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(loss.item())

        print(f"[Train] epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f}")

    # Further Eval/Adapt loop continues here, utilizing the modularized scripts...
    print("[Execute] Completed basic pass for modularized training execution.")

if __name__ == "__main__":
    main()
