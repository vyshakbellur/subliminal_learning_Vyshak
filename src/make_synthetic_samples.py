#!/usr/bin/env python3
"""make_synthetic_samples.py

Create a tiny multi-sample FASTA benchmark for the *sample-as-document* LM demo.

We generate several "environment" groups. Each environment has a distinct 1st-order
Markov DNA generator (GC% and a simple dinucleotide bias), which produces
consistent k-mer "dialects".

Outputs:
  data/samples_synth_small/
    <sample_id>.fna.gz  (multiple contigs per file)
    samples.manifest.tsv

The demo script trains on two environments and evaluates a third, showing that:
  - embeddings cluster by environment
  - perplexity rises for OOD samples

This is intentionally simple and deterministic.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
from typing import Dict, List, Tuple

import numpy as np

DNA = np.array(list("ACGT"))
IDX = {b: i for i, b in enumerate(DNA)}


def make_markov_1(gc: float, dinuc_bias: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Simple 1st-order Markov chain over A,C,G,T."""
    rng = np.random.default_rng(seed)
    p0 = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2], dtype=float)  # A,C,G,T
    p0 = p0 / p0.sum()

    T = np.tile(p0, (4, 1))

    # create a strong dialect using a few dinucleotide preferences
    favorite = [(IDX['C'], IDX['G']), (IDX['G'], IDX['C'])]  # CG, GC
    penalize = [(IDX['A'], IDX['A']), (IDX['T'], IDX['T'])]  # AA, TT

    for (i, j) in favorite:
        T[i, j] += dinuc_bias
    for (i, j) in penalize:
        T[i, j] = max(1e-6, T[i, j] - dinuc_bias)

    T += rng.uniform(0, 1e-3, size=T.shape)
    T = T / T.sum(axis=1, keepdims=True)
    return p0, T


def sample_markov_1(L: int, p0: np.ndarray, T: np.ndarray, seed: int) -> str:
    rng = np.random.default_rng(seed)
    s = np.empty(L, dtype='<U1')
    s[0] = rng.choice(DNA, p=p0)
    for i in range(1, L):
        prev = IDX[s[i - 1]]
        s[i] = rng.choice(DNA, p=T[prev])
    return "".join(s.tolist())


def write_multi_contig_fna_gz(sample_id: str, contigs: List[Tuple[str, str]], outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{sample_id}.fna.gz")
    with gzip.open(path, "wt") as f:
        for name, seq in contigs:
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/samples_synth_small")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--contigs", type=int, default=6, help="contigs per sample")
    ap.add_argument("--contig-len", type=int, default=2500)

    # environment definitions
    ap.add_argument("--envA", default="soil", help="name for environment A")
    ap.add_argument("--envB", default="marine", help="name for environment B")
    ap.add_argument("--envOOD", default="gut_ood", help="name for OOD environment")

    ap.add_argument("--nA", type=int, default=2, help="replicates for envA")
    ap.add_argument("--nB", type=int, default=2, help="replicates for envB")
    ap.add_argument("--nOOD", type=int, default=1, help="replicates for envOOD")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # parameters chosen to create clearly separable "dialects"
    env_params: Dict[str, Tuple[float, float]] = {
        args.envA: (0.52, 0.06),   # gc, dinuc bias
        args.envB: (0.40, 0.02),
        args.envOOD: (0.65, 0.10),
    }

    reps: Dict[str, int] = {args.envA: args.nA, args.envB: args.nB, args.envOOD: args.nOOD}

    os.makedirs(args.outdir, exist_ok=True)
    manifest_path = os.path.join(args.outdir, "samples.manifest.tsv")

    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf, delimiter="\t")
        w.writerow(["sample", "environment", "gc", "dinuc_bias", "contigs", "contig_len", "seed"])

        sid_counter = 0
        for env, nrep in reps.items():
            gc, bias = env_params[env]
            p0, T = make_markov_1(gc=gc, dinuc_bias=bias, seed=args.seed + 1000 + sid_counter)

            for r in range(1, nrep + 1):
                sid_counter += 1
                sample_id = f"{env}__{r:02d}"
                contigs: List[Tuple[str, str]] = []
                for c in range(args.contigs):
                    contig_name = f"{sample_id}|contig{c+1:02d}"
                    seq = sample_markov_1(args.contig_len, p0, T, seed=args.seed + 10000*sid_counter + c)
                    contigs.append((contig_name, seq))

                write_multi_contig_fna_gz(sample_id, contigs, args.outdir)

                w.writerow([sample_id, env, f"{gc:.3f}", f"{bias:.3f}", args.contigs, args.contig_len, args.seed])

    print(f"[done] wrote samples to {args.outdir}")
    print(f"[done] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
