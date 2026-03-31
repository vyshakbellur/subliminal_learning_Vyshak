#!/usr/bin/env python3
"""
plot_results.py
===============
Quick terminal + CSV summary of a completed subliminal model run.
Works without matplotlib — prints an ASCII scatter plot.

Usage:
    .venv/bin/python scripts/plot_results.py [--run-dir outputs/real_mgnify]
"""
import argparse
import csv
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ascii_scatter(rows, width=60, height=20):
    """Print a tiny ASCII scatter of PC1 vs PC2, coloured by environment."""
    if not rows:
        return
    xs = [r["pc1"] for r in rows]
    ys = [r["pc2"] for r in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xrange = xmax - xmin or 1
    yrange = ymax - ymin or 1

    grid = [[" "] * width for _ in range(height)]
    env_symbols = {}
    sym_pool = ["M", "F", "S", "G", "A", "B", "C", "D"]
    for r in rows:
        env = r["env"]
        if env not in env_symbols:
            env_symbols[env] = sym_pool[len(env_symbols) % len(sym_pool)]
        sym = env_symbols[env]
        col = int((r["pc1"] - xmin) / xrange * (width - 1))
        row = height - 1 - int((r["pc2"] - ymin) / yrange * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = sym

    print("\n  PC2")
    print("  ↑")
    for g in grid:
        print("  |" + "".join(g))
    print("  +" + "─" * width + "→ PC1")
    print()
    for env, sym in env_symbols.items():
        print(f"    [{sym}] {env}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(PROJECT_ROOT / "outputs" / "real_mgnify"))
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    summary_path = run_dir / "samples_summary.csv"
    pca_path     = run_dir / "samples_pca2.csv"

    if not summary_path.exists():
        print(f"[Error] {summary_path} not found. Run the model first.")
        return 1

    # ── Summary table ─────────────────────────────────────────────────────────
    with open(summary_path) as f:
        summary = list(csv.DictReader(f))

    print(f"\n{'='*72}")
    print(f"{'SAMPLE':30s} {'ENV':12s} {'PPL_BASE':>10} {'PPL_ADAPT':>10} {'DELTA':>8}")
    print(f"{'='*72}")
    for r in sorted(summary, key=lambda x: float(x["ppl_adapt"]), reverse=True):
        sid   = r["sample_id"]
        split = r["split"]
        env   = "marine" if "marine" in sid else ("freshwater" if "freshwater" in sid else split)
        pb    = float(r["ppl_base"])
        pa    = float(r["ppl_adapt"])
        delta = float(r["delta_ppl"])
        flag  = " ← HIGH (out-of-domain?)" if pa > 1000 else ""
        print(f"  {sid:28s} {env:12s} {pb:10.1f} {pa:10.1f} {delta:8.2f}{flag}")
    print()

    # ── Interpretation ────────────────────────────────────────────────────────
    marine_ppl     = [float(r["ppl_adapt"]) for r in summary if "marine"     in r["sample_id"]]
    freshwater_ppl = [float(r["ppl_adapt"]) for r in summary if "freshwater" in r["sample_id"]]

    if marine_ppl and freshwater_ppl:
        m_mean = sum(marine_ppl)     / len(marine_ppl)
        f_mean = sum(freshwater_ppl) / len(freshwater_ppl)
        print(f"  Mean ppl_adapt — marine: {m_mean:.1f}   freshwater: {f_mean:.1f}")
        if f_mean > m_mean * 1.3:
            print("  ✓ Freshwater samples show higher perplexity → OOD signal detected!")
        elif m_mean > f_mean * 1.3:
            print("  ✓ Marine samples show higher perplexity (model trained on freshwater?)")
        else:
            print("  ~ Environments show similar perplexity (may need more data or epochs).")
        print()

    # ── PCA scatter ───────────────────────────────────────────────────────────
    if pca_path.exists():
        with open(pca_path) as f:
            pca_rows = list(csv.DictReader(f))
        rows = []
        for r in pca_rows:
            sid = r["sample_id"]
            env = "marine" if "marine" in sid else ("freshwater" if "freshwater" in sid else "other")
            rows.append({"sid": sid, "env": env, "pc1": float(r["pc1"]), "pc2": float(r["pc2"])})
        print("  ASCII PCA plot (pooled embeddings):")
        ascii_scatter(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
