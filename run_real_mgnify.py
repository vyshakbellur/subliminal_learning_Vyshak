#!/usr/bin/env python3
"""
run_real_mgnify.py
==================
One-command pipeline:
  1. Fetch real assembled metagenome samples from EBI MGnify
     (5 marine from MGYS00006028 + 5 freshwater from MGYS00006752)
  2. Run the Subliminal Sample LM on them

Usage (from project root, using the pre-built venv):
    .venv/bin/python run_real_mgnify.py

Options:
    --n-per-env N     samples per environment (default: 5)
    --max-mb N        max MB to download per sample (default: 30)
    --skip-download   skip download, re-use data/samples_real/
    --device auto|mps|cpu
    --epochs N        training epochs (default: 2)
    --adapt-steps N   adaptation steps per eval sample (default: 25)
    --dry-run         just print what would be downloaded
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Optional

# ─── Study configuration ──────────────────────────────────────────────────────
STUDIES = {
    "marine":     "MGYS00005294",   # bioGEOTRACES (Global Ocean)
    "freshwater": "MGYS00006752",   # Lakes & Ponds (Global)
    "arctic":     "MGYS00005221",   # Arctic Ocean
    "benthic":    "MGYS00005063",   # Marine Benthic (Sediment)
    "deepsea":    "MGYS00002008",   # Deep Sea (Global)
}
API_BASE     = "https://www.ebi.ac.uk/metagenomics/api/v1"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data" / "samples_real"
OUT_DIR      = PROJECT_ROOT / "outputs" / "real_mgnify"

# Headers to be polite to the EBI API
HEADERS = {
    "Accept":     "application/json",
    "User-Agent": "subliminal-lm-research/1.0 (contact: vyshakathreya@gmail.com)",
}


# ─── MGnify API helpers ───────────────────────────────────────────────────────

def api_get(path: str, params: Optional[dict] = None) -> dict:
    """GET from MGnify API, return parsed JSON."""
    url = f"{API_BASE}/{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def get_analyses(study_id: str, want: int) -> list[str]:
    """
    Return up to `want` analysis accession IDs from a study.
    Tries assembly first, then metagenomic, then falls back to all.
    """
    def _fetch(experiment_type: Optional[str]) -> list[str]:
        ids: list[str] = []
        page = 1
        limit = 9999 if want <= 0 else want
        while len(ids) < limit:
            p: dict = {"page_size": 100, "page": page}
            if experiment_type:
                p["experiment_type"] = experiment_type
            try:
                data = api_get(f"studies/{study_id}/analyses", p)
            except urllib.error.URLError as exc:
                print(f"  [API] network error page {page}: {exc}")
                break
            items = data.get("data", [])
            if not items:
                break
            for item in items:
                ids.append(item["id"])
                if len(ids) >= limit:
                    break
            # follow pagination
            if not data.get("links", {}).get("next"):
                break
            page += 1
        return ids

    # Try assembly first (contigs), then metagenomic (shotgun reads)
    for etype in ["assembly", "metagenomic"]:
        ids = _fetch(etype)
        if ids:
            print(f"  [API] Found {len(ids)} {etype} analyses")
            return ids
    print(f"  [API] No assembly/metagenomic analyses; trying all experiment types")
    ids = _fetch(None)
    return ids


def find_fasta_url(analysis_id: str) -> Optional[str]:
    """
    Inspect /analyses/{id}/downloads and return the URL for the best
    nucleotide FASTA (gzip): assembled contigs or processed reads.
    Returns None if not found.
    """
    try:
        data = api_get(f"analyses/{analysis_id}/downloads")
    except Exception as exc:
        print(f"  [API] downloads endpoint failed for {analysis_id}: {exc}")
        return None

    candidates: list[tuple[int, str]] = []
    for item in data.get("data", []):
        attrs = item.get("attributes", {})
        desc  = attrs.get("description", {}).get("label", "").lower()
        alias = attrs.get("alias", "").lower()
        fmt   = attrs.get("file-format", {}).get("name", "").lower()
        url   = item.get("links", {}).get("self", "")
        if not url:
            continue

        # Skip amino acid (.faa) and ORF (.ffn) files — we need nucleotide DNA
        if url.endswith(".faa.gz") or url.endswith(".ffn.gz"):
            continue

        # Score: higher = more likely to be useful nucleotide FASTA
        score = 0

        # Assembled contigs (best for LM)
        if "contig" in desc or "contig" in alias:
            score += 5
        if "assembled" in desc or "assembled" in alias:
            score += 4

        # Processed nucleotide reads (shotgun metagenomes — also good)
        if "processed" in desc and ("nucleotide" in desc or "reads" in desc):
            score += 3
        elif "processed" in desc:
            score += 1

        # Prefer FASTA format
        if "fasta" in fmt or url.endswith(".fasta.gz") or url.endswith(".fna.gz"):
            score += 2

        # Penalize annotation-filtered subsets (we want ALL reads)
        if "annotation" in desc or "interpro" in alias:
            score -= 2
        if "nofunction" in alias or "pcds" in desc:
            score -= 1

        # Prefer the first chunk (_1) if multi-part
        if "_1.fasta.gz" in alias or "_1.fna.gz" in alias:
            score += 1

        if score > 0:
            candidates.append((score, url))

    if not candidates:
        print(f"  [Downloads] No suitable FASTA found for {analysis_id}")
        return None
    candidates.sort(reverse=True)
    best_score, best_url = candidates[0]
    print(f"  [Downloads] Best: score={best_score} {best_url.split('/')[-1]}")
    return best_url


# ─── Streaming download with size cap ─────────────────────────────────────────

def stream_download_gz(url: str, out_path: Path, max_bytes: int) -> bool:
    """
    Download up to max_bytes from url.
    Handles both pre-compressed .gz and plain FASTA.
    Saves as a valid .fna.gz file.
    Returns True on success.
    """
    print(f"  ↓ {url[:80]}...")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": HEADERS["User-Agent"],
            "Range": f"bytes=0-{max_bytes - 1}",   # request only what we need
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read(max_bytes)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # Range not satisfiable – server sent full file
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]}),
                timeout=120,
            ) as resp:
                raw = resp.read(max_bytes)
        else:
            print(f"  ✗ HTTP {exc.code}: {exc}")
            return False
    except Exception as exc:
        print(f"  ✗ Download error: {exc}")
        return False

    # Decompress whatever we got, then re-save as valid gzip
    is_gz = raw[:2] == b"\x1f\x8b"
    try:
        if is_gz:
            # Partial gzip: use raw decompressor, ignore checksum
            d = zlib.decompressobj(zlib.MAX_WBITS | 16)
            try:
                plain = d.decompress(raw)
            except zlib.error:
                plain = d.decompress(raw[:-4]) if len(raw) > 4 else b""
        else:
            plain = raw
    except Exception as exc:
        print(f"  ✗ Decompress error: {exc}")
        return False

    if not plain:
        print("  ✗ Zero bytes after decompress")
        return False

    # Keep only complete FASTA records (don't cut mid-sequence)
    text = plain.decode("utf-8", errors="replace")
    parts = text.split(">")
    complete_parts = parts[:-1] if len(parts) > 1 else parts  # drop last (may be truncated)
    clean = ">".join(complete_parts)
    if not clean.startswith(">"):
        clean = clean  # already fine if only 1 record

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        f.write(clean)

    # Quick sanity: count sequences
    n_seqs = clean.count(">")
    total_bases = len(clean) - clean.count("\n") - clean.count(">")
    print(f"  ✓ {out_path.name}  ({n_seqs} sequences, ~{total_bases:,} bases)")
    return n_seqs > 0


def verify_fasta_gz(path: Path) -> int:
    """Count sequences in a .fna.gz; return 0 if invalid."""
    try:
        count = 0
        with gzip.open(path, "rt", errors="replace") as f:
            for line in f:
                if line.startswith(">"):
                    count += 1
                    if count >= 5:
                        break
        return count
    except Exception:
        return 0


# ─── GPU & Scaling Helpers ───────────────────────────────────────────────────

def get_gpu_tier(device_name: str) -> str:
    """Categorize GPU for auto-scaling hyperparameters."""
    name = device_name.upper()
    if any(x in name for x in ["H100", "A100", "A800"]):
        return "ULTRA"
    if any(x in name for x in ["L4", "V100", "A10G", "T4"]):
        # T4 is borderline, but better than a Mac CPU for most things
        return "PRO"
    return "BASE"

# ─── Model runner ─────────────────────────────────────────────────────────────

def run_model(
    train_files: list[Path],
    eval_files: list[Path],
    out_dir: Path,
    device: str,
    epochs: int,
    adapt_steps: int,
    arch: str,
    kl_weight: float = 0.01,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-scale based on GPU tier
    gpu_name = "cpu"
    try:
        import torch
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
        elif device == "mps":
            gpu_name = "Apple Silicon"
    except: pass
    
    tier = get_gpu_tier(gpu_name)
    
    # Scaling Map
    configs = {
        "ULTRA": {"d_model": 128, "layers": 4, "heads": 8,  "batch": 32},
        "PRO":   {"d_model": 64,  "layers": 2, "heads": 4,  "batch": 16},
        "BASE":  {"d_model": 32,  "layers": 1, "heads": 4,  "batch": 8},
    }
    cfg = configs[tier]
    print(f"\n[Auto-Scale] Detected GPU Tier: {tier} ({gpu_name})")
    print(f"             Applying: d_model={cfg['d_model']}, layers={cfg['layers']}, batch={cfg['batch']}")

    if arch == "ssmoe":
        lm_script = PROJECT_ROOT / "src" / "ssmoe_sample_lm.py"
    elif arch == "hierarchical_moe":
        lm_script = PROJECT_ROOT / "src" / "hierarchical_moe_lm.py"
    elif arch == "hierarchical":
        lm_script = PROJECT_ROOT / "src" / "hierarchical_sample_lm.py"
    else:
        lm_script = PROJECT_ROOT / "src" / "subliminal_sample_lm.py"
        
    cmd = [
        sys.executable, "-u", str(lm_script),
        "--train-fasta", *[str(p) for p in train_files],
        "--eval-fasta",  *[str(p) for p in eval_files],
        "--kmer",         "31",
        "--stride",       "15" if arch == "hierarchical" else "31",
        "--vocab-size",   "32768",
        "--d-model",      str(cfg["d_model"]),
        "--layers",       str(cfg["layers"]),
        "--heads",        str(cfg["heads"]),
        "--epochs",       str(epochs),
        "--train-block",  "128",
        "--batch",        str(cfg["batch"]),
        "--adapt-steps",  str(adapt_steps),
        "--adapt-lr",     "0.2",
        "--kl-weight",    str(kl_weight),
        "--max-tokens",   "200000",
        "--save",         str(out_dir),
        "--device",       device,
        "--seed",         "42",
    ]
    print(f"\n[Model] Launching {arch} pipeline...")
    subprocess.check_call(cmd)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-env",     type=int, default=30,   help="Samples per environment (0 for ALL)")
    ap.add_argument("--train-split",   type=float, default=0.8, help="Ratio of Marine samples for training")
    ap.add_argument("--max-mb",        type=int, default=30,   help="Max MB per sample download")
    ap.add_argument("--data-dir",      default=str(PROJECT_ROOT / "data" / "samples_real"), help="Custom directory for FASTA samples")
    ap.add_argument("--skip-download", action="store_true",    help="Use existing data in --data-dir")
    ap.add_argument("--device",        default="auto",         help="auto | mps | cpu")
    ap.add_argument("--epochs",        type=int, default=1,    help="Training epochs")
    ap.add_argument("--adapt-steps",   type=int, default=3,   help="Adaptation steps per sample")
    ap.add_argument("--kl-weight",     type=float, default=0.01, help="Variational KL weight")
    ap.add_argument("--dry-run",       action="store_true",    help="Show plan without downloading")
    ap.add_argument("--skip-training",  action="store_true",    help="Download data only (Save credits/GPU compute)")
    ap.add_argument("--arch",          choices=["moe", "hierarchical", "ssm", "hierarchical_moe", "ssmoe"], default="hierarchical", help="Architecture to execute")
    ap.add_argument("--seed",          type=int, default=42,   help="Random seed for data split")
    args = ap.parse_args()

    # ── Device detection ──────────────────────────────────────────────────────
    try:
        import torch
        if args.device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
                print("[Device] Apple Silicon MPS ✓  (GPU acceleration enabled)")
            else:
                device = "cpu"
                print("[Device] CPU (no MPS found)")
        else:
            device = args.device
            print(f"[Device] {device} (user-specified)")
    except ImportError:
        print("[Error] PyTorch not found. Make sure you run with the venv:")
        print("        .venv/bin/python run_real_mgnify.py")
        return 1

    max_bytes = args.max_mb * 1024 * 1024

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[Dry run] Would download:")
        for env, study_id in STUDIES.items():
            print(f"  {env:12s}  study={study_id}  n={args.n_per_env}  max={args.max_mb}MB each")
        
        # Determine GPU Tier for dry-run insight
        gpu_name = "cpu"
        try:
            import torch
            if device == "cuda": gpu_name = torch.cuda.get_device_name(0)
            elif device == "mps": gpu_name = "Apple Silicon"
        except: pass
        tier = get_gpu_tier(gpu_name)
        
        print(f"\n[Auto-Scale Plan]")
        print(f"  Device: {device} ({gpu_name}) -> Tier: {tier}")
        if tier == "ULTRA":   print("  Scaling: High-Capacity (128-dim, 4 layers, 32 batch)")
        elif tier == "PRO":   print("  Scaling: Professional (64-dim, 2 layers, 16 batch)")
        else:                 print("  Scaling: Baseline (32-dim, 1 layer, 8 batch)")
            
        print(f"\n  Output dir: {OUT_DIR}")
        print(f"  Epochs: {args.epochs}, adapt_steps: {args.adapt_steps}, arch: {args.arch}")
        return 0

    # ── Data Inventory & Download ─────────────────────────────────────────────
    raw_dir = Path(args.data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Inventory] Checking samples in: {raw_dir}")
    sample_files: dict[str, list[Path]] = {env: [] for env in STUDIES}
    
    # 1. First, check what we already have
    for env in STUDIES:
        # Look for env__*.fna.gz or env__*.fasta.gz
        found = sorted(list(raw_dir.glob(f"{env}__*.fna.gz")) + list(raw_dir.glob(f"{env}__*.fasta.gz")))
        sample_files[env] = [p for p in found if verify_fasta_gz(p) > 0]
        if sample_files[env]:
            print(f"  ✓ {env:12s}: {len(sample_files[env])} valid samples found")
        else:
            print(f"  ! {env:12s}: 0 samples found")

    # 2. Download missing samples if needed
    if not args.skip_download:
        for env, study_id in STUDIES.items():
            current_count = len(sample_files[env])
            target = args.n_per_env if args.n_per_env > 0 else 999  # 0 means "all" but we cap at a reasonable default for safety
            
            if current_count < target:
                needed = target - current_count
                print(f"\n[Fetch] {env.upper()} (Target {target}, Have {current_count}, Need {needed}) — study {study_id}")
                analyses = get_analyses(study_id, target)
                # Filter out ones we already have by checking ana_id in filename
                existing_ids = [p.name.split("__")[-1].replace(".fna.gz", "").replace(".fasta.gz", "") for p in sample_files[env]]
                new_analyses = [a for a in analyses if a not in existing_ids][:needed]
                
                for i, ana_id in enumerate(new_analyses, current_count + 1):
                    out = raw_dir / f"{env}__{i:02d}_{ana_id}.fna.gz"
                    url = find_fasta_url(ana_id)
                    if url and stream_download_gz(url, out, max_bytes):
                        sample_files[env].append(out)
                    else:
                        out.unlink(missing_ok=True)

    n_marine     = len(sample_files["marine"])
    n_freshwater = len(sample_files["freshwater"])
    print(f"\n[Data] Final Inventory: marine={n_marine}  freshwater={n_freshwater}")

    if n_marine + n_freshwater < 2:
        print("\n[Error] Too few samples. Possible causes:")
        print("  • No internet access to EBI (try connecting without VPN)")
        print("  • The study has no assembly analyses (try --n-per-env 10)")
        print("\nFallback: run the synthetic demo with:  python scripts/run_demo.py")
        return 1

    # ── Logic for Training Split (Per category) ──────────────────────────────
    import random
    random.seed(args.seed)
    
    train_files = []
    eval_files = []
    
    for env, files in sample_files.items():
        random.shuffle(files)
        # Take 20 for train, next 10 for eval
        env_train = files[:20]
        env_eval = files[20:30]
        
        train_files.extend(env_train)
        eval_files.extend(env_eval)
        print(f"  {env:12s}: {len(env_train)} train, {len(env_eval)} eval")
    
    # Eval on EVERYTHING downloaded (train + test) or just test?
    # Usually we want to see train samples again to see "adaptation" quality,
    # but the user asked for 20 samples of training and 10 samples of testing.
    # To be clear, we will evaluate on the 50 test samples.
    # If the user wants to see the train samples too in the output, we can add them.
    # The original script adds EVERYTHING to eval_files. Let's keep that but ensure we have at least our 50 test.
    
    # Remove duplicates but keep order (train followed by test)
    seen = set()
    final_eval = []
    for x in eval_files + train_files:
        if x not in seen:
            final_eval.append(x)
            seen.add(x)
    eval_files = final_eval
    
    print(f"\n[Split] Total samples: {sum(len(f) for f in sample_files.values())}")
    print(f"        Training:      {len(train_files)} samples (20 per env)")
    print(f"        Evaluation:    {len(eval_files)} samples (10 test + 20 train per env)")
    
    if args.skip_training:
        print("\n[Skip] --skip-training set. Data is downloaded and staged. Exiting before model phase.")
        return 0

    # ── Model phase ───────────────────────────────────────────────────────────
    out_dir_arch = PROJECT_ROOT / "outputs" / f"real_mgnify_{args.arch}"
    print(f"\n[Train] {len(train_files)} files, [Eval] {len(eval_files)} files")
    print(f"[Config] device={device}  epochs={args.epochs}  adapt_steps={args.adapt_steps} arch={args.arch} kl={args.kl_weight}")
    
    run_model(
        train_files=train_files, 
        eval_files=eval_files, 
        out_dir=out_dir_arch, 
        device=device, 
        epochs=args.epochs, 
        adapt_steps=args.adapt_steps, 
        arch=args.arch, 
        kl_weight=args.kl_weight
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    summary_csv = out_dir_arch / "samples_summary.csv"
    pca_csv     = out_dir_arch / "samples_pca2.csv"
    print(f"\n{'='*60}")
    print("DONE — outputs in:", out_dir_arch)
    print(f"{'='*60}")
    if summary_csv.exists():
        print("\nsamples_summary.csv (perplexity + delta by sample):")
        with open(summary_csv) as f:
            for line in f:
                print(" ", line.rstrip())
    print(f"\nPCA coordinates: {pca_csv}")
    print("\nNEXT STEP: open samples_pca2.csv and plot PC1 vs PC2,")
    print("  colouring points by environment — marine and freshwater")
    print("  should form distinct clusters if the model learned real signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
