import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "samples_synth_small"
    out_dir = root / "outputs" / "demo_subliminal"

    # (Re)make synthetic demo dataset
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    subprocess.check_call([
        sys.executable,
        str(root / "src" / "make_synthetic_samples.py"),
        "--outdir", str(data_dir),
        "--seed", "42",
        "--contigs", "2",
        "--contig-len", "5000",
        "--envA", "soil",
        "--envB", "marine",
        "--envOOD", "gut_ood",
        "--nA", "2",
        "--nB", "2",
        "--nOOD", "1",
    ])

    # Train on soil+marine only, evaluate on all (including gut_ood)
    train = [
        str(data_dir / "soil__*.fna.gz"),
        str(data_dir / "marine__*.fna.gz"),
    ]
    eval_ = [str(data_dir / "*.fna.gz")]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(root / "src" / "subliminal_sample_lm.py"),
        "--train-fasta", *train,
        "--eval-fasta", *eval_,
        "--kmer", "31",
        "--stride", "1",
        "--vocab-size", "32768",
        "--d-model", "32",
        "--layers", "1",
        "--heads", "4",
        "--epochs", "1",
        "--train-block", "64",
        "--train-overlap", "0.5",
        "--embed-block", "128",
        "--embed-step", "128",
        "--batch", "32",
        "--adapt-steps", "30",
        "--adapt-lr", "0.2",
        "--save", str(out_dir),
        "--seed", "42",
    ]

    print(" ".join(cmd))
    subprocess.check_call(cmd)

    print("\n[OK] Demo complete.")
    print(f"  Outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
