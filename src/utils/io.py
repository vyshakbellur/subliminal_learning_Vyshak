import gzip
from typing import IO, List, Tuple, Any

def _open_text(path: str) -> IO[Any]:
    """Open a file for reading text, handling gzip automatically."""
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
