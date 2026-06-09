"""Shared FASTA I/O utilities."""

import gzip
from pathlib import Path
from typing import Iterator


def open_fasta(path: str | Path):
    """Open a FASTA file, transparently handling gzip."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")


def iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (header, sequence) pairs from a FASTA file (plain or gzipped)."""
    header = None
    parts: list[str] = []
    with open_fasta(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header = line[1:]
                parts = []
            else:
                parts.append(line.upper())
    if header is not None:
        yield header, "".join(parts)


def write_fasta(records: Iterator[tuple[str, str]], path: str | Path, wrap: int = 60) -> int:
    """Write (header, sequence) pairs to a FASTA file. Returns count written."""
    path = Path(path)
    count = 0
    opener = gzip.open(path, "wt") if path.suffix == ".gz" else open(path, "w")
    with opener as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), wrap):
                fh.write(seq[i:i + wrap] + "\n")
            count += 1
    return count


def accession_from_header(header: str) -> str:
    """Extract the first whitespace-delimited token from a FASTA header."""
    return header.split()[0]
