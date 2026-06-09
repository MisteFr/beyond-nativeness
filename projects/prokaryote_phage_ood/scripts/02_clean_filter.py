#!/usr/bin/env python3
"""Filter, deduplicate, and optionally subsample downloaded proteins.

For phage group: filters the broad virus download to retain only bacteriophage
proteins by checking lineage and organism name.

Usage:
    python scripts/02_clean_filter.py --group bacteria --max_seqs 5000
    python scripts/02_clean_filter.py --group archaea
    python scripts/02_clean_filter.py --group phage
"""

import argparse
import csv
import gzip
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import iter_fasta, write_fasta

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN = 50
MAX_LEN = 1022
MAX_PCT_NS = 0.05

# Lineage keywords for identifying bacteriophage entries
PHAGE_LINEAGE_KEYWORDS = [
    "Caudoviricetes", "Duplodnaviria", "Microviridae", "Inoviridae",
    "Tectiviridae", "Leviviridae", "Cystoviridae", "Corticoviridae",
    "Plasmaviridae", "Caudovirales",
]


def normalize_seq(seq: str) -> str:
    return seq.upper().replace("-", "").replace(" ", "")


def pct_nonstandard(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for c in seq if c not in STANDARD_AA) / len(seq)


def is_phage(lineage: str, organism: str) -> bool:
    """Check if a viral entry is a bacteriophage."""
    lineage_lower = lineage.lower()
    for kw in PHAGE_LINEAGE_KEYWORDS:
        if kw.lower() in lineage_lower:
            return True
    if "phage" in organism.lower():
        return True
    return False


def load_tsv_metadata(tsv_path: Path) -> dict[str, dict]:
    """Load TSV into {accession: {lineage, organism_name, ...}}."""
    meta = {}
    opener = gzip.open(tsv_path, "rt") if tsv_path.suffix == ".gz" else open(tsv_path)
    with opener as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            acc = row.get("Entry", row.get("accession", "")).strip()
            if not acc:
                continue
            meta[acc] = {
                "organism_name": row.get("Organism", row.get("organism_name", "")).strip(),
                "organism_id": row.get("Organism (ID)", row.get("organism_id", "")).strip(),
                "lineage": row.get("Lineage", row.get("lineage", "")).strip(),
                "protein_name": row.get("Protein names", row.get("protein_name", "")).strip(),
                "length": row.get("Length", row.get("length", "")).strip(),
            }
    return meta


def parse_uniprot_accession(header: str) -> str:
    """Extract accession from UniProt FASTA header (sp|ACC|NAME ...)."""
    if "|" in header:
        return header.split("|")[1]
    return header.split()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True,
                        choices=["bacteria", "archaea", "phage",
                                 "plants", "fungi", "insects",
                                 "plant_virus", "invertebrate_virus"])
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--max_seqs", type=int, default=0, help="0 = no limit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faa_path = raw_dir / f"{args.group}.faa.gz"
    tsv_path = raw_dir / f"{args.group}.tsv.gz"
    out_faa = out_dir / f"{args.group}_clean.faa"
    out_tsv = out_dir / f"{args.group}_clean.tsv"

    print(f"Group: {args.group}")
    print(f"Input FASTA: {faa_path}")
    print(f"Input TSV:   {tsv_path}")

    # Load metadata
    meta = load_tsv_metadata(tsv_path)
    print(f"Loaded {len(meta):,} metadata rows")

    # Load and filter sequences
    records = []
    n_raw = 0
    n_too_short = 0
    n_too_long = 0
    n_nonstandard = 0
    n_not_phage = 0

    for header, raw_seq in iter_fasta(faa_path):
        n_raw += 1
        acc = parse_uniprot_accession(header)
        seq = normalize_seq(raw_seq)

        # Length filter
        if len(seq) < MIN_LEN:
            n_too_short += 1
            continue
        if len(seq) > MAX_LEN:
            n_too_long += 1
            continue

        # Non-standard AA filter
        ns = pct_nonstandard(seq)
        if ns >= MAX_PCT_NS:
            n_nonstandard += 1
            continue

        # Phage-specific: filter to bacteriophages only
        if args.group == "phage":
            m = meta.get(acc, {})
            lineage = m.get("lineage", "")
            organism = m.get("organism_name", "")
            if not is_phage(lineage, organism):
                n_not_phage += 1
                continue

        m = meta.get(acc, {})
        records.append({
            "accession": acc,
            "seq": seq,
            "organism_name": m.get("organism_name", ""),
            "organism_id": m.get("organism_id", ""),
            "lineage": m.get("lineage", ""),
            "protein_name": m.get("protein_name", ""),
            "length": len(seq),
            "pct_nonstandard": ns,
        })

    print(f"\nRaw sequences:     {n_raw:>8,}")
    print(f"Too short (<{MIN_LEN}):  {n_too_short:>8,}")
    print(f"Too long (>{MAX_LEN}):  {n_too_long:>8,}")
    print(f"Non-standard AA:   {n_nonstandard:>8,}")
    if args.group == "phage":
        print(f"Not phage:         {n_not_phage:>8,}")
    print(f"After filters:     {len(records):>8,}")

    # Exact dedup by SHA-256
    seen = set()
    deduped = []
    for rec in records:
        h = hashlib.sha256(rec["seq"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(rec)
    n_dups = len(records) - len(deduped)
    print(f"Duplicates removed: {n_dups:>7,}")
    print(f"After dedup:       {len(deduped):>8,}")

    # Subsample if requested
    if args.max_seqs > 0 and len(deduped) > args.max_seqs:
        rng = __import__("numpy").random.default_rng(args.seed)
        indices = rng.choice(len(deduped), size=args.max_seqs, replace=False)
        indices.sort()
        deduped = [deduped[i] for i in indices]
        print(f"Subsampled to:     {len(deduped):>8,}")

    # Write FASTA
    def fasta_records():
        for rec in deduped:
            header = f"{rec['accession']} {rec['protein_name']} [{rec['organism_name']}]"
            yield header, rec["seq"]

    n_written = write_fasta(fasta_records(), out_faa)
    print(f"\nWrote {n_written:,} sequences to {out_faa}")

    # Write TSV
    fields = ["accession", "organism_name", "organism_id", "lineage",
              "protein_name", "length", "pct_nonstandard"]
    with open(out_tsv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for rec in deduped:
            writer.writerow({k: rec[k] for k in fields})
    print(f"Wrote {len(deduped):,} rows to {out_tsv}")

    # Quick organism breakdown
    from collections import Counter
    org_counts = Counter(rec["organism_name"] for rec in deduped)
    print(f"\nTop 10 organisms:")
    for org, cnt in org_counts.most_common(10):
        print(f"  {cnt:>5,}  {org}")

    print("Done.")


if __name__ == "__main__":
    main()
