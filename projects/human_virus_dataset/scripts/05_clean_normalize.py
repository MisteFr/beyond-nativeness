"""Merge, clean, and normalize human-virus proteins from UniProt and NCBI NP_.

Inputs:
  data/raw/uniprot/human_virus.faa.gz    — UniProt Swiss-Prot FASTA (6,044 seqs)
  data/raw/uniprot/human_virus.tsv.gz    — UniProt metadata (Entry, Protein names, Organism, Organism (ID), ...)
  data/processed/ncbi_human_np.faa       — NCBI RefSeq NP_ FASTA (620 seqs)
  data/processed/ncbi_human_np.tsv       — NCBI NP_ metadata (accession, title, organism, taxid, length)

Outputs:
  data/processed/human_virus_clean.faa   — merged, normalized, deduplicated FASTA
  data/processed/human_virus_clean.tsv   — unified metadata TSV
  Appends a stats block to data/stats_report.txt

Deduplication: exact sequence hash (SHA-256). UniProt records take precedence over
NCBI NP_ when the same sequence appears in both sources.

Length filtering is NOT applied here; downstream tools (e.g. ESM probe pipeline)
apply their own cutoffs at embedding time. The `in_length_range` column (50–1022 AA)
is provided as a convenience flag.
"""

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import iter_fasta, write_fasta

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
ESM_MIN_LEN = 50
ESM_MAX_LEN = 1022


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_uniprot_header(header: str) -> tuple[str, str, str]:
    """Return (accession, protein_name, organism) from a UniProt FASTA header.

    Header format: sp|ENTRY_ID|ENTRY_NAME description OS=Organism OX=taxid ...
    """
    # Extract accession: second pipe-delimited field
    acc = header.split("|")[1] if "|" in header else header.split()[0]

    # Protein name: everything between the third pipe-field end and "OS="
    rest = header.split("|", 2)[2] if header.count("|") >= 2 else header
    # rest starts with "ENTRY_NAME description OS=..."
    # drop the entry name token, keep the description up to OS=
    tokens = rest.split(None, 1)
    desc_and_rest = tokens[1] if len(tokens) > 1 else ""
    os_match = re.search(r"\s+OS=", desc_and_rest)
    protein_name = desc_and_rest[:os_match.start()].strip() if os_match else desc_and_rest.strip()

    # Organism: between OS= and OX= (or end of line)
    organism = ""
    os_m = re.search(r"OS=(.+?)(?:\s+OX=|$)", header)
    if os_m:
        organism = os_m.group(1).strip()

    return acc, protein_name, organism


def _load_uniprot_tsv(tsv_path: Path) -> dict[str, dict]:
    """Load UniProt TSV into a dict keyed by Entry (accession).

    Returns {entry_id: {taxid, protein_existence}} for easy joining.
    """
    import gzip
    meta: dict[str, dict] = {}
    opener = gzip.open(tsv_path, "rt") if tsv_path.suffix == ".gz" else open(tsv_path)
    with opener as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            meta[row["Entry"]] = {
                "taxid": row.get("Organism (ID)", "").strip(),
                "protein_existence": row.get("Protein existence", "").strip(),
            }
    return meta


def _parse_ncbi_header(header: str) -> tuple[str, str, str]:
    """Return (accession, title, organism) from an NCBI FASTA header.

    Header format: NP_XXXXX.1 |title [Organism name]
    """
    acc = header.split()[0]
    # title: after the first `|` up to `[`
    title = ""
    organism = ""
    pipe_match = re.search(r"\|(.+?)(?:\s*\[|$)", header)
    if pipe_match:
        title = pipe_match.group(1).strip()
    bracket_match = re.search(r"\[(.+?)\]", header)
    if bracket_match:
        organism = bracket_match.group(1).strip()
    return acc, title, organism


def _load_ncbi_tsv(tsv_path: Path) -> dict[str, dict]:
    """Load NCBI NP_ TSV into a dict keyed by accession."""
    meta: dict[str, dict] = {}
    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            meta[row["accession"]] = {
                "title": row.get("title", "").strip(),
                "organism": row.get("organism", "").strip(),
                "taxid": row.get("taxid", "").strip(),
            }
    return meta


# ---------------------------------------------------------------------------
# Sequence normalization
# ---------------------------------------------------------------------------

def normalize_seq(seq: str) -> str:
    """Uppercase and strip gaps/whitespace from a sequence."""
    return seq.upper().replace("-", "").replace(" ", "")


def pct_nonstandard(seq: str) -> float:
    """Fraction of characters not in the standard 20-AA alphabet."""
    if not seq:
        return 0.0
    return sum(1 for c in seq if c not in STANDARD_AA) / len(seq)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parents[1]
    parser.add_argument("--uniprot_faa",   default=str(root / "data/raw/uniprot/human_virus.faa.gz"))
    parser.add_argument("--uniprot_tsv",   default=str(root / "data/raw/uniprot/human_virus.tsv.gz"))
    parser.add_argument("--ncbi_faa",      default=str(root / "data/processed/ncbi_human_np.faa"))
    parser.add_argument("--ncbi_tsv",      default=str(root / "data/processed/ncbi_human_np.tsv"))
    parser.add_argument("--out_faa",       default=str(root / "data/processed/human_virus_clean.faa"))
    parser.add_argument("--out_tsv",       default=str(root / "data/processed/human_virus_clean.tsv"))
    parser.add_argument("--stats_report",  default=str(root / "data/stats_report.txt"))
    args = parser.parse_args()

    out_faa  = Path(args.out_faa)
    out_tsv  = Path(args.out_tsv)
    out_faa.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load UniProt records (source = "uniprot")
    # ------------------------------------------------------------------
    print("Loading UniProt sequences...", flush=True)
    uniprot_meta = _load_uniprot_tsv(Path(args.uniprot_tsv))

    # List of (accession, source, protein_name, organism, taxid, norm_seq, pct_ns)
    records: list[dict] = []

    for header, raw_seq in iter_fasta(args.uniprot_faa):
        acc, protein_name, organism = _parse_uniprot_header(header)
        seq = normalize_seq(raw_seq)
        extra = uniprot_meta.get(acc, {})
        taxid = extra.get("taxid", "")
        records.append({
            "accession":    acc,
            "source":       "uniprot",
            "protein_name": protein_name,
            "organism":     organism,
            "taxid":        taxid,
            "seq":          seq,
            "pct_ns":       pct_nonstandard(seq),
        })

    n_uniprot_in = len(records)
    print(f"  Loaded {n_uniprot_in:,} UniProt sequences")

    # ------------------------------------------------------------------
    # 2. Load NCBI NP_ records (source = "ncbi_np")
    # ------------------------------------------------------------------
    print("Loading NCBI NP_ sequences...", flush=True)
    ncbi_meta = _load_ncbi_tsv(Path(args.ncbi_tsv))

    ncbi_records: list[dict] = []
    for header, raw_seq in iter_fasta(args.ncbi_faa):
        acc_from_header, title_from_header, org_from_header = _parse_ncbi_header(header)
        seq = normalize_seq(raw_seq)
        extra = ncbi_meta.get(acc_from_header, {})
        taxid   = extra.get("taxid", "")
        title   = extra.get("title", "") or title_from_header
        organism = extra.get("organism", "") or org_from_header
        ncbi_records.append({
            "accession":    acc_from_header,
            "source":       "ncbi_np",
            "protein_name": title,
            "organism":     organism,
            "taxid":        taxid,
            "seq":          seq,
            "pct_ns":       pct_nonstandard(seq),
        })

    n_ncbi_in = len(ncbi_records)
    print(f"  Loaded {n_ncbi_in:,} NCBI NP_ sequences")

    # ------------------------------------------------------------------
    # 3. Merge: UniProt first (takes precedence in dedup), then NCBI NP_
    # ------------------------------------------------------------------
    all_records = records + ncbi_records
    n_total_in = len(all_records)

    # ------------------------------------------------------------------
    # 4. Exact deduplication by SHA-256 of normalized sequence
    # ------------------------------------------------------------------
    print("Deduplicating (exact sequence hash)...", flush=True)
    seen_hashes: dict[str, str] = {}   # hash → accession of first occurrence
    deduped: list[dict] = []
    n_dup_within_uniprot = 0
    n_dup_within_ncbi    = 0
    n_dup_cross          = 0

    for rec in all_records:
        h = hashlib.sha256(rec["seq"].encode()).hexdigest()
        if h in seen_hashes:
            # Classify duplicate type
            first_acc = seen_hashes[h]
            first_source = next(r["source"] for r in deduped if r["accession"] == first_acc)
            if rec["source"] == "uniprot" and first_source == "uniprot":
                n_dup_within_uniprot += 1
            elif rec["source"] == "ncbi_np" and first_source == "ncbi_np":
                n_dup_within_ncbi += 1
            else:
                n_dup_cross += 1
        else:
            seen_hashes[h] = rec["accession"]
            deduped.append(rec)

    n_removed = n_total_in - len(deduped)
    print(f"  Removed {n_removed:,} exact duplicates "
          f"({n_dup_within_uniprot} within-UniProt, "
          f"{n_dup_within_ncbi} within-NCBI, "
          f"{n_dup_cross} cross-source)")
    print(f"  Retained {len(deduped):,} unique sequences")

    # ------------------------------------------------------------------
    # 5. Write FASTA
    # ------------------------------------------------------------------
    print(f"Writing FASTA to {out_faa}...", flush=True)

    def fasta_records():
        for rec in deduped:
            header = f"{rec['accession']} {rec['source']}|{rec['protein_name']} [{rec['organism']}]"
            yield header, rec["seq"]

    n_written = write_fasta(fasta_records(), out_faa)
    print(f"  Wrote {n_written:,} sequences")

    # ------------------------------------------------------------------
    # 6. Write TSV
    # ------------------------------------------------------------------
    print(f"Writing TSV to {out_tsv}...", flush=True)
    tsv_fields = [
        "accession", "source", "protein_name", "organism", "taxid",
        "length", "pct_nonstandard", "in_length_range",
    ]
    with open(out_tsv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=tsv_fields, delimiter="\t")
        writer.writeheader()
        for rec in deduped:
            length = len(rec["seq"])
            writer.writerow({
                "accession":      rec["accession"],
                "source":         rec["source"],
                "protein_name":   rec["protein_name"],
                "organism":       rec["organism"],
                "taxid":          rec["taxid"],
                "length":         length,
                "pct_nonstandard": f"{rec['pct_ns']:.4f}",
                "in_length_range": str(ESM_MIN_LEN <= length <= ESM_MAX_LEN),
            })
    print(f"  Wrote {len(deduped):,} rows")

    # ------------------------------------------------------------------
    # 7. Stats report
    # ------------------------------------------------------------------
    lengths = [len(r["seq"]) for r in deduped]
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    def percentile(p: float) -> int:
        idx = int(p / 100 * n)
        return lengths_sorted[min(idx, n - 1)]

    n_nonstandard_any  = sum(1 for r in deduped if r["pct_ns"] > 0)
    n_nonstandard_5pct = sum(1 for r in deduped if r["pct_ns"] >= 0.05)
    n_in_range = sum(1 for r in deduped if ESM_MIN_LEN <= len(r["seq"]) <= ESM_MAX_LEN)

    source_counts = Counter(r["source"] for r in deduped)
    organism_counts = Counter(r["organism"] for r in deduped)
    top_organisms = organism_counts.most_common(10)

    lines = [
        "",
        "=" * 70,
        "CLEANING & NORMALIZATION STATS (05_clean_normalize.py)",
        "=" * 70,
        f"Input:  {n_uniprot_in:>6,} UniProt sequences",
        f"        {n_ncbi_in:>6,} NCBI NP_ sequences",
        f"        {n_total_in:>6,} total before deduplication",
        "",
        f"Deduplication (exact SHA-256):",
        f"  Removed: {n_removed:,} total",
        f"    Within-UniProt:  {n_dup_within_uniprot}",
        f"    Within-NCBI NP_: {n_dup_within_ncbi}",
        f"    Cross-source:    {n_dup_cross}",
        "",
        f"Final dataset: {len(deduped):,} unique sequences",
        f"  UniProt:  {source_counts['uniprot']:,}",
        f"  NCBI NP_: {source_counts['ncbi_np']:,}",
        "",
        "Length distribution:",
        f"  Min:    {lengths_sorted[0]:>6}",
        f"  P5:     {percentile(5):>6}",
        f"  Median: {percentile(50):>6}",
        f"  P95:    {percentile(95):>6}",
        f"  Max:    {lengths_sorted[-1]:>6}",
        "",
        f"Non-standard AA:",
        f"  Sequences with any non-standard AA:    {n_nonstandard_any:,}",
        f"  Sequences with ≥5% non-standard AA:   {n_nonstandard_5pct:,}",
        "",
        f"In ESM length range ({ESM_MIN_LEN}–{ESM_MAX_LEN} AA): "
        f"{n_in_range:,} / {len(deduped):,} ({100*n_in_range/max(len(deduped),1):.1f}%)",
        "",
        "Top 10 organisms:",
    ]
    for org, cnt in top_organisms:
        lines.append(f"  {cnt:>5,}  {org}")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    with open(args.stats_report, "a") as fh:
        fh.write(report + "\n")
    print(f"Stats appended to {args.stats_report}")
    print("Done.")


if __name__ == "__main__":
    main()
