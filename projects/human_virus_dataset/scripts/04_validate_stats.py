"""Validate and report statistics for both protein sources.

Reports per-source (UniProt and NCBI NP_):
  - Total protein count
  - Existence level distribution (UniProt only)
  - Protein length distribution (min, P5, median, P95, max)
  - Top 20 viral families/genera by protein count (from lineage/organism fields)
  - Fraction of sequences with >5% non-standard amino acids

Writes to stdout and to data/stats_report.txt.

Usage:
  python 04_validate_stats.py
"""

import argparse
import csv
import gzip
import sys
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent))
from utils import iter_fasta, accession_from_header

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
SPOT_CHECK_ACCESSIONS = {
    # UniProt accessions for well-known human-virus proteins
    "P04585",   # HIV-1 Gag
    "P03452",   # Influenza A HA
    "P0DTC2",   # SARS-CoV-2 Spike
}
SPOT_CHECK_REFSEQ = {
    # RefSeq NP_ accessions — only accessions present in the manual NCBI export
    "NP_057849",   # HIV-1 Gag (present in user's NCBI download)
}


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = p / 100 * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def nonstandard_fraction(seq: str) -> float:
    return sum(1 for aa in seq if aa not in STANDARD_AA) / max(len(seq), 1)


def analyze_fasta(fasta_path: Path, label: str) -> dict:
    """Compute length stats and non-standard AA fraction for a FASTA file."""
    lengths = []
    n_nonstandard_flagged = 0
    for _header, seq in iter_fasta(fasta_path):
        lengths.append(len(seq))
        if nonstandard_fraction(seq) > 0.05:
            n_nonstandard_flagged += 1
    lengths.sort()
    return {
        "label": label,
        "count": len(lengths),
        "length_min": lengths[0] if lengths else 0,
        "length_p5": percentile(lengths, 5),
        "length_median": percentile(lengths, 50),
        "length_p95": percentile(lengths, 95),
        "length_max": lengths[-1] if lengths else 0,
        "n_nonstandard_flagged": n_nonstandard_flagged,
    }


def parse_uniprot_lineage(lineage: str) -> str:
    """Extract the viral family from a UniProt lineage string.

    UniProt lineage is a semicolon-separated taxonomy string like:
      Viruses; Duplodnaviria; ...; Herpesviridae; Alphaherpesvirinae; ...
    We return the first token that ends with 'viridae' or 'virinae',
    falling back to the last token.
    """
    parts = [p.strip() for p in lineage.split(";") if p.strip()]
    for part in parts:
        if part.endswith("viridae") or part.endswith("virinae"):
            return part
    return parts[-1] if parts else "Unknown"


def taxonomy_from_uniprot_tsv(tsv_path: Path) -> tuple[Counter, Counter]:
    """Return (existence_counter, family_counter) from UniProt TSV."""
    existence_counts: Counter = Counter()
    family_counts: Counter = Counter()
    with gzip.open(tsv_path, "rt") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            existence_counts[row.get("Protein existence", row.get("existence", "?"))] += 1
            lineage = row.get("Taxonomic lineage", row.get("lineage", ""))
            family_counts[parse_uniprot_lineage(lineage)] += 1
    return existence_counts, family_counts


def organism_from_ncbi_tsv(tsv_path: Path) -> Counter:
    """Return organism counter from NCBI metadata TSV."""
    counter: Counter = Counter()
    with open(tsv_path, "r") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            org = row.get("organism", "Unknown")
            # Use genus-level: first two words of organism name
            parts = org.split()
            genus = " ".join(parts[:2]) if len(parts) >= 2 else org
            counter[genus] += 1
    return counter


def spot_check_uniprot(fasta_path: Path) -> set[str]:
    """Return which spot-check accessions are present in UniProt FASTA."""
    found = set()
    for header, _seq in iter_fasta(fasta_path):
        # UniProt headers: sp|ACCESSION|ENTRY_NAME Description OS=...
        # Accession is field [1] when split by '|'
        parts = header.split("|")
        if len(parts) >= 2:
            acc = parts[1]
            if acc in SPOT_CHECK_ACCESSIONS:
                found.add(acc)
    return found


def spot_check_ncbi(fasta_path: Path) -> set[str]:
    """Return which spot-check NP_ accessions are present in NCBI FASTA."""
    found = set()
    for header, _seq in iter_fasta(fasta_path):
        acc = accession_from_header(header).split(".")[0]
        if acc in SPOT_CHECK_REFSEQ:
            found.add(acc)
    return found


def fmt_report(stats: dict, existence: Counter | None, top_taxa: Counter,
               taxa_label: str) -> str:
    lines = []
    lines.append(f"=== {stats['label']} ===")
    lines.append(f"Total proteins         : {stats['count']:,}")
    lines.append(f"Length min / P5        : {stats['length_min']} / {stats['length_p5']:.0f}")
    lines.append(f"Length median          : {stats['length_median']:.0f}")
    lines.append(f"Length P95 / max       : {stats['length_p95']:.0f} / {stats['length_max']}")
    lines.append(f">5% non-standard AA    : {stats['n_nonstandard_flagged']:,} "
                 f"({100*stats['n_nonstandard_flagged']/max(stats['count'],1):.2f}%)")
    if existence is not None:
        lines.append(f"\nProtein existence levels:")
        for level, cnt in sorted(existence.items()):
            lines.append(f"  {level:<50} {cnt:>6,}")
    lines.append(f"\nTop 20 {taxa_label}:")
    for name, cnt in top_taxa.most_common(20):
        lines.append(f"  {name:<55} {cnt:>6,}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uniprot_faa",
        default=str(Path(__file__).parents[1] / "data" / "raw" / "uniprot" / "human_virus.faa.gz"),
    )
    parser.add_argument(
        "--uniprot_tsv",
        default=str(Path(__file__).parents[1] / "data" / "raw" / "uniprot" / "human_virus.tsv.gz"),
    )
    parser.add_argument(
        "--ncbi_faa",
        default=str(Path(__file__).parents[1] / "data" / "processed" / "ncbi_human_np.faa"),
    )
    parser.add_argument(
        "--ncbi_tsv",
        default=str(Path(__file__).parents[1] / "data" / "processed" / "ncbi_human_np.tsv"),
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parents[1] / "data" / "stats_report.txt"),
    )
    args = parser.parse_args()

    uniprot_faa = Path(args.uniprot_faa)
    uniprot_tsv = Path(args.uniprot_tsv)
    ncbi_faa = Path(args.ncbi_faa)
    ncbi_tsv = Path(args.ncbi_tsv)
    out_path = Path(args.out)

    sections = []

    # --- UniProt ---
    if uniprot_faa.exists():
        print("Analyzing UniProt FASTA...", flush=True)
        u_stats = analyze_fasta(uniprot_faa, "UniProt Swiss-Prot (reviewed, human-virus, PE 1-3)")
        u_existence: Counter | None = None
        u_families: Counter = Counter()
        if uniprot_tsv.exists():
            u_existence, u_families = taxonomy_from_uniprot_tsv(uniprot_tsv)
        sections.append(fmt_report(u_stats, u_existence, u_families, "viral families (UniProt lineage)"))

        print("  Spot-checking known proteins...", flush=True)
        found = spot_check_uniprot(uniprot_faa)
        missing = SPOT_CHECK_ACCESSIONS - found
        sections.append(
            f"Spot-check UniProt accessions:\n"
            + "\n".join(f"  {'✓' if a in found else '✗'} {a}" for a in sorted(SPOT_CHECK_ACCESSIONS))
            + (f"\n  WARNING: missing {missing}" if missing else "")
        )
    else:
        sections.append(f"[SKIP] UniProt FASTA not found: {uniprot_faa}")

    sections.append("")

    # --- NCBI NP_ ---
    if ncbi_faa.exists():
        print("Analyzing NCBI NP_ FASTA...", flush=True)
        n_stats = analyze_fasta(ncbi_faa, "NCBI RefSeq NP_ (human-virus, manually curated)")
        n_orgs: Counter = Counter()
        if ncbi_tsv.exists():
            n_orgs = organism_from_ncbi_tsv(ncbi_tsv)
        sections.append(fmt_report(n_stats, None, n_orgs, "organisms (NCBI, genus level)"))

        print("  Spot-checking known proteins...", flush=True)
        found = spot_check_ncbi(ncbi_faa)
        missing = SPOT_CHECK_REFSEQ - found
        sections.append(
            f"Spot-check NCBI NP_ accessions:\n"
            + "\n".join(f"  {'✓' if a in found else '✗'} {a}" for a in sorted(SPOT_CHECK_REFSEQ))
            + (f"\n  WARNING: missing {missing}" if missing else "")
        )
    else:
        sections.append(f"[SKIP] NCBI NP_ FASTA not found: {ncbi_faa}")

    report = "\n".join(sections)
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
