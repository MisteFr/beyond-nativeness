#!/usr/bin/env python3
"""Download post-cutoff non-viral proteins from UniProtKB Swiss-Prot.

Query: reviewed:true AND NOT taxonomy_id:10239
       AND date_created:[2025-01-01 TO *] AND length:[50 TO 1022]

These are high-quality, reviewed proteins added to Swiss-Prot after Jan 2025,
guaranteed absent from ESMC (released Dec 2024) and all earlier model training sets.

Outputs:
  data/postcutoff_nonviral.faa.gz       — FASTA sequences
  data/postcutoff_nonviral.tsv.gz       — metadata TSV (with date_created)
  data/postcutoff_nonviral_filtered.faa — filtered, optionally subsampled
"""

import argparse
import gzip
import re
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path

import numpy as np
import requests

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
QUERY = (
    "reviewed:true AND NOT taxonomy_id:10239 "
    "AND date_created:[2025-01-01 TO *] "
    "AND length:[50 TO 1022]"
)
TSV_FIELDS = (
    "accession,protein_name,organism_name,organism_id,lineage,"
    "length,protein_existence,date_created"
)
PAGE_SIZE = 500
RETRY_WAIT = 10
MAX_RETRIES = 5
NEXT_LINK_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MAX_NONSTANDARD_FRAC = 0.05
TARGET_N = 5200  # match existing class sizes


def build_url(base: str, params: dict, fields: str | None = None) -> str:
    """Build URL with literal commas in fields (UniProt requires this)."""
    encoded = urllib.parse.urlencode(params)
    if fields:
        encoded += "&fields=" + urllib.parse.quote(fields, safe=",")
    return base + "?" + encoded


def fetch_page(url: str) -> requests.Response:
    """GET with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  Request failed ({e}), retrying in {RETRY_WAIT}s...", flush=True)
            time.sleep(RETRY_WAIT)
    raise RuntimeError("Unreachable")


def next_url_from_link(link_header: str, base_params: dict, fields: str | None) -> str | None:
    """Extract cursor from Link header and rebuild full URL."""
    match = NEXT_LINK_RE.search(link_header)
    if not match:
        return None
    raw = match.group(1)
    query_str = raw.split("?", 1)[-1]
    qs = urllib.parse.parse_qs(query_str)
    cursors = qs.get("cursor", [])
    if not cursors:
        return None
    return build_url(BASE_URL, {**base_params, "cursor": cursors[0]}, fields)


def download_format(out_path: Path, fmt: str, query: str, fields: str | None = None) -> int:
    """Paginate through UniProt results and write gzipped output. Returns record count."""
    params: dict = {"query": query, "format": fmt, "size": PAGE_SIZE}
    count = 0
    page = 1
    url: str | None = build_url(BASE_URL, params, fields)

    with gzip.open(out_path, "wt") as fh:
        while url:
            print(f"  Fetching page {page} ({fmt})...", flush=True)
            resp = fetch_page(url)

            if page == 1:
                total = resp.headers.get("X-Total-Results", "?")
                print(f"  Total records reported by server: {total}", flush=True)

            text = resp.text
            fh.write(text)

            if fmt == "fasta":
                count += text.count("\n>") + (1 if text.startswith(">") else 0)
            elif fmt == "tsv":
                lines = text.splitlines()
                count += len(lines) - (1 if page == 1 else 0) - lines.count("")

            link_header = resp.headers.get("Link", "")
            url = next_url_from_link(link_header, params, fields)
            page += 1

    return count


def parse_fasta_gz(path: Path) -> list[tuple[str, str]]:
    """Parse gzipped FASTA, return (accession, sequence) pairs."""
    records = []
    header, parts = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header.split()[0], "".join(parts)))
                header = line[1:]
                parts = []
            else:
                parts.append(line.upper())
    if header is not None:
        records.append((header.split()[0], "".join(parts)))
    return records


def filter_and_write(records: list[tuple[str, str]], out_path: Path,
                     target_n: int = TARGET_N, seed: int = 42) -> int:
    """Filter by AA quality, subsample if needed, write plain FASTA. Returns count."""
    filtered = []
    for acc, seq in records:
        n_nonstandard = sum(1 for c in seq if c not in STANDARD_AA)
        if n_nonstandard / max(len(seq), 1) <= MAX_NONSTANDARD_FRAC:
            filtered.append((acc, seq))

    print(f"  After AA quality filter: {len(filtered):,} / {len(records):,}")

    # Subsample if we have more than target
    if len(filtered) > target_n:
        rng = np.random.default_rng(seed)
        # Length-stratified sampling: 10 decile bins
        lengths = np.array([len(s) for _, s in filtered])
        bins = np.percentile(lengths, np.arange(0, 101, 10))
        bin_idx = np.digitize(lengths, bins[1:-1])

        chosen_indices = []
        for b in range(10):
            members = np.where(bin_idx == b)[0]
            n_pick = max(1, round(target_n * len(members) / len(filtered)))
            if len(members) <= n_pick:
                chosen_indices.extend(members)
            else:
                chosen_indices.extend(rng.choice(members, size=n_pick, replace=False))

        # Trim to exact target if needed
        chosen_indices = np.array(chosen_indices)
        if len(chosen_indices) > target_n:
            chosen_indices = rng.choice(chosen_indices, size=target_n, replace=False)

        chosen_indices.sort()
        filtered = [filtered[i] for i in chosen_indices]
        print(f"  After length-stratified subsampling: {len(filtered):,}")

    with open(out_path, "w") as fh:
        for acc, seq in filtered:
            fh.write(f">{acc}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i+60] + "\n")

    return len(filtered)


def print_stats(records: list[tuple[str, str]], tsv_path: Path):
    """Print summary statistics."""
    lengths = [len(s) for _, s in records]
    print(f"\n=== Download Summary ===")
    print(f"Total sequences: {len(records):,}")
    print(f"Length: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    # Parse TSV for taxonomy/date info
    try:
        import csv
        dates = []
        lineages = []
        with gzip.open(tsv_path, "rt") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                if "Date of creation" in row:
                    dates.append(row["Date of creation"])
                elif "date_created" in row:
                    dates.append(row["date_created"])
                lineage = row.get("Taxonomic lineage", "") or row.get("lineage", "")
                if lineage:
                    # Extract top-level kingdom/domain
                    parts = [p.strip() for p in lineage.split(",")]
                    if parts:
                        lineages.append(parts[0])

        if dates:
            dates_clean = sorted(d for d in dates if d)
            print(f"Date range: {dates_clean[0]} to {dates_clean[-1]}")

        if lineages:
            top_lineages = Counter(lineages).most_common(10)
            print(f"\nTop taxonomic groups:")
            for name, count in top_lineages:
                print(f"  {name}: {count:,} ({100*count/len(lineages):.1f}%)")
    except Exception as e:
        print(f"  (Could not parse TSV for stats: {e})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default=str(Path(__file__).parents[1] / "data"),
                        help="Output directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist")
    parser.add_argument("--target_n", type=int, default=TARGET_N,
                        help=f"Target number of sequences after subsampling (default: {TARGET_N})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faa_gz = out_dir / "postcutoff_nonviral.faa.gz"
    tsv_gz = out_dir / "postcutoff_nonviral.tsv.gz"
    faa_filtered = out_dir / "postcutoff_nonviral_filtered.faa"

    # --- Download FASTA ---
    if faa_gz.exists() and not args.force:
        print(f"[skip] {faa_gz} already exists (use --force to re-download)")
    else:
        print(f"Downloading FASTA -> {faa_gz}")
        print(f"Query: {QUERY}")
        n = download_format(faa_gz, fmt="fasta", query=QUERY)
        print(f"  Wrote {n} sequences to {faa_gz}")

    # --- Download TSV ---
    if tsv_gz.exists() and not args.force:
        print(f"[skip] {tsv_gz} already exists (use --force to re-download)")
    else:
        print(f"Downloading TSV metadata -> {tsv_gz}")
        n = download_format(tsv_gz, fmt="tsv", query=QUERY, fields=TSV_FIELDS)
        print(f"  Wrote ~{n} rows to {tsv_gz}")

    # --- Filter and subsample ---
    print(f"\nParsing {faa_gz} ...")
    records = parse_fasta_gz(faa_gz)
    print(f"  Parsed {len(records):,} sequences")

    print_stats(records, tsv_gz)

    print(f"\nFiltering and subsampling -> {faa_filtered}")
    n_final = filter_and_write(records, faa_filtered, target_n=args.target_n, seed=args.seed)
    print(f"  Final output: {n_final:,} sequences in {faa_filtered}")

    print("\nDone.")


if __name__ == "__main__":
    main()
