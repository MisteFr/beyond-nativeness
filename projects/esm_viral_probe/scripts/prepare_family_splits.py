#!/usr/bin/env python3
"""
prepare_family_splits.py
========================
Map viral accessions in the Human Virus dataset to their NCBI taxonomic
family using the NCBI taxonomy dump files.

Output: family_metadata.tsv with columns: accession, family

Usage:
  python prepare_family_splits.py \\
      --source_meta /path/to/human_virus_clean.tsv \\
      --taxdir      data/taxonomy \\
      --out         datasets/human_virus/data/controls/leave_family_out/family_metadata.tsv
"""

import argparse
import os
import sys
from collections import Counter


# ---------------------------------------------------------------------------
# Taxonomy loading
# ---------------------------------------------------------------------------

def load_nodes(nodes_path: str):
    """
    Return (parent, rank, max_taxid).
    parent[tid] = parent tid
    rank[tid]   = rank string (e.g. "family", "genus", "species")
    nodes.dmp columns: taxid | parent | rank | embl_code | division_id | ...
    """
    max_taxid = 0
    with open(nodes_path) as fh:
        for line in fh:
            tid = int(line.split("|", 1)[0].strip())
            if tid > max_taxid:
                max_taxid = tid

    parent = [0] * (max_taxid + 1)
    rank   = [""] * (max_taxid + 1)

    with open(nodes_path) as fh:
        for line in fh:
            parts = line.split("|")
            tid = int(parts[0].strip())
            parent[tid] = int(parts[1].strip())
            rank[tid]   = parts[2].strip()

    return parent, rank, max_taxid


def load_sci_names(names_path: str, max_taxid: int) -> list:
    """
    Return sci_name[taxid] = scientific name string (empty string if absent).
    names.dmp columns: taxid | name_txt | unique_name | name_class
    Only "scientific name" entries are loaded.
    """
    sci_name = [""] * (max_taxid + 1)
    with open(names_path) as fh:
        for line in fh:
            parts = line.split("|")
            if parts[3].strip() != "scientific name":
                continue
            tid = int(parts[0].strip())
            if tid <= max_taxid:
                sci_name[tid] = parts[1].strip()
    return sci_name


def taxid_to_family(taxid: int, parent: list, rank: list,
                    sci_name: list, max_taxid: int) -> str:
    """Walk up NCBI taxonomy tree from taxid until rank == 'family'."""
    seen = set()
    curr = taxid
    while 0 < curr <= max_taxid and curr not in seen:
        if rank[curr] == "family":
            return sci_name[curr] if curr < len(sci_name) else "Unknown"
        seen.add(curr)
        curr = parent[curr]
    return "Unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source_meta", required=True,
                        help="human_virus_clean.tsv (accession, source, protein_name, "
                             "organism, taxid, ...)")
    parser.add_argument("--taxdir", required=True,
                        help="Directory containing nodes.dmp and names.dmp")
    parser.add_argument("--out", required=True,
                        help="Output TSV path (accession, family)")
    args = parser.parse_args()

    nodes_path = os.path.join(args.taxdir, "nodes.dmp")
    names_path = os.path.join(args.taxdir, "names.dmp")

    for p in [args.source_meta, nodes_path, names_path]:
        if not os.path.exists(p):
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # ---- Load taxonomy ----
    print("Loading nodes.dmp ...")
    parent, rank, max_taxid = load_nodes(nodes_path)
    print(f"  max_taxid = {max_taxid:,}")

    print("Loading names.dmp ...")
    sci_name = load_sci_names(names_path, max_taxid)

    # ---- Load viral source metadata ----
    print(f"Loading {args.source_meta} ...")
    # Columns: accession, source, protein_name, organism, taxid, length, ...
    records: list[tuple[str, int]] = []
    with open(args.source_meta) as fh:
        header = next(fh)
        cols = header.rstrip("\n").split("\t")
        acc_idx  = cols.index("accession")
        taxid_idx = cols.index("taxid")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            acc   = parts[acc_idx]
            try:
                tid = int(parts[taxid_idx])
            except (ValueError, IndexError):
                tid = 0
            records.append((acc, tid))
    print(f"  Loaded {len(records):,} records")

    # ---- Map taxid → family ----
    print("Walking taxonomy tree to assign families ...")
    results: list[tuple[str, str]] = []
    for acc, tid in records:
        fam = taxid_to_family(tid, parent, rank, sci_name, max_taxid)
        results.append((acc, fam))

    # ---- Summary ----
    family_counts = Counter(fam for _, fam in results)
    n_unknown = family_counts.get("Unknown", 0)
    print(f"\nFamily assignment summary ({len(results):,} sequences):")
    print(f"  Unknown: {n_unknown:,} ({100*n_unknown/len(results):.1f}%)")
    print(f"  Families found: {len(family_counts) - (1 if 'Unknown' in family_counts else 0)}")
    for fam, cnt in family_counts.most_common(20):
        if fam != "Unknown":
            print(f"    {fam:<35}  {cnt:>5}")

    # ---- Write output ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("accession\tfamily\n")
        for acc, fam in results:
            fh.write(f"{acc}\t{fam}\n")
    print(f"\nWrote {len(results):,} rows to {args.out}")


if __name__ == "__main__":
    main()
