#!/usr/bin/env python3
"""
preprocess.py
=============
Filter, balance, and split viral/non-viral protein sequences for the ESM2
probing experiment, using homology-cluster-aware splits to prevent data
leakage across train/val/test.

Steps:
  1. Read viral (RefSeq) and non-viral (UniProtKB Swiss-Prot) FASTAs
  2. Filter: drop sequences outside [MIN_LEN, MAX_LEN] or with >5% non-std AA
  3. Subsample to N_SAMPLES per class (stratified by length decile)
  4. Combine all sequences and run MMseqs2 easy-linclust at MIN_SEQ_ID
     identity + COV coverage (bidirectional).  Sequences from the same
     cluster are always kept in the same split — this is the "UniRef-style"
     homology split control that prevents train/test leakage.
  5. Assign clusters randomly to train / val / test targeting 60/20/20 by
     sequence count (seed-controlled).
  6. Write per-split, per-class FASTA files + a metadata TSV.

Usage:
  python preprocess.py \\
      --viral    ../data/viral/refseq_viral.faa \\
      --nonviral ../data/nonviral/uniprot_nonviral.faa \\
      --outdir   ../data/processed \\
      --n_samples 50000 \\
      --seed 42
"""

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_LEN    = 50      # Minimum sequence length (aa)
MAX_LEN    = 1022    # ESM2 max tokens; filter longer seqs before clustering
VALID_AA   = set("ACDEFGHIKLMNPQRSTVWY")  # Standard 20 amino acids
MIN_SEQ_ID = 0.30    # MMseqs2 cluster identity threshold (UniRef30-style)
COV        = 0.80    # Minimum bidirectional coverage for clustering


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

def parse_fasta(path: str) -> Iterator[tuple[str, str]]:
    """Yield (header, sequence) pairs from a FASTA file."""
    header, seq_parts = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.upper())
    if header is not None:
        yield header, "".join(seq_parts)


def write_fasta(records: list[tuple[str, str]], path: str) -> None:
    """Write (header, sequence) pairs to a FASTA file (60-char wrapped)."""
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i+60] + "\n")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def is_valid(seq: str) -> bool:
    """Return True if the sequence passes quality filters."""
    if len(seq) < MIN_LEN or len(seq) > MAX_LEN:
        return False
    non_standard = sum(1 for aa in seq if aa not in VALID_AA)
    return non_standard / len(seq) < 0.05


# ---------------------------------------------------------------------------
# Balanced subsampling (stratified by sequence length decile)
# ---------------------------------------------------------------------------

def balanced_sample(records: list, n: int, rng: random.Random) -> list:
    """
    Subsample to `n` records preserving the length distribution.
    Splits into 10 decile buckets and samples proportionally from each.
    """
    if len(records) <= n:
        return list(records)

    lengths = [len(seq) for _, seq in records]
    sorted_lengths = sorted(lengths)
    decile_boundaries = [
        sorted_lengths[int(len(sorted_lengths) * i / 10)]
        for i in range(1, 10)
    ]

    def get_decile(length):
        for i, boundary in enumerate(decile_boundaries):
            if length <= boundary:
                return i
        return 9

    buckets: dict[int, list] = defaultdict(list)
    for rec in records:
        buckets[get_decile(len(rec[1]))].append(rec)

    sampled = []
    for d, bucket_recs in sorted(buckets.items()):
        k = max(1, int(n * len(bucket_recs) / len(records)))
        sampled.extend(rng.sample(bucket_recs, min(k, len(bucket_recs))))

    rng.shuffle(sampled)
    return sampled[:n]


# ---------------------------------------------------------------------------
# Homology clustering with MMseqs2
# ---------------------------------------------------------------------------

def run_mmseqs2_cluster(records: list[tuple[str, str, str]],
                        tmpdir: str,
                        threads: int,
                        min_seq_id: float,
                        cov: float) -> dict[str, str]:
    """
    Cluster sequences with MMseqs2 easy-linclust.
    
    When you have a large set of protein sequences (e.g., thousands of viral proteins), many will be nearly identical — same protein from closely related strains. If you train or analyze on all of them, you get:

    Biased results (overrepresented sequences dominate)
    Inflated dataset size
    Train/test leakage (near-identical sequences in both splits)
    Clustering solves this by grouping similar sequences together and picking one representative per cluster. Downstream code then typically keeps only the representative (or one per cluster), reducing redundancy while preserving diversity.

    In the ESM2 viral probe context specifically, this is likely used to deduplicate the viral protein dataset before embedding/training, so the model isn't skewed by sequence families that happen to have many nearly-identical members deposited in the database.


    Parameters
    ----------
    records   : list of (safe_id, header, seq) — safe_id has no whitespace
    tmpdir    : scratch directory for MMseqs2 temp files
    threads   : CPU threads to pass to MMseqs2
    min_seq_id: minimum sequence identity (0–1)
    cov       : minimum bidirectional coverage (0–1)

    Returns
    -------
    dict mapping safe_id → cluster_rep_id
    """
    mmseqs_bin = shutil.which("mmseqs")
    if mmseqs_bin is None:
        raise RuntimeError(
            "mmseqs not found on PATH. "
            "Install with: conda install -c bioconda mmseqs2"
        )

    fasta_in  = os.path.join(tmpdir, "input.faa")
    out_prefix = os.path.join(tmpdir, "clust")
    mm_tmp    = os.path.join(tmpdir, "mmtmp")
    os.makedirs(mm_tmp, exist_ok=True)

    # Write input FASTA (use safe_id as header so we can map back)
    with open(fasta_in, "w") as fh:
        for safe_id, _header, seq in records:
            fh.write(f">{safe_id}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i+60] + "\n")

    print(f"  Running MMseqs2 easy-linclust "
          f"(id={min_seq_id:.0%}, cov={cov:.0%}, threads={threads}) ...")

    cmd = [
        mmseqs_bin, "easy-linclust",
        fasta_in, out_prefix, mm_tmp,
        "--min-seq-id",  str(min_seq_id),
        "-c",            str(cov),
        "--cov-mode",    "0",       # bidirectional coverage
        "--threads",     str(threads),
        "-v",            "1",       # minimal verbosity
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"MMseqs2 failed with exit code {result.returncode}")

    # Parse cluster TSV: rep_id <TAB> member_id
    cluster_tsv = out_prefix + "_cluster.tsv"
    id_to_cluster: dict[str, str] = {}
    with open(cluster_tsv) as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            id_to_cluster[member] = rep

    n_clusters = len(set(id_to_cluster.values()))
    print(f"  MMseqs2: {len(records):,} sequences → {n_clusters:,} clusters")
    return id_to_cluster


# ---------------------------------------------------------------------------
# Cluster-aware train / val / test split
# ---------------------------------------------------------------------------

def cluster_aware_split(
    records: list[tuple[str, str, str]],      # (safe_id, header, seq)
    id_to_cluster: dict[str, str],
    rng: random.Random,
    train_frac: float = 0.60,
    val_frac:   float = 0.20,
) -> dict[str, str]:
    """
    Assign each sequence to a split ('train' / 'val' / 'test') such that
    all members of the same cluster are always in the same split.
    
    => to avoid leakage

    Clusters are shuffled randomly then greedily filled into train until
    we reach train_frac of total sequences, then val, rest goes to test.

    Returns
    -------
    dict mapping safe_id → split_name
    """
    # Group sequence IDs by cluster representative
    clusters: dict[str, list[str]] = defaultdict(list)
    for safe_id, _header, _seq in records:
        rep = id_to_cluster.get(safe_id, safe_id)  # fallback: self-cluster
        clusters[rep].append(safe_id)

    cluster_list = list(clusters.items())  # [(rep, [ids...]), ...]
    rng.shuffle(cluster_list)

    n_total      = len(records)
    n_train_tgt  = int(n_total * train_frac)
    n_val_tgt    = int(n_total * val_frac)

    id_to_split: dict[str, str] = {}
    n_train, n_val = 0, 0

    for rep, members in cluster_list:
        if n_train < n_train_tgt:
            split = "train"
            n_train += len(members)
        elif n_val < n_val_tgt:
            split = "val"
            n_val += len(members)
        else:
            split = "test"
        for sid in members:
            id_to_split[sid] = split

    return id_to_split


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--viral",     required=True, help="RefSeq viral FASTA")
    parser.add_argument("--nonviral",  required=True, help="UniProtKB non-viral FASTA")
    parser.add_argument("--outdir",    required=True, help="Output directory")
    parser.add_argument("--n_samples", type=int, default=50_000,
                        help="Sequences per class (default: 50000)")
    parser.add_argument("--threads",   type=int, default=4,
                        help="CPU threads for MMseqs2 (default: 4)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("ESM2 Viral Probe — Preprocessing (cluster-aware splits)")
    print("=" * 60)
    print(f"Homology threshold: {MIN_SEQ_ID:.0%} identity, {COV:.0%} coverage")
    print()

    # ---- Step 1: Load, filter, and subsample ----
    all_records: list[tuple[str, str, str]] = []  # (safe_id, header, seq)
    label_map: dict[str, str] = {}  # safe_id → "viral" | "nonviral"

    for label, path in [("viral", args.viral), ("nonviral", args.nonviral)]:
        print(f"[{label}] Loading {path} ...")
        raw = list(parse_fasta(path))
        print(f"  Raw sequences: {len(raw):,}")

        filtered = [(h, s) for h, s in raw if is_valid(s)]
        print(f"  After quality filter: {len(filtered):,}")

        sampled = balanced_sample(filtered, args.n_samples, rng)
        print(f"  After length-stratified subsample: {len(sampled):,}")

        for idx, (header, seq) in enumerate(sampled):
            # Build a safe ID (no whitespace) for MMseqs2 headers
            acc = header.split()[0]
            safe_id = f"{label}_{idx}_{acc}"
            all_records.append((safe_id, header, seq))
            label_map[safe_id] = label

        print()

    print(f"Total sequences for clustering: {len(all_records):,}")

    # ---- Step 2: MMseqs2 homology clustering ----
    tmpdir = os.path.join(args.outdir, "_mmseqs_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    try:
        id_to_cluster = run_mmseqs2_cluster(
            all_records, tmpdir,
            threads=args.threads,
            min_seq_id=MIN_SEQ_ID,
            cov=COV,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ---- Step 3: Cluster-aware splits ----
    print("\nAssigning clusters to train/val/test splits ...")
    id_to_split = cluster_aware_split(all_records, id_to_cluster, rng)

    # Tally and report
    split_label_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for safe_id, split in id_to_split.items():
        split_label_counts[split][label_map[safe_id]] += 1

    for split in ["train", "val", "test"]:
        counts = split_label_counts[split]
        total  = sum(counts.values())
        print(f"  {split:5s}: {total:6,}  "
              f"(viral={counts['viral']:,}, nonviral={counts['nonviral']:,})")

    # ---- Step 4: Write output FASTAs ----
    print("\nWriting output FASTAs ...")

    # Organise into {label: {split: [(header, seq)]}}
    split_data: dict[str, dict[str, list]] = {
        lbl: {sp: [] for sp in ["train", "val", "test"]}
        for lbl in ["viral", "nonviral"]
    }
    for safe_id, header, seq in all_records:
        lbl   = label_map[safe_id]
        split = id_to_split[safe_id]
        split_data[lbl][split].append((header, seq))

    for lbl in ["viral", "nonviral"]:
        for split in ["train", "val", "test"]:
            out_path = os.path.join(args.outdir, f"{lbl}_{split}.faa")
            write_fasta(split_data[lbl][split], out_path)
            print(f"  Wrote {out_path}  ({len(split_data[lbl][split]):,} seqs)")

    # ---- Step 5: Write metadata TSV ----
    meta_path = os.path.join(args.outdir, "metadata.tsv")
    with open(meta_path, "w") as fh:
        fh.write("accession\tlabel\tsplit\n")
        for safe_id, header, _ in all_records:
            acc   = header.split()[0]
            lbl   = label_map[safe_id]
            split = id_to_split[safe_id]
            fh.write(f"{acc}\t{lbl}\t{split}\n")
    print(f"\nMetadata written to {meta_path}")

    # ---- Report cross-split contamination check ----
    print("\nCross-split contamination check (shared clusters across splits):")
    cluster_to_splits: dict[str, set] = defaultdict(set)
    for safe_id, split in id_to_split.items():
        rep = id_to_cluster.get(safe_id, safe_id)
        cluster_to_splits[rep].add(split)
    contaminated = sum(1 for splits in cluster_to_splits.values() if len(splits) > 1)
    print(f"  Clusters spanning multiple splits: {contaminated}"
          f" (should be 0 if clustering was deterministic)")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
