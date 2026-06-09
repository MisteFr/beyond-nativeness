#!/usr/bin/env python3
"""
generate_random_sequences.py
============================
Generate OOD control sequences for the random-OOD embedding experiment.

Two types of sequences are generated:

1. **Uniform random**: iid amino acid sampling from the 20 standard AAs,
   with lengths drawn from the empirical length distribution of existing
   viral + non-viral sequences.

2. **Shuffled**: randomly permute the amino acid order of each real sequence
   (preserving exact AA composition and length). Separate files for viral
   and non-viral sources.

Usage:
    python generate_random_sequences.py \
        --proc_dir  /path/to/esm_viral_probe/datasets/human_virus/data/processed \
        --out_dir   data \
        --n_random  5000 \
        --seed      42
"""

import argparse
import json
import os
import time
from typing import Iterator

import numpy as np


STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"


def parse_fasta(path: str) -> Iterator[tuple[str, str]]:
    """Yield (accession, sequence) pairs from a FASTA file."""
    header, seq_parts = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header.split()[0], "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.upper())
    if header is not None:
        yield header.split()[0], "".join(seq_parts)


def write_fasta(records: list[tuple[str, str]], path: str, line_width: int = 60):
    """Write (header, sequence) pairs to a FASTA file."""
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), line_width):
                fh.write(seq[i:i + line_width] + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--proc_dir", required=True,
                        help="Directory with {viral,nonviral}_{train,val,test}.faa")
    parser.add_argument("--out_dir", default="data",
                        help="Output directory")
    parser.add_argument("--n_random", type=int, default=5000,
                        help="Number of uniform random sequences (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load all existing sequences
    # ------------------------------------------------------------------
    print("[1/4] Loading existing sequences...", flush=True)
    viral_records = []
    nonviral_records = []
    for split in ("train", "val", "test"):
        for label, storage in [("viral", viral_records), ("nonviral", nonviral_records)]:
            faa = os.path.join(args.proc_dir, f"{label}_{split}.faa")
            if not os.path.exists(faa):
                print(f"  WARNING: {faa} not found — skipping", flush=True)
                continue
            records = list(parse_fasta(faa))
            storage.extend(records)
            print(f"  {label}_{split}: {len(records):,} sequences", flush=True)

    all_records = viral_records + nonviral_records
    all_lengths = np.array([len(seq) for _, seq in all_records])
    print(f"  Total: {len(all_records):,} sequences", flush=True)
    print(f"  Length range: {all_lengths.min()}–{all_lengths.max()} aa "
          f"(mean {all_lengths.mean():.0f}, median {np.median(all_lengths):.0f})", flush=True)

    # ------------------------------------------------------------------
    # 2. Generate uniform random sequences
    # ------------------------------------------------------------------
    print(f"\n[2/4] Generating {args.n_random:,} uniform random sequences...", flush=True)
    sampled_lengths = rng.choice(all_lengths, size=args.n_random, replace=True)
    aa_arr = np.array(list(STANDARD_AAS))

    random_records = []
    for i, length in enumerate(sampled_lengths):
        seq = "".join(rng.choice(aa_arr, size=int(length)))
        header = f"RAND_{i+1:05d} random_uniform len={length}"
        random_records.append((header, seq))

    random_path = os.path.join(args.out_dir, "random_uniform.faa")
    write_fasta(random_records, random_path)
    random_lengths = sampled_lengths
    print(f"  Written: {random_path} ({len(random_records):,} seqs)", flush=True)
    print(f"  Length range: {random_lengths.min()}–{random_lengths.max()} aa "
          f"(mean {random_lengths.mean():.0f})", flush=True)

    # ------------------------------------------------------------------
    # 3. Generate shuffled sequences
    # ------------------------------------------------------------------
    print(f"\n[3/4] Generating shuffled sequences...", flush=True)

    shuffled_viral = []
    for acc, seq in viral_records:
        aa_list = list(seq)
        rng.shuffle(aa_list)
        shuffled_seq = "".join(aa_list)
        header = f"SHUF_V_{acc} shuffled_viral len={len(seq)}"
        shuffled_viral.append((header, shuffled_seq))

    shuffled_nonviral = []
    for acc, seq in nonviral_records:
        aa_list = list(seq)
        rng.shuffle(aa_list)
        shuffled_seq = "".join(aa_list)
        header = f"SHUF_N_{acc} shuffled_nonviral len={len(seq)}"
        shuffled_nonviral.append((header, shuffled_seq))

    shuf_v_path = os.path.join(args.out_dir, "shuffled_viral.faa")
    shuf_n_path = os.path.join(args.out_dir, "shuffled_nonviral.faa")
    write_fasta(shuffled_viral, shuf_v_path)
    write_fasta(shuffled_nonviral, shuf_n_path)
    print(f"  Shuffled viral:     {shuf_v_path} ({len(shuffled_viral):,} seqs)", flush=True)
    print(f"  Shuffled non-viral: {shuf_n_path} ({len(shuffled_nonviral):,} seqs)", flush=True)

    # ------------------------------------------------------------------
    # 4. Save metadata
    # ------------------------------------------------------------------
    print(f"\n[4/4] Saving metadata...", flush=True)
    stats = {
        "seed": args.seed,
        "source_dir": os.path.abspath(args.proc_dir),
        "n_source_viral": len(viral_records),
        "n_source_nonviral": len(nonviral_records),
        "n_random_uniform": len(random_records),
        "n_shuffled_viral": len(shuffled_viral),
        "n_shuffled_nonviral": len(shuffled_nonviral),
        "length_stats": {
            "source_mean": float(all_lengths.mean()),
            "source_median": float(np.median(all_lengths)),
            "source_min": int(all_lengths.min()),
            "source_max": int(all_lengths.max()),
            "random_mean": float(random_lengths.mean()),
            "random_median": float(np.median(random_lengths)),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    stats_path = os.path.join(args.out_dir, "generation_stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  Stats: {stats_path}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
