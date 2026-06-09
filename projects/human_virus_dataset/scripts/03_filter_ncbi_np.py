"""Filter NCBI RefSeq proteins to NP_ accessions only.

NP_ = proteins from manually annotated genomic RefSeq records (highest curation level).
Excludes YP_ (computationally predicted), WP_ (non-redundant), etc.

Inputs:
  data/raw/ncbi/human_virus_refseq.faa.gz
  data/raw/ncbi/human_virus_refseq.tsv.gz

Outputs:
  data/processed/ncbi_human_np.faa
  data/processed/ncbi_human_np.tsv
"""

import argparse
import csv
import gzip
from pathlib import Path

import sys

# Allow importing utils from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from utils import iter_fasta, write_fasta, accession_from_header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_dir",
        default=str(Path(__file__).parents[1] / "data" / "raw" / "ncbi"),
    )
    parser.add_argument(
        "--out_dir",
        default=str(Path(__file__).parents[1] / "data" / "processed"),
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faa_in = raw_dir / "human_virus_refseq.faa.gz"
    tsv_in = raw_dir / "human_virus_refseq.tsv.gz"
    faa_out = out_dir / "ncbi_human_np.faa"
    tsv_out = out_dir / "ncbi_human_np.tsv"

    # --- Filter FASTA ---
    print(f"Filtering FASTA: {faa_in}", flush=True)
    kept = 0
    total = 0

    def np_records():
        nonlocal kept, total
        for header, seq in iter_fasta(faa_in):
            total += 1
            acc = accession_from_header(header)
            # NCBI RefSeq FASTA headers often look like:
            #   NP_123456.1 protein title [Organism name]
            # The accession is the first token; strip version suffix for prefix check
            bare_acc = acc.split(".")[0]
            if bare_acc.startswith("NP_"):
                kept += 1
                yield header, seq

    n_written = write_fasta(np_records(), faa_out)
    print(f"  Total records: {total:,}  |  NP_ kept: {kept:,}  ({100*kept/max(total,1):.1f}%)")
    print(f"  Wrote {n_written} sequences to {faa_out}")

    # --- Filter metadata TSV ---
    print(f"Filtering TSV: {tsv_in}", flush=True)
    kept_meta = 0
    total_meta = 0

    with gzip.open(tsv_in, "rt") as fh_in, open(tsv_out, "w", newline="") as fh_out:
        reader = csv.DictReader(fh_in, delimiter="\t")
        assert reader.fieldnames is not None, "TSV has no header"
        writer = csv.DictWriter(fh_out, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            total_meta += 1
            bare_acc = row["accession"].split(".")[0]
            if bare_acc.startswith("NP_"):
                writer.writerow(row)
                kept_meta += 1

    print(f"  Total metadata rows: {total_meta:,}  |  NP_ kept: {kept_meta:,}")
    print(f"  Wrote metadata to {tsv_out}")
    print("Done.")


if __name__ == "__main__":
    main()
