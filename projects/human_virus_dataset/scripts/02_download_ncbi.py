"""Import manually downloaded NCBI Virus protein FASTA into pipeline outputs.

This project now uses a manually exported FASTA file from the NCBI Virus UI
instead of querying Entrez directly. The expected local input is:

  sequences-2.fasta

Outputs:
  data/raw/ncbi/human_virus_refseq.faa.gz   — gzipped copy of the FASTA
  data/raw/ncbi/human_virus_refseq.tsv.gz   — metadata derived from headers
"""

import argparse
import csv
import gzip
import re
from pathlib import Path
from typing import Dict

HEADER_RE = re.compile(r"^(?P<accession>\S+)\s+\|(?P<title>.*?)(?:\s+\[(?P<organism>.+)\])?$")


def iter_fasta(path: Path):
    header = None
    seq_chunks = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
    if header is not None:
        yield header, "".join(seq_chunks)


def parse_header(header: str) -> Dict[str, str]:
    match = HEADER_RE.match(header)
    if not match:
        first_token = header.split()[0]
        return {
            "accession": first_token,
            "title": header[len(first_token):].strip(),
            "organism": "",
            "taxid": "",
        }

    return {
        "accession": match.group("accession"),
        "title": (match.group("title") or "").strip(),
        "organism": (match.group("organism") or "").strip(),
        "taxid": "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_fasta",
        default=str(Path(__file__).parents[1] / "sequences-2.fasta"),
        help="Manual FASTA downloaded from the NCBI Virus UI",
    )
    parser.add_argument(
        "--out_dir",
        default=str(Path(__file__).parents[1] / "data" / "raw" / "ncbi"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Deprecated and ignored; kept for backward compatibility",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="Deprecated and ignored; kept for backward compatibility",
    )
    args = parser.parse_args()

    input_fasta = Path(args.input_fasta)
    if not input_fasta.exists():
        raise FileNotFoundError(
            f"Manual NCBI FASTA not found: {input_fasta}\n"
            "Download it from the NCBI Virus UI and place it at this path, or pass --input_fasta."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faa_path = out_dir / "human_virus_refseq.faa.gz"
    tsv_path = out_dir / "human_virus_refseq.tsv.gz"

    if (faa_path.exists() or tsv_path.exists()) and not args.force:
        existing = [str(p) for p in (faa_path, tsv_path) if p.exists()]
        raise FileExistsError(
            "Output file(s) already exist. Use --force to overwrite:\n"
            + "\n".join(existing)
        )

    faa_tmp = faa_path.with_name(faa_path.name + ".part")
    tsv_tmp = tsv_path.with_name(tsv_path.name + ".part")

    print(f"Importing manual NCBI FASTA: {input_fasta}", flush=True)
    count = 0

    with gzip.open(faa_tmp, "wt") as faa_fh, gzip.open(tsv_tmp, "wt", newline="") as tsv_fh:
        writer = csv.DictWriter(
            tsv_fh,
            fieldnames=["accession", "title", "organism", "taxid", "length"],
            delimiter="\t",
        )
        writer.writeheader()

        for header, seq in iter_fasta(input_fasta):
            count += 1
            faa_fh.write(f">{header}\n")
            faa_fh.write(f"{seq}\n")

            row = parse_header(header)
            row["length"] = str(len(seq))
            writer.writerow(row)

    if count == 0:
        faa_tmp.unlink(missing_ok=True)
        tsv_tmp.unlink(missing_ok=True)
        raise RuntimeError(f"No FASTA records found in {input_fasta}")

    faa_tmp.replace(faa_path)
    tsv_tmp.replace(tsv_path)

    print(f"  Imported {count:,} sequences", flush=True)
    print(f"  FASTA:    {faa_path}", flush=True)
    print(f"  Metadata: {tsv_path}", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
