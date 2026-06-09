#!/usr/bin/env python3
"""Download bacteria, archaea, or phage proteins from UniProtKB Swiss-Prot.

Queries:
  bacteria: reviewed:true AND taxonomy_id:2 AND (existence:1 OR existence:2 OR existence:3) AND NOT keyword:KW-0181
  archaea:  reviewed:true AND taxonomy_id:2157 AND (existence:1 OR existence:2 OR existence:3) AND NOT keyword:KW-0181
  phage:    reviewed:true AND taxonomy_id:10239 AND (existence:1 OR existence:2 OR existence:3) AND NOT keyword:KW-0181
            (downloads ALL reviewed viruses; phage filtering happens in 02_clean_filter.py)

Usage:
    python scripts/01_download_uniprot.py --group bacteria --out_dir data/raw
    python scripts/01_download_uniprot.py --group archaea  --out_dir data/raw
    python scripts/01_download_uniprot.py --group phage    --out_dir data/raw
"""

import argparse
import gzip
import re
import time
import urllib.parse
from pathlib import Path

import requests

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
PAGE_SIZE = 500
RETRY_WAIT = 10
MAX_RETRIES = 5
NEXT_LINK_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')

QUERIES = {
    "bacteria":            "reviewed:true AND taxonomy_id:2 AND (existence:1 OR existence:2 OR existence:3) AND NOT keyword:KW-0181",
    "archaea":             "reviewed:true AND taxonomy_id:2157 AND (existence:1 OR existence:2 OR existence:3) AND NOT keyword:KW-0181",
    "phage":               "reviewed:true AND taxonomy_id:10239 AND (existence:1 OR existence:2 OR existence:3) AND NOT keyword:KW-0181",
    "plants":              "reviewed:true AND taxonomy_id:33090 AND (existence:1 OR existence:2 OR existence:3) AND NOT keyword:KW-0181",
    "fungi":               "reviewed:true AND taxonomy_id:4751 AND (existence:1 OR existence:2 OR existence:3) AND NOT keyword:KW-0181",
    "insects":             "reviewed:true AND taxonomy_id:50557 AND (existence:1 OR existence:2 OR existence:3) AND NOT keyword:KW-0181",
    "plant_virus":         "reviewed:true AND taxonomy_id:10239 AND virus_host_id:33090 AND (existence:1 OR existence:2 OR existence:3) AND NOT keyword:KW-0181",
    "invertebrate_virus":  "reviewed:true AND taxonomy_id:10239 AND virus_host_id:6656 AND NOT virus_host_id:9606 AND (existence:1 OR existence:2 OR existence:3) AND NOT keyword:KW-0181",
}

TSV_FIELDS = "accession,id,protein_name,organism_name,organism_id,lineage,length,protein_existence"


def build_url(base: str, params: dict, fields: str | None = None) -> str:
    encoded = urllib.parse.urlencode(params)
    if fields:
        encoded += "&fields=" + urllib.parse.quote(fields, safe=",")
    return base + "?" + encoded


def fetch_page(url: str) -> requests.Response:
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


def download_format(query: str, out_path: Path, fmt: str, fields: str | None = None) -> int:
    params = {"query": query, "format": fmt, "size": PAGE_SIZE}
    count = 0
    page = 1
    url = build_url(BASE_URL, params, fields)

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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, choices=list(QUERIES.keys()))
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query = QUERIES[args.group]
    faa_path = out_dir / f"{args.group}.faa.gz"
    tsv_path = out_dir / f"{args.group}.tsv.gz"

    print(f"Group: {args.group}")
    print(f"Query: {query}")

    # FASTA
    if faa_path.exists() and not args.force:
        print(f"[skip] {faa_path} exists (use --force)")
    else:
        print(f"Downloading FASTA -> {faa_path}")
        n = download_format(query, faa_path, fmt="fasta")
        print(f"  Wrote {n:,} sequences")

    # TSV metadata
    if tsv_path.exists() and not args.force:
        print(f"[skip] {tsv_path} exists (use --force)")
    else:
        print(f"Downloading TSV -> {tsv_path}")
        n = download_format(query, tsv_path, fmt="tsv", fields=TSV_FIELDS)
        print(f"  Wrote ~{n:,} rows")

    print("Done.")


if __name__ == "__main__":
    main()
