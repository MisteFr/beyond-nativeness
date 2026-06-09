"""Download human-virus proteins from UniProtKB Swiss-Prot via REST API.

Query: reviewed:true AND virus_host_id:9606 AND (existence:1 OR existence:2 OR existence:3)

Outputs:
  data/raw/uniprot/human_virus.faa.gz   — FASTA sequences
  data/raw/uniprot/human_virus.tsv.gz   — metadata TSV
"""

import argparse
import gzip
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
QUERY = "reviewed:true AND virus_host_id:9606 AND (existence:1 OR existence:2 OR existence:3)"
TSV_FIELDS = "accession,id,protein_name,organism_name,organism_id,lineage,length,protein_existence"
PAGE_SIZE = 500
RETRY_WAIT = 10  # seconds to wait after a failed request before retrying
MAX_RETRIES = 5
NEXT_LINK_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


def build_url(base: str, params: dict, fields: str | None = None) -> str:
    """Build a URL with proper encoding.

    The UniProt API requires literal commas in the `fields` parameter — passing
    them via requests' `params` dict percent-encodes commas as %2C, which the
    server rejects with 400. We therefore construct the URL manually and use
    safe=',' so commas remain unencoded only in the fields value.
    """
    encoded = urllib.parse.urlencode(params)
    if fields:
        encoded += "&fields=" + urllib.parse.quote(fields, safe=",")
    return base + "?" + encoded


def fetch_page(url: str) -> requests.Response:
    """GET with simple retry logic."""
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
    """Extract the next-page URL from a UniProt Link header.

    UniProt's Link header sometimes returns a relative URL (query string only,
    or starting mid-field) rather than an absolute URL. To avoid this, we extract
    only the `cursor` parameter from whatever URL is in the Link header, then
    rebuild the full absolute URL ourselves using build_url.
    """
    match = NEXT_LINK_RE.search(link_header)
    if not match:
        return None
    raw = match.group(1)
    # raw might be a full URL ("https://...?cursor=X") or a bare query string
    # ("cursor=X&...") — split on "?" to isolate the query portion either way
    query_str = raw.split("?", 1)[-1]
    qs = urllib.parse.parse_qs(query_str)
    cursors = qs.get("cursor", [])
    if not cursors:
        return None
    # Rebuild the full URL with the cursor added to the original params
    return build_url(BASE_URL, {**base_params, "cursor": cursors[0]}, fields)


def download_format(out_path: Path, fmt: str, fields: str | None = None) -> int:
    """Paginate through UniProt results and write to out_path (gzipped).

    Returns total number of records written.
    """
    params: dict = {
        "query": QUERY,
        "format": fmt,
        "size": PAGE_SIZE,
    }

    count = 0
    page = 1
    url: str | None = build_url(BASE_URL, params, fields)

    with gzip.open(out_path, "wt") as fh:
        while url:
            print(f"  Fetching page {page} ({fmt})...", flush=True)
            resp = fetch_page(url)

            # On first page, read total from X-Total-Results header
            if page == 1:
                total = resp.headers.get("X-Total-Results", "?")
                print(f"  Total records reported by server: {total}", flush=True)

            text = resp.text
            fh.write(text)

            # Count records: for FASTA count ">", for TSV count lines minus header
            if fmt == "fasta":
                count += text.count("\n>") + (1 if text.startswith(">") else 0)
            elif fmt == "tsv":
                lines = text.splitlines()
                # Skip the header line on the first page
                count += len(lines) - (1 if page == 1 else 0) - lines.count("")

            # Build next-page URL by extracting the cursor from the Link header
            link_header = resp.headers.get("Link", "")
            url = next_url_from_link(link_header, params, fields)

            page += 1

    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        default=str(Path(__file__).parents[1] / "data" / "raw" / "uniprot"),
        help="Output directory",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faa_path = out_dir / "human_virus.faa.gz"
    tsv_path = out_dir / "human_virus.tsv.gz"

    # --- FASTA ---
    if faa_path.exists() and not args.force:
        print(f"[skip] {faa_path} already exists (use --force to re-download)")
    else:
        print(f"Downloading FASTA → {faa_path}")
        n = download_format(faa_path, fmt="fasta")
        print(f"  Wrote {n} sequences to {faa_path}")

    # --- TSV metadata ---
    if tsv_path.exists() and not args.force:
        print(f"[skip] {tsv_path} already exists (use --force to re-download)")
    else:
        print(f"Downloading TSV metadata → {tsv_path}")
        n = download_format(tsv_path, fmt="tsv", fields=TSV_FIELDS)
        print(f"  Wrote ~{n} rows to {tsv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
