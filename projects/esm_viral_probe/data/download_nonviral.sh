#!/bin/bash
# ============================================================
# Download UniProtKB/Swiss-Prot non-viral protein sequences
# We download the full Swiss-Prot and then exclude viral entries
# (OC line does not contain "Viruses" in the taxonomy lineage)
# Output: data/nonviral/uniprot_nonviral.faa
# ============================================================

set -euo pipefail

OUTDIR="$(dirname "$0")/nonviral"
mkdir -p "$OUTDIR"

UNIPROT_URL="https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
GZ="$OUTDIR/uniprot_sprot.fasta.gz"
RAW="$OUTDIR/uniprot_sprot.fasta"

# ---- Step 1: Download Swiss-Prot FASTA ----
if [ -f "$GZ" ]; then
    echo "[$(date)] $GZ already exists — skipping download"
else
    echo "[$(date)] Downloading Swiss-Prot FASTA (~270 MB)..."
    wget -q --show-progress -O "$GZ" "$UNIPROT_URL"
fi

echo "[$(date)] Decompressing..."
gunzip -k -f "$GZ"

# ---- Step 2: Download the full Swiss-Prot DAT for taxonomy filtering ----
# Swiss-Prot FASTA headers contain OS= and OX= but not full lineage.
# The simplest proxy: exclude entries with "OX=" (NCBI taxon ID) in the
# Viruses superkingdom. We use a pre-downloaded list of viral taxon IDs,
# or we filter by "Viruses" keyword present in UniProt FASTA headers:
# UniProt FASTA format: >sp|ACCESSION|ENTRY_NAME ... OS=... OX=... GN=... PE=... SV=...
# Viral sequences are tagged with OS= containing virus-related terms OR
# we use a taxonomy query via the UniProt REST API to get a list of viral accessions.

echo "[$(date)] Fetching list of viral UniProt accessions via REST API (taxonomy:10239)..."
VIRAL_ACC="$OUTDIR/viral_accessions.txt"

# UniProt REST: all Swiss-Prot accessions in Viruses (taxon 10239)
curl -s "https://rest.uniprot.org/uniprotkb/search?query=taxonomy_id:10239+AND+reviewed:true&fields=accession&format=tsv&size=500" \
    -o "$OUTDIR/viral_page1.tsv" || true

# For a robust approach, iterate pages
# Pass $OUTDIR into the Python heredoc via environment variable
export OUTDIR
python3 - <<'PYEOF'
import urllib.request, json, os, sys

out_file = os.path.join(os.environ["OUTDIR"], "viral_accessions.txt")
base_url = (
    "https://rest.uniprot.org/uniprotkb/search"
    "?query=taxonomy_id:10239+AND+reviewed:true"
    "&fields=accession&format=tsv&size=500"
)

accessions = set()
url = base_url
page = 0
while url:
    page += 1
    print(f"  Page {page}: {url[:80]}...", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode()
            # Parse TSV (skip header)
            for line in body.strip().split("\n")[1:]:
                acc = line.strip()
                if acc:
                    accessions.add(acc)
            # Check Link header for next page
            link = resp.headers.get("Link", "")
            url = None
            if 'rel="next"' in link:
                import re
                m = re.search(r'<([^>]+)>; rel="next"', link)
                if m:
                    url = m.group(1)
    except Exception as e:
        print(f"  Warning: {e}", file=sys.stderr)
        break

print(f"Total viral Swiss-Prot accessions: {len(accessions)}")
with open(out_file, "w") as f:
    f.write("\n".join(sorted(accessions)) + "\n")
PYEOF

echo "[$(date)] Filtering out viral sequences from Swiss-Prot FASTA..."

export OUTDIR
python3 - <<'PYEOF'
import os

base           = os.environ["OUTDIR"]
viral_acc_file = os.path.join(base, "viral_accessions.txt")
raw_fasta      = os.path.join(base, "uniprot_sprot.fasta")
out_fasta      = os.path.join(base, "uniprot_nonviral.faa")

# Load viral accessions into a set for O(1) lookup
with open(viral_acc_file) as f:
    viral_accs = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(viral_accs):,} viral accessions to exclude")

kept = 0
skipped = 0
current_is_viral = False

with open(raw_fasta) as fin, open(out_fasta, "w") as fout:
    for line in fin:
        if line.startswith(">"):
            # Header format: >sp|P12345|ENTRY_NAME ...
            parts = line.split("|")
            acc = parts[1] if len(parts) >= 2 else ""
            current_is_viral = acc in viral_accs
            if current_is_viral:
                skipped += 1
            else:
                kept += 1
                fout.write(line)
        elif not current_is_viral:
            fout.write(line)

print(f"Kept {kept:,} non-viral sequences, skipped {skipped:,} viral sequences")
print(f"Output: {out_fasta}")
PYEOF

echo "[$(date)] Done. Non-viral sequences saved to $OUTDIR/uniprot_nonviral.faa"
