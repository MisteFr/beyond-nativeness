#!/usr/bin/env bash
set -euo pipefail

# Activate the project environment first: conda activate beyond-nativeness

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/prokaryote_phage_ood"

cd "$PROJECT"

echo "=== Downloading bacteria ==="
python scripts/01_download_uniprot.py --group bacteria --out_dir data/raw

echo ""
echo "=== Downloading archaea ==="
python scripts/01_download_uniprot.py --group archaea --out_dir data/raw

echo ""
echo "=== Downloading phage (all reviewed viruses) ==="
python scripts/01_download_uniprot.py --group phage --out_dir data/raw

echo ""
echo "=== Done ==="
