#!/usr/bin/env bash
set -euo pipefail

# Activate the project environment first: conda activate beyond-nativeness

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/prokaryote_phage_ood"

cd "$PROJECT"

for GROUP in plants fungi insects plant_virus invertebrate_virus; do
    echo "=== Downloading $GROUP ==="
    python scripts/01_download_uniprot.py --group "$GROUP" --out_dir data/raw
    echo ""
done

echo "=== Done ==="
