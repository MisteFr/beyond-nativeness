#!/usr/bin/env bash
set -euo pipefail

# Activate the project environment first: conda activate beyond-nativeness

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/prokaryote_phage_ood"

cd "$PROJECT"

echo "=== Filtering bacteria (subsample to 5000) ==="
python scripts/02_clean_filter.py --group bacteria --raw_dir data/raw --out_dir data/processed --max_seqs 5000

echo ""
echo "=== Filtering archaea (all available) ==="
python scripts/02_clean_filter.py --group archaea --raw_dir data/raw --out_dir data/processed

echo ""
echo "=== Filtering phage (from broad virus download) ==="
python scripts/02_clean_filter.py --group phage --raw_dir data/raw --out_dir data/processed

echo ""
echo "=== Sequence counts ==="
for f in data/processed/*_clean.faa; do
    echo "  $(grep -c '^>' "$f") sequences in $(basename "$f")"
done

echo "=== Done ==="
