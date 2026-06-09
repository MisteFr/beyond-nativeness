#!/usr/bin/env bash
set -euo pipefail

# Activate the project environment first: conda activate beyond-nativeness

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/prokaryote_phage_ood"

cd "$PROJECT"

echo "=== Filtering plants (subsample 5000) ==="
python scripts/02_clean_filter.py --group plants --max_seqs 5000

echo "=== Filtering fungi (subsample 5000) ==="
python scripts/02_clean_filter.py --group fungi --max_seqs 5000

echo "=== Filtering insects (subsample 5000) ==="
python scripts/02_clean_filter.py --group insects --max_seqs 5000

echo "=== Filtering plant_virus (all) ==="
python scripts/02_clean_filter.py --group plant_virus

echo "=== Filtering invertebrate_virus (all) ==="
python scripts/02_clean_filter.py --group invertebrate_virus

echo "=== Done ==="
