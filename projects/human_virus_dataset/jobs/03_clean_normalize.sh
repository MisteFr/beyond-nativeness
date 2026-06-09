#!/usr/bin/env bash
set -euo pipefail

PROJ="${BEYOND_NATIVENESS_ROOT:?set BEYOND_NATIVENESS_ROOT to the repo root}/projects/human_virus_dataset"

# Activate the project environment first: conda activate beyond-nativeness

cd "$PROJ"

# --- Step 5: Merge, clean, deduplicate ---
echo "--- Step 5: Merge, clean, deduplicate ---"
python scripts/05_clean_normalize.py
