#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${BEYOND_NATIVENESS_ROOT:?set BEYOND_NATIVENESS_ROOT to the repo root}/projects/human_virus_dataset"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

# Activate the project environment first: conda activate beyond-nativeness

# --- Step 3: Filter NCBI to NP_ ---
echo "--- Step 3: Filter NCBI to NP_ accessions ---"
python "${SCRIPTS_DIR}/03_filter_ncbi_np.py"

# --- Step 4: Validate and report stats (family composition for Table 2) ---
echo "--- Step 4: Validate and report stats ---"
python "${SCRIPTS_DIR}/04_validate_stats.py"
