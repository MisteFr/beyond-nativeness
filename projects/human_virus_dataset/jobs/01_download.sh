#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${BEYOND_NATIVENESS_ROOT:?set BEYOND_NATIVENESS_ROOT to the repo root}/projects/human_virus_dataset"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

# Activate the project environment first: conda activate beyond-nativeness

# --- Step 1: UniProt download ---
echo "--- Step 1: UniProt download ---"
python "${SCRIPTS_DIR}/01_download_uniprot.py"

# --- Step 2: NCBI manual FASTA import ---
echo "--- Step 2: NCBI manual FASTA import ---"
python "${SCRIPTS_DIR}/02_download_ncbi.py"
