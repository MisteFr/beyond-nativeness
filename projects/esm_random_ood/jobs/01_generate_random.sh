#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# Step 1: Generate random + shuffled OOD control protein sequences
#
# Input:  esm_viral_probe/datasets/human_virus/data/processed/
# Output: data/{random_uniform,shuffled_viral,shuffled_nonviral}.faa
# ============================================================

# Activate the project environment first: conda activate beyond-nativeness

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repo root}/projects/esm_random_ood"
PROBE_PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe"

python "$PROJECT/scripts/generate_random_sequences.py" \
    --proc_dir "$PROBE_PROJECT/datasets/human_virus/data/processed" \
    --out_dir  "$PROJECT/data" \
    --n_random 5000 \
    --seed     42

echo ""
echo "[$(date)] Done. Output files:"
ls -lh "$PROJECT/data/"*.faa "$PROJECT/data/generation_stats.json"
