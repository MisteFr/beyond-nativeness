#!/usr/bin/env bash
# ============================================================
# ESM3 Medium masked reconstruction via Forge API
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm3_masked_reconstruction"

DATA_DIR="$PROJECT/data"
OUT_DIR="$PROJECT/results/esm3_medium"

export FORGE_TOKEN="${FORGE_TOKEN:?Set FORGE_TOKEN to your EvolutionaryScale Forge API key (see docs/forge_api_setup.md)}"


# Activate the project environment first: conda activate beyond-nativeness

# ---- Validate inputs ----
echo "Validating input files..."
for LABEL in viral nonviral; do
    for SPLIT in train val test; do
        FASTA="$DATA_DIR/${LABEL}_${SPLIT}.faa"
        if [ ! -f "$FASTA" ]; then
            echo "ERROR: Missing required file: $FASTA"
            exit 1
        fi
    done
done
echo "  All 6 FASTA splits found."

mkdir -p "$OUT_DIR"

# ---- Run experiment ----
echo ""
echo "Starting ESM3 Medium API masked reconstruction..."
python "$PROJECT/scripts/run_masked_reconstruction_forge.py" \
    --data_dir    "$DATA_DIR" \
    --out_dir     "$OUT_DIR" \
    --model_name  esm3-medium-2024-08 \
    --client_type esm3 \
    --mask_rate   0.15 \
    --n_seeds     3 \
    --timeout     120

echo ""
echo "======================================"
echo "Completed: $(date)"
echo "Output files:"
ls -lh "$OUT_DIR/" 2>/dev/null || echo "  (output dir empty)"
echo "======================================"
