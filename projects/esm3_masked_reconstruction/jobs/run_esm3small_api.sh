#!/usr/bin/env bash
# ============================================================
# ESM3 Small (Forge API) masked token reconstruction
#
# Runs the same masked reconstruction experiment as
# run_reconstruction.sh, but uses the ESM3-small-2024-08
# model via EvolutionaryScale Forge API instead of loading
# ESM3 Open locally.
#
# Fully resumable: re-submit if interrupted — completed
# sequences are skipped automatically.
#
# Estimated runtime: ~31,200 API calls at ~1-3s each = 9-26h
# Input:  esm_viral_probe/datasets/human_virus/data/processed/
# Output: esm3_masked_reconstruction/results/esm3_small_api/
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm3_masked_reconstruction"
PROBE_PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe"

DATA_DIR="$PROBE_PROJECT/datasets/human_virus/data/processed"
OUT_DIR="$PROJECT/results/esm3_small_api"
FORGE_TOKEN="${FORGE_TOKEN:?Set FORGE_TOKEN to your EvolutionaryScale Forge API key (see docs/forge_api_setup.md)}"
FORGE_MODEL="esm3-small-2024-08"


# Activate the project environment first: conda activate beyond-nativeness

echo "Python: $(which python)"
echo "ESM:    $(python -c 'import esm; print(esm.__version__)' 2>/dev/null || echo 'not found')"

# ---- Validate inputs ----
echo ""
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
mkdir -p "$PROJECT/logs"

# ---- Run experiment ----
echo ""
echo "Starting ESM3 Small API masked reconstruction..."
echo "  Model:      $FORGE_MODEL"
echo "  Data dir:   $DATA_DIR"
echo "  Output dir: $OUT_DIR"
echo ""

export FORGE_TOKEN="$FORGE_TOKEN"

python "$PROJECT/scripts/run_masked_reconstruction_api.py" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$OUT_DIR" \
    --model      "$FORGE_MODEL" \
    --mask_rate  0.15 \
    --n_seeds    3 \
    --timeout    120

echo ""
echo "======================================"
echo "Completed: $(date)"
echo "Output files:"
ls -lh "$OUT_DIR/" 2>/dev/null || echo "  (output dir empty)"
echo "======================================"
