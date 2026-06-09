#!/usr/bin/env bash
# ============================================================
# Random uniform — ESMC 6B API masked reconstruction
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm3_masked_reconstruction"
FASTA="${BEYOND_NATIVENESS_ROOT}/projects/esm_random_ood/data/random_uniform.faa"
OUT_DIR="$PROJECT/results/esmc_6b/random_uniform"

export FORGE_TOKEN="${FORGE_TOKEN:?Set FORGE_TOKEN to your EvolutionaryScale Forge API key (see docs/forge_api_setup.md)}"


# Activate the project environment first: conda activate beyond-nativeness

if [ ! -f "$FASTA" ]; then
    echo "ERROR: Missing FASTA: $FASTA"
    exit 1
fi
echo "Input: $FASTA ($(wc -l < "$FASTA") lines)"

mkdir -p "$OUT_DIR"

python "$PROJECT/scripts/run_masked_reconstruction_single_fasta.py" \
    --fasta       "$FASTA" \
    --label       random \
    --out_dir     "$OUT_DIR" \
    --backend     forge \
    --model_name  esmc-6b-2024-12 \
    --client_type esmc \
    --mask_rate   0.15 \
    --n_seeds     3 \
    --timeout     120

echo ""
echo "======================================"
echo "Completed: $(date)"
echo "Output:"
ls -lh "$OUT_DIR/" 2>/dev/null || echo "  (empty)"
echo "======================================"
