#!/usr/bin/env bash
# ============================================================
# ESM3 masked token reconstruction experiment
#
# Masks 15% of amino acid positions in each protein sequence
# and measures how well ESM3-open reconstructs them. Compares
# perplexity and recovery rate between viral and non-viral seqs.
#
# Input:  esm_viral_probe/datasets/human_virus/data/processed/
# Output: esm3_masked_reconstruction/results/
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm3_masked_reconstruction"
PROBE_PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe"

DATA_DIR="$PROBE_PROJECT/datasets/human_virus/data/processed"
OUT_DIR="$PROJECT/results"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_TOKEN=$(cat ${HOME}/.cache/huggingface/token 2>/dev/null || true)


# Activate the project environment first: conda activate beyond-nativeness

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi not available)"
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

# ---- Run experiment ----
echo ""
echo "Starting masked reconstruction experiment..."
echo "  Data dir:   $DATA_DIR"
echo "  Output dir: $OUT_DIR"
echo "  HF cache:   $HF_CACHE"
echo ""

python "$PROJECT/scripts/run_masked_reconstruction.py" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$OUT_DIR" \
    --cache_dir  "$HF_CACHE" \
    --batch_size 4 \
    --mask_rate  0.15 \
    --n_seeds    3 \
    --device     cuda

echo ""
echo "======================================"
echo "Completed: $(date)"
echo "Output files:"
ls -lh "$OUT_DIR/" 2>/dev/null || echo "  (output dir empty)"
echo "======================================"
