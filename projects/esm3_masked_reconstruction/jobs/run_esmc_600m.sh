#!/usr/bin/env bash
# ============================================================
# ESMC 600M masked token reconstruction experiment (local GPU)
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm3_masked_reconstruction"

DATA_DIR="$PROJECT/data"
OUT_DIR="$PROJECT/results/esmc_600m"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_TOKEN=$(cat ${HOME}/.cache/huggingface/token 2>/dev/null || true)


# Activate the project environment first: conda activate beyond-nativeness

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi not available)"

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
echo "Starting ESMC 600M masked reconstruction..."
python "$PROJECT/scripts/run_masked_reconstruction_esmc.py" \
    --data_dir   "$DATA_DIR" \
    --out_dir    "$OUT_DIR" \
    --model      esmc_600m \
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
