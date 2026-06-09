#!/usr/bin/env bash
# ============================================================
# ESMC-600M zero-shot masked reconstruction PPL
# ============================================================

set -euo pipefail

# Activate the project environment first: conda activate beyond-nativeness

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repo root}/projects/esm_zeroshot_ppl"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_TOKEN=$(cat ${HOME}/.cache/huggingface/token 2>/dev/null || true)

# ---- Validate inputs ----
echo "Validating input files..."
for LABEL in viral nonviral; do
    for SPLIT in train val test; do
        FASTA="$PROJECT/data/${LABEL}_${SPLIT}.faa"
        if [ ! -f "$FASTA" ]; then
            echo "ERROR: Missing required file: $FASTA"
            exit 1
        fi
    done
done
echo "  All 6 FASTA splits found."

mkdir -p "$PROJECT/results/esmc_600m"

# ---- Run experiment ----
echo ""
echo "Starting ESMC-600M masked reconstruction..."
python "$PROJECT/scripts/run_masked_reconstruction_esmc.py" \
    --data_dir   "$PROJECT/data" \
    --out_dir    "$PROJECT/results/esmc_600m" \
    --model      esmc_600m \
    --cache_dir  "$HF_CACHE" \
    --batch_size 4 \
    --mask_rate  0.15 \
    --n_seeds    3 \
    --device     cuda

echo ""
echo "Completed."
ls -lh "$PROJECT/results/esmc_600m/" 2>/dev/null || echo "  (empty)"
