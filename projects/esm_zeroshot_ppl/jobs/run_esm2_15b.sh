#!/usr/bin/env bash
# ============================================================
# ESM2-15B zero-shot masked reconstruction PPL + LL (bf16 weights)
#   Needs a large-memory GPU (~30 GB weights + activations, e.g. A100-80G/H200).
#   3 seeds x ~10.4k sequences at batch=1: expect ~18-24 h wall.
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

mkdir -p "$PROJECT/results/esm2_15b"

# ---- Run experiment ----
echo ""
echo "Starting ESM2-15B masked reconstruction (bf16 weights)..."
python "$PROJECT/scripts/run_masked_reconstruction_esm2.py" \
    --model      esm2_15b \
    --data_dir   "$PROJECT/data" \
    --out_dir    "$PROJECT/results/esm2_15b" \
    --cache_dir  "$HF_CACHE" \
    --batch_size 1 \
    --device     cuda \
    --dtype      bf16

echo ""
echo "Completed."
ls -lh "$PROJECT/results/esm2_15b/" 2>/dev/null || echo "  (empty)"
