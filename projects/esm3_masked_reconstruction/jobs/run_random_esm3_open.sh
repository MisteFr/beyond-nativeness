#!/usr/bin/env bash
# ============================================================
# Random uniform — ESM3 Open masked reconstruction (local GPU)
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm3_masked_reconstruction"
FASTA="${BEYOND_NATIVENESS_ROOT}/projects/esm_random_ood/data/random_uniform.faa"
OUT_DIR="$PROJECT/results/random_uniform"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_TOKEN=$(cat ${HOME}/.cache/huggingface/token 2>/dev/null || true)


# Activate the project environment first: conda activate beyond-nativeness

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi not available)"

# ---- Validate input ----
if [ ! -f "$FASTA" ]; then
    echo "ERROR: Missing FASTA: $FASTA"
    exit 1
fi
echo "Input: $FASTA ($(wc -l < "$FASTA") lines)"

mkdir -p "$OUT_DIR"

# ---- Run ----
python "$PROJECT/scripts/run_masked_reconstruction_single_fasta.py" \
    --fasta     "$FASTA" \
    --label     random \
    --out_dir   "$OUT_DIR" \
    --backend   esm3_open \
    --cache_dir "$HF_CACHE" \
    --batch_size 2 \
    --mask_rate 0.15 \
    --n_seeds   3 \
    --device    cuda

echo ""
echo "======================================"
echo "Completed: $(date)"
echo "Output:"
ls -lh "$OUT_DIR/" 2>/dev/null || echo "  (empty)"
echo "======================================"
