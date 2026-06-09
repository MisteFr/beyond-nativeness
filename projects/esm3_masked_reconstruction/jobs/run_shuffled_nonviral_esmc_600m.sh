#!/usr/bin/env bash
# ============================================================
# Shuffled nonviral — ESMC 600M masked reconstruction (local GPU)
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm3_masked_reconstruction"
FASTA="${BEYOND_NATIVENESS_ROOT}/projects/esm_random_ood/data/shuffled_nonviral.faa"
OUT_DIR="$PROJECT/results/esmc_600m/shuffled_nonviral"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_TOKEN=$(cat ${HOME}/.cache/huggingface/token 2>/dev/null || true)


# Activate the project environment first: conda activate beyond-nativeness

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi not available)"

if [ ! -f "$FASTA" ]; then
    echo "ERROR: Missing FASTA: $FASTA"
    exit 1
fi
echo "Input: $FASTA ($(wc -l < "$FASTA") lines)"

mkdir -p "$OUT_DIR"

python "$PROJECT/scripts/run_masked_reconstruction_single_fasta.py" \
    --fasta     "$FASTA" \
    --label     shuffled_nonviral \
    --out_dir   "$OUT_DIR" \
    --backend   esmc_local \
    --model     esmc_600m \
    --cache_dir "$HF_CACHE" \
    --batch_size 4 \
    --mask_rate 0.15 \
    --n_seeds   3 \
    --device    cuda

echo ""
echo "======================================"
echo "Completed: $(date)"
echo "Output:"
ls -lh "$OUT_DIR/" 2>/dev/null || echo "  (empty)"
echo "======================================"
