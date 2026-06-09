#!/usr/bin/env bash
set -euo pipefail

# Activate the project environment first: conda activate beyond-nativeness
# Requires a GPU.

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/prokaryote_phage_ood"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

cd "$PROJECT"

echo "=== Masked reconstruction: plants ==="
python scripts/04_masked_reconstruction.py \
    --fasta     data/processed/plants_clean.faa \
    --label     plants \
    --out_dir   results/masked_reconstruction \
    --model     esmc_600m \
    --cache_dir "$HF_CACHE" \
    --batch_size 4 \
    --device    cuda

echo "=== Done ==="
