#!/usr/bin/env bash
# ============================================================
# ProtT5-XL-U50 (3B, T5 enc-dec) — encoder embeddings + span-denoising pseudo-PPL
# over the full nativeness pool. Fills the ProtT5 row of appfig_crossarch_pca_ppl.py (and appfig_family_nativization_nonesm.py).
# Requires a CUDA GPU. Override pool/limit for a smoke run:
#   POOLS=controls LIMIT=8 bash jobs/score_prott5_xl.sh
# ============================================================
set -euo pipefail
PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/cross_architecture_nativeness"
export HF_HOME="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/data/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
POOLS="${POOLS:-all}"
LIMIT="${LIMIT:-0}"

echo "pools=$POOLS limit=$LIMIT"
# Activate the project environment first: conda activate beyond-nativeness
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

cd "$PROJECT"
python scripts/score_prott5.py \
    --model_key  prott5_xl \
    --pools      "$POOLS" \
    --batch_size 8 \
    --mask_rate  0.15 \
    --n_seeds    3 \
    --max_len    1022 \
    --limit      "$LIMIT" \
    --device     cuda
echo "[$(date)] ProtT5 ($POOLS limit=$LIMIT) done."
