#!/usr/bin/env bash
# ============================================================
# EvoDiff OA-DM-38M (discrete diffusion, OADM) — small end of the EvoDiff scaling
# ladder for the §4.2 per-family scaling figure (640M already scored on the full pool).
# POOLS=human only: scaling needs the human pool (viral families + cellular tau ref).
# Requires a CUDA GPU and the dedicated EvoDiff venv (its deps conflict with the
# main beyond-nativeness env; see README "EvoDiff environment").
# Run:    bash jobs/score_evodiff_38m.sh
# Smoke:  POOLS=human LIMIT=16 bash jobs/score_evodiff_38m.sh
# ============================================================
set -euo pipefail
PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/cross_architecture_nativeness"
export HF_HOME="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/data/hf_cache"
MODEL_KEY="${MODEL_KEY:-evodiff_oadm_38m}"
POOLS="${POOLS:-human}"
LIMIT="${LIMIT:-0}"

echo "model=$MODEL_KEY pools=$POOLS limit=$LIMIT"
# Activate the EvoDiff venv first (see README): source "$PROJECT/envs/evodiff_venv/bin/activate"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

cd "$PROJECT"
python scripts/score_evodiff.py \
    --model_key  "$MODEL_KEY" \
    --pools      "$POOLS" \
    --batch_size 16 \
    --n_seeds    5 \
    --max_len    1022 \
    --limit      "$LIMIT" \
    --device     cuda
echo "[$(date)] EvoDiff $MODEL_KEY ($POOLS limit=$LIMIT) done."
