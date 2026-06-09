#!/usr/bin/env bash
# ============================================================
# EvoDiff OA-DM-640M (discrete diffusion, OADM) — pre-decoder embeddings +
# order-agnostic masked-reconstruction pseudo-PPL over the full nativeness pool.
# Fills the EvoDiff row of paper_figures/appfig_crossarch_pca_ppl.py (and appfig_family_nativization_nonesm.py).
# Runs in the dedicated evodiff_venv (evodiff deps conflict with the main beyond-nativeness env).
# Smoke:  sbatch --export=ALL,POOLS=controls,LIMIT=16 jobs/score_evodiff_640m.sh
# ============================================================
set -euo pipefail
PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/cross_architecture_nativeness"
export HF_HOME="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/data/hf_cache"
POOLS="${POOLS:-all}"
LIMIT="${LIMIT:-0}"

source "$PROJECT/envs/evodiff_venv/bin/activate"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

cd "$PROJECT"
python scripts/score_evodiff.py \
    --model_key  evodiff_oadm_640m \
    --pools      "$POOLS" \
    --batch_size 16 \
    --n_seeds    5 \
    --max_len    1022 \
    --limit      "$LIMIT" \
    --device     cuda
echo "[$(date)] EvoDiff ($POOLS limit=$LIMIT) done."
