#!/usr/bin/env bash
# ============================================================
# EvoDiff OA-DM ELBO-vs-pooled rerun for the §4.2 per-family scaling.
# Re-scores the HUMAN pool only, emitting BOTH per-sequence perplexities from the
# same OADM draws: mean_perplexity (pooled, legacy) + mean_perplexity_elbo (the
# Hoogeboom OA-ARDM per-residue NLL bound EvoDiff is trained on). 24 seeds so the
# ELBO (an expectation over the masking timestep t) is well-converged; seeded
# np.random for reproducible masks. --ppl_only skips embeddings (scaling needs PPL).
# Writes to the *_elbo keys so the main full-pool rho(PC1,PPL) keys stay untouched.
#
# Submit (gpu_test QOS cap = 2 jobs, both fit):
#   sbatch --export=ALL,MODEL_KEY=evodiff_oadm_38m_elbo  jobs/score_evodiff_elbo.sh
#   sbatch --export=ALL,MODEL_KEY=evodiff_oadm_640m_elbo jobs/score_evodiff_elbo.sh
# Smoke:
#   sbatch --export=ALL,MODEL_KEY=evodiff_oadm_38m_elbo,LIMIT=16,NSEEDS=4 jobs/score_evodiff_elbo.sh
# ============================================================
set -euo pipefail
PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/cross_architecture_nativeness"
export HF_HOME="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/data/hf_cache"
MODEL_KEY="${MODEL_KEY:-}"
if [ -z "$MODEL_KEY" ]; then
    echo "ERROR: set MODEL_KEY=evodiff_oadm_38m_elbo or evodiff_oadm_640m_elbo" >&2
    exit 1
fi
POOLS="${POOLS:-human}"
NSEEDS="${NSEEDS:-24}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:-0}"

source "$PROJECT/envs/evodiff_venv/bin/activate"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

cd "$PROJECT"
python scripts/score_evodiff.py \
    --model_key  "$MODEL_KEY" \
    --pools      "$POOLS" \
    --batch_size 16 \
    --n_seeds    "$NSEEDS" \
    --seed       "$SEED" \
    --max_len    1022 \
    --ppl_only \
    --limit      "$LIMIT" \
    --device     cuda
echo "[$(date)] EvoDiff ELBO $MODEL_KEY ($POOLS n_seeds=$NSEEDS limit=$LIMIT) done."
