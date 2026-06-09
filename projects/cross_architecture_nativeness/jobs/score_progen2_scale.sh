#!/usr/bin/env bash
# ============================================================
# ProGen2 per-family SCALING run (autoregressive, true causal PPL).
# Parametrized over the ProGen2 ladder for the §4.2 scaling figure:
#   MODEL_KEY in {progen2_small (151M), progen2_large (2.7B), progen2_xlarge (6.4B)}
#   (progen2_base 764M already scored on the full pool — reuse its human TSV.)
# POOLS=human only: the scaling figure needs the human pool (5,203 viral families +
# 5,197 cellular reference for the native-like threshold tau) — ~6x cheaper than 'all'.
# Requires a CUDA GPU.
# Run, e.g.:
#   MODEL_KEY=progen2_small  BATCH=8  bash jobs/score_progen2_scale.sh
#   MODEL_KEY=progen2_large  BATCH=8  bash jobs/score_progen2_scale.sh
#   MODEL_KEY=progen2_xlarge BATCH=4  bash jobs/score_progen2_scale.sh
# ============================================================
set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/cross_architecture_nativeness"
HF_CACHE="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/data/hf_cache"

MODEL_KEY="${MODEL_KEY:?set MODEL_KEY to progen2_small or progen2_large or progen2_xlarge}"
POOLS="${POOLS:-human}"
BATCH="${BATCH:-8}"
PPL_ONLY="${PPL_ONLY:-}"            # set to 1 to skip embeddings (PPL only; big models)

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM

echo "Model: $MODEL_KEY | pools=$POOLS | batch=$BATCH"

# Activate the project environment first: conda activate beyond-nativeness
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# --- validate input FASTAs (human pool) -------------------------------------
HV="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/datasets/human_virus/data/processed"
for F in "$HV/viral_train.faa" "$HV/viral_test.faa" "$HV/nonviral_train.faa"; do
    [ -f "$F" ] || { echo "ERROR: Missing FASTA: $F"; exit 1; }
done
echo "  Input FASTAs found."

cd "$PROJECT"
python scripts/score_progen2.py \
    --model_key  "$MODEL_KEY" \
    --pools      "$POOLS" \
    --batch_size "$BATCH" \
    --max_len    1022 \
    --device     cuda \
    ${PPL_ONLY:+--ppl_only}

echo ""
echo "================================================"
echo "[$(date)] ProGen2 $MODEL_KEY ($POOLS) outputs complete."
echo "PPL: $PROJECT/results/$MODEL_KEY/per_sequence_results.tsv"
echo "================================================"
