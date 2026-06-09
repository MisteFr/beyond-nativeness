#!/usr/bin/env bash
# ============================================================
# ProGen2-base (764M, autoregressive) — embeddings + TRUE causal PPL on the
# full nativeness pool (human virus, 8 phage/cellular OOD groups, 3 controls;
# ~66,790 seqs). Fills the ProGen2 row of paper_figures/appfig_crossarch_pca_ppl.py (and appfig_family_nativization_nonesm.py).
# Tests whether PC1≈PPL + the viral-OOD gradient generalize beyond masked-LM ESM.
# Requires a CUDA GPU.
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/cross_architecture_nativeness"
HF_CACHE="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/data/hf_cache"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

echo "Model: progen2_base (hugohrban/progen2-base)"

# Activate the project environment first: conda activate beyond-nativeness

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# --- validate a few representative input FASTAs ------------------------------
HV="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/datasets/human_virus/data/processed"
PH="${BEYOND_NATIVENESS_ROOT}/projects/prokaryote_phage_ood/data/processed"
SH="${BEYOND_NATIVENESS_ROOT}/projects/prokaryote_phage_ood/data/shuffled"
for F in "$HV/viral_train.faa" "$PH/archaea_clean.faa" "$SH/random_uniform.faa"; do
    [ -f "$F" ] || { echo "ERROR: Missing FASTA: $F"; exit 1; }
done
echo "  Input FASTAs found."

cd "$PROJECT"
python scripts/score_progen2.py \
    --model_key  progen2_base \
    --pools      all \
    --batch_size 8 \
    --max_len    1022 \
    --device     cuda

echo ""
echo "================================================"
echo "[$(date)] ProGen2-base nativeness outputs complete."
echo "Embeddings: $PROJECT/data/embeddings/progen2_base"
echo "PPL:        $PROJECT/results/progen2_base"
echo "================================================"
