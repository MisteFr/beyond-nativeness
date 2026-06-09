#!/usr/bin/env bash
# ============================================================
# ESM2-650M masked reconstruction on shuffled/random controls.
# Three FASTAs: shuffled_viral, shuffled_nonviral, random_uniform.
# Writes TSVs under results/esm2_650m/{group}/per_sequence_results.tsv
# (per-group OOD PPLs used by the family-nativization figures).
# ============================================================

set -euo pipefail

# Activate the project environment first: conda activate beyond-nativeness

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repo root}/projects/esm_zeroshot_ppl"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

# Input FASTAs (synthetic control pools produced by the esm_random_ood module)
SHUF_V="${BEYOND_NATIVENESS_ROOT}/projects/esm_random_ood/data/shuffled_viral.faa"
SHUF_N="${BEYOND_NATIVENESS_ROOT}/projects/esm_random_ood/data/shuffled_nonviral.faa"
RAND_U="${BEYOND_NATIVENESS_ROOT}/projects/esm_random_ood/data/random_uniform.faa"

for F in "$SHUF_V" "$SHUF_N" "$RAND_U"; do
    [ -f "$F" ] || { echo "ERROR: Missing FASTA: $F"; exit 1; }
done
echo "  All 3 FASTAs found."

OUT_BASE="$PROJECT/results/esm2_650m"
mkdir -p "$OUT_BASE/shuffled_viral" "$OUT_BASE/shuffled_nonviral" "$OUT_BASE/random_uniform"

# --- Run three groups sequentially -------------------------------------------
echo ""
echo "[1/3] shuffled_viral ..."
python "$PROJECT/scripts/run_masked_reconstruction_esm2_oodgroups.py" \
    --model      esm2_650m \
    --fasta      "$SHUF_V" \
    --label      shuffled_viral \
    --out_tsv    "$OUT_BASE/shuffled_viral/per_sequence_results.tsv" \
    --cache_dir  "$HF_CACHE" \
    --batch_size 4

echo ""
echo "[2/3] shuffled_nonviral ..."
python "$PROJECT/scripts/run_masked_reconstruction_esm2_oodgroups.py" \
    --model      esm2_650m \
    --fasta      "$SHUF_N" \
    --label      shuffled_nonviral \
    --out_tsv    "$OUT_BASE/shuffled_nonviral/per_sequence_results.tsv" \
    --cache_dir  "$HF_CACHE" \
    --batch_size 4

echo ""
echo "[3/3] random_uniform ..."
python "$PROJECT/scripts/run_masked_reconstruction_esm2_oodgroups.py" \
    --model      esm2_650m \
    --fasta      "$RAND_U" \
    --label      random \
    --out_tsv    "$OUT_BASE/random_uniform/per_sequence_results.tsv" \
    --cache_dir  "$HF_CACHE" \
    --batch_size 4

echo ""
echo "Completed."
ls -lh "$OUT_BASE/shuffled_viral/" "$OUT_BASE/shuffled_nonviral/" "$OUT_BASE/random_uniform/" 2>/dev/null
