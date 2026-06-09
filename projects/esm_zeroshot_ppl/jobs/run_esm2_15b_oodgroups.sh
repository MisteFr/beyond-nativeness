#!/usr/bin/env bash
# ============================================================
# ESM2-15B embeddings + masked-reconstruction PPL on the three
# synthetic controls (shuffled_viral, shuffled_nonviral, random_uniform).
# Biggest-model counterpart to run_esm2_650m_oodgroups.sh; outputs fill
# the control rows of the per-group OOD PPLs used by the figures.
#
# bf16 weights (~30 GB) — needs a large-memory GPU (A100-80G / H200).
# ============================================================

set -euo pipefail

# Activate the project environment first: conda activate beyond-nativeness

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repo root}/projects/esm_zeroshot_ppl"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
EMB_ROOT="${BEYOND_NATIVENESS_ROOT}/projects/esm_random_ood/data/embeddings"

EMBED_SCRIPT="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/scripts/extract_embeddings.py"
PPL_SCRIPT="$PROJECT/scripts/run_masked_reconstruction_esm2_oodgroups.py"

HF_MODEL="facebook/esm2_t48_15B_UR50D"
MODEL_KEY="esm2_15b"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

echo "Model: $MODEL_KEY ($HF_MODEL)  dtype=bf16"

# Input FASTAs (synthetic control pools produced by the esm_random_ood module)
SHUF_V="${BEYOND_NATIVENESS_ROOT}/projects/esm_random_ood/data/shuffled_viral.faa"
SHUF_N="${BEYOND_NATIVENESS_ROOT}/projects/esm_random_ood/data/shuffled_nonviral.faa"
RAND_U="${BEYOND_NATIVENESS_ROOT}/projects/esm_random_ood/data/random_uniform.faa"

for F in "$SHUF_V" "$SHUF_N" "$RAND_U"; do
    [ -f "$F" ] || { echo "ERROR: Missing FASTA: $F"; exit 1; }
done
echo "  All 3 FASTAs found."

EMB_OUT="$EMB_ROOT/$MODEL_KEY"
PPL_OUT="$PROJECT/results/$MODEL_KEY"
mkdir -p "$EMB_OUT" "$PPL_OUT/shuffled_viral" "$PPL_OUT/shuffled_nonviral" "$PPL_OUT/random_uniform"

# --- Phase 1: Embeddings -----------------------------------------------------
echo ""
echo "=========  Phase 1: Embeddings  ========="
for PAIR in "shuffled_viral:$SHUF_V" "shuffled_nonviral:$SHUF_N" "random_uniform:$RAND_U"; do
    NAME="${PAIR%%:*}"
    FASTA="${PAIR##*:}"
    OUT="$EMB_OUT/${NAME}.npz"
    if [ -f "$OUT" ]; then
        echo "$NAME embeddings already exist — skipping"
        continue
    fi
    echo ""
    echo "Embedding $NAME  ($(grep -c '^>' "$FASTA") seqs)"
    python "$EMBED_SCRIPT" \
        --fasta      "$FASTA" \
        --outfile    "$OUT" \
        --model      "$HF_MODEL" \
        --cache_dir  "$HF_CACHE" \
        --batch_size 1 \
        --device     cuda \
        --max_len    1022 \
        --dtype      bf16
done

# --- Phase 2: Masked-reconstruction PPL --------------------------------------
echo ""
echo "=========  Phase 2: PPL  ========="
for TRIPLE in "shuffled_viral:$SHUF_V:shuffled_viral" \
              "shuffled_nonviral:$SHUF_N:shuffled_nonviral" \
              "random_uniform:$RAND_U:random"; do
    IFS=':' read -r NAME FASTA LABEL <<<"$TRIPLE"
    OUT="$PPL_OUT/${NAME}/per_sequence_results.tsv"
    if [ -f "$OUT" ]; then
        echo "$NAME PPL already exists — skipping"
        continue
    fi
    echo ""
    echo "PPL $NAME  ($(grep -c '^>' "$FASTA") seqs)"
    python "$PPL_SCRIPT" \
        --model      "$MODEL_KEY" \
        --fasta      "$FASTA" \
        --label      "$LABEL" \
        --out_tsv    "$OUT" \
        --cache_dir  "$HF_CACHE" \
        --batch_size 1 \
        --dtype      bf16
done

echo ""
echo "ESM2-15B control outputs complete."
echo "Embeddings: $EMB_OUT"
echo "PPL:        $PPL_OUT"
