#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ESM2-650M embeddings + masked-reconstruction PPL on the 8
# tree-of-life groups in prokaryote_phage_ood.
# Activate the project environment first: conda activate beyond-nativeness
# Requires a GPU.
# ============================================================

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/prokaryote_phage_ood"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

EMBED_SCRIPT="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/scripts/extract_embeddings.py"
PPL_SCRIPT="${BEYOND_NATIVENESS_ROOT}/projects/esm_zeroshot_ppl/scripts/run_masked_reconstruction_esm2_oodgroups.py"

HF_MODEL="facebook/esm2_t33_650M_UR50D"
MODEL_KEY="esm2_650m"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

echo "================================================"
echo "Model: $MODEL_KEY ($HF_MODEL)"
echo "================================================"

POOLS=(archaea bacteria fungi insects plants phage plant_virus invertebrate_virus)

EMB_OUT="$PROJECT/data/embeddings/$MODEL_KEY"
PPL_OUT="$PROJECT/results/masked_reconstruction_$MODEL_KEY"
mkdir -p "$EMB_OUT" "$PPL_OUT"

echo ""
echo "=========  Phase 1: Embeddings  ========="
for POOL in "${POOLS[@]}"; do
    FASTA="$PROJECT/data/processed/${POOL}_clean.faa"
    OUT="$EMB_OUT/${POOL}.npz"
    if [ ! -f "$FASTA" ]; then
        echo "ERROR: missing FASTA $FASTA"; exit 1
    fi
    if [ -f "$OUT" ]; then
        echo "[$(date)] $POOL embeddings already exist — skipping"
        continue
    fi
    echo ""
    echo "[$(date)] Embedding $POOL  ($(grep -c '^>' "$FASTA") seqs)"
    python "$EMBED_SCRIPT" \
        --fasta      "$FASTA" \
        --outfile    "$OUT" \
        --model      "$HF_MODEL" \
        --cache_dir  "$HF_CACHE" \
        --batch_size 16 \
        --device     cuda \
        --max_len    1022
done

echo ""
echo "=========  Phase 2: Masked-reconstruction PPL  ========="
for POOL in "${POOLS[@]}"; do
    FASTA="$PROJECT/data/processed/${POOL}_clean.faa"
    OUT="$PPL_OUT/${POOL}/per_sequence_results.tsv"
    mkdir -p "$(dirname "$OUT")"
    if [ -f "$OUT" ]; then
        echo "[$(date)] $POOL PPL already exists — skipping"
        continue
    fi
    echo ""
    echo "[$(date)] PPL $POOL  ($(grep -c '^>' "$FASTA") seqs)"
    python "$PPL_SCRIPT" \
        --model      "$MODEL_KEY" \
        --fasta      "$FASTA" \
        --label      "$POOL" \
        --out_tsv    "$OUT" \
        --cache_dir  "$HF_CACHE" \
        --batch_size 4
done

echo ""
echo "================================================"
echo "[$(date)] All ESM2-650M outputs complete."
echo "Embeddings: $EMB_OUT"
echo "PPL:        $PPL_OUT"
echo "================================================"
