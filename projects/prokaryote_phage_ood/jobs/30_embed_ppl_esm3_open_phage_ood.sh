#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ESM3-open embeddings + masked-reconstruction PPL on the 8
# tree-of-life groups in prokaryote_phage_ood.
# Activate the project environment first: conda activate beyond-nativeness
# Requires a GPU.
# ============================================================

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/prokaryote_phage_ood"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

EMBED_SCRIPT="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/scripts/extract_embeddings_esm3.py"
PPL_SCRIPT="${BEYOND_NATIVENESS_ROOT}/projects/esm3_masked_reconstruction/scripts/run_masked_reconstruction_single_fasta.py"

MODEL_KEY="esm3_open"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

echo "================================================"
echo "Model: ESM3-open (~1.4B)"
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
        --model      esm3-open \
        --cache_dir  "$HF_CACHE" \
        --batch_size 2 \
        --device     cuda \
        --max_len    1022
done

echo ""
echo "=========  Phase 2: Masked-reconstruction PPL  ========="
for POOL in "${POOLS[@]}"; do
    FASTA="$PROJECT/data/processed/${POOL}_clean.faa"
    OUT_DIR="$PPL_OUT/${POOL}"
    OUT="$OUT_DIR/per_sequence_results.tsv"
    mkdir -p "$OUT_DIR"
    if [ -f "$OUT" ]; then
        echo "[$(date)] $POOL PPL already exists — skipping"
        continue
    fi
    echo ""
    echo "[$(date)] PPL $POOL  ($(grep -c '^>' "$FASTA") seqs)"
    python "$PPL_SCRIPT" \
        --fasta      "$FASTA" \
        --label      "$POOL" \
        --out_dir    "$OUT_DIR" \
        --backend    esm3_open \
        --cache_dir  "$HF_CACHE" \
        --batch_size 1
done

echo ""
echo "================================================"
echo "[$(date)] All ESM3-open outputs complete."
echo "Embeddings: $EMB_OUT"
echo "PPL:        $PPL_OUT"
echo "================================================"
