#!/usr/bin/env bash
# ============================================================
# Human Virus experiment — Step 2c: ESM3-small + ESM3-medium (Forge API)
#
# Extracts embeddings via EvolutionaryScale Forge API (no local GPU needed):
#   esm3_small   (1536-dim)  esm3-small-2024-08
#   esm3_medium  (1536-dim)  esm3-medium-2024-08
#
# Checkpointing: each split writes progress to <outfile>.cache.npy
# — safe to re-run if interrupted.
#
# Input:  datasets/human_virus/data/processed/
# Output: datasets/human_virus/data/embeddings/{model_key}/
#
# Run after hv01_preprocess.sh. Requires a Forge API key in FORGE_TOKEN.
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repository root}/projects/esm_viral_probe"
SCRIPTS="$PROJECT/scripts"
PROC_DIR="$PROJECT/datasets/human_virus/data/processed"
FORGE_TOKEN="${FORGE_TOKEN:?Set FORGE_TOKEN to your EvolutionaryScale Forge API key}"

# Activate the project environment first: conda activate beyond-nativeness

SPLITS=(viral_train nonviral_train viral_val nonviral_val viral_test nonviral_test)

for MODEL_KEY in esm3_small esm3_medium; do
    if [ "$MODEL_KEY" = "esm3_small" ]; then
        FORGE_MODEL="esm3-small-2024-08"
    else
        FORGE_MODEL="esm3-medium-2024-08"
    fi

    EMB_DIR="$PROJECT/datasets/human_virus/data/embeddings/$MODEL_KEY"
    mkdir -p "$EMB_DIR"

    echo ""
    echo "========================================"
    echo "Model: $MODEL_KEY  ($FORGE_MODEL)"
    echo "  Output: $EMB_DIR"
    echo "========================================"

    for SPLIT in "${SPLITS[@]}"; do
        FASTA="$PROC_DIR/${SPLIT}.faa"
        OUTFILE="$EMB_DIR/${SPLIT}.npz"

        if [ ! -f "$FASTA" ]; then
            echo "ERROR: Missing input: $FASTA"
            exit 1
        fi

        if [ -f "$OUTFILE" ]; then
            echo "[$(date)] [$MODEL_KEY/$SPLIT] Already complete — skipping."
            continue
        fi

        echo ""
        echo "[$(date)] [$MODEL_KEY/$SPLIT] Starting..."
        python "$SCRIPTS/extract_embeddings_forge.py" \
            --fasta   "$FASTA" \
            --outfile "$OUTFILE" \
            --token   "$FORGE_TOKEN" \
            --model   "$FORGE_MODEL"

        echo "[$(date)] [$MODEL_KEY/$SPLIT] Done."
        ls -lh "$OUTFILE"
    done

    echo ""
    echo "[$MODEL_KEY] Embedding files:"
    ls -lh "$EMB_DIR/"
done

echo ""
echo "[$(date)] Forge embeddings complete."

echo ""
echo "Verifying shapes:"
for MODEL_KEY in esm3_small esm3_medium; do
    EMB_DIR="$PROJECT/datasets/human_virus/data/embeddings/$MODEL_KEY"
    python3 -c "
import numpy as np, os
emb_dir = '$EMB_DIR'
for f in sorted(os.listdir(emb_dir)):
    if f.endswith('.npz'):
        d = np.load(os.path.join(emb_dir, f), allow_pickle=True)
        print(f'  $MODEL_KEY/{f}: {d[\"embeddings\"].shape}')
"
done
