#!/usr/bin/env bash
# ============================================================
# Human Virus experiment — Step 2b: ESMC-300M + ESM3-open embeddings  (requires GPU)
#
# Extracts mean-pool embeddings for:
#   esmc_300m  (960-dim)   via EvolutionaryScale esm package
#   esm3_open  (1536-dim)  via EvolutionaryScale esm package (gated; needs HF_TOKEN)
#
# Input:  datasets/human_virus/data/processed/
# Output: datasets/human_virus/data/embeddings/{model_key}/
#
# Run after hv01_preprocess.sh.
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repository root}/projects/esm_viral_probe"
PROC_DIR="$PROJECT/datasets/human_virus/data/processed"

export HF_HOME="$PROJECT/data/hf_cache"
export TRANSFORMERS_CACHE="$PROJECT/data/hf_cache"
# esm3_open is gated on Hugging Face: set HF_TOKEN in the environment (falls back to the HF CLI cache).
export HF_TOKEN="${HF_TOKEN:-$(cat "${HF_HOME}/token" 2>/dev/null || true)}"
mkdir -p "$HF_HOME"

# Activate the project environment first: conda activate beyond-nativeness

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

# ---- Install ESM package ----
echo ""
echo "Checking EvolutionaryScale ESM package..."
if ! python -c "from esm.models.esmc import ESMC" 2>/dev/null; then
    echo "  Installing EvolutionaryScale esm + httpx..."
    pip install esm httpx --quiet
fi
echo "  ESM package: $(python -c 'import esm; print(esm.__version__)')"

# ---- Validate inputs ----
echo ""
echo "Validating input files..."
for LABEL in viral nonviral; do
    for SPLIT in train val test; do
        FASTA="$PROC_DIR/${LABEL}_${SPLIT}.faa"
        if [ ! -f "$FASTA" ]; then
            echo "ERROR: Missing required file: $FASTA"
            exit 1
        fi
    done
done
echo "  All 6 FASTA splits found."

# ─────────────────────────────────────────────────────────────────────────────
# ESMC-300M
# ─────────────────────────────────────────────────────────────────────────────
MODEL_KEY="esmc_300m"
ESMC_MODEL="esmc_300m"
BATCH=16
EMB_DIR="$PROJECT/datasets/human_virus/data/embeddings/$MODEL_KEY"
mkdir -p "$EMB_DIR"

echo ""
echo "========================================"
echo "Model: $MODEL_KEY  (batch=$BATCH)"
echo "========================================"

for LABEL in viral nonviral; do
    for SPLIT in train val test; do
        FASTA="$PROC_DIR/${LABEL}_${SPLIT}.faa"
        OUT="$EMB_DIR/${LABEL}_${SPLIT}.npz"

        if [ -f "$OUT" ]; then
            echo "[$(date)] Already exists: $OUT — skipping"
            continue
        fi

        echo "[$(date)] Extracting: $MODEL_KEY / $LABEL / $SPLIT"
        python "$PROJECT/scripts/extract_embeddings_esmc.py" \
            --fasta      "$FASTA" \
            --outfile    "$OUT" \
            --model      "$ESMC_MODEL" \
            --cache_dir  "$HF_HOME" \
            --batch_size "$BATCH" \
            --device     cuda \
            --max_len    1022
    done
done
echo "[$(date)] $MODEL_KEY complete."; ls -lh "$EMB_DIR/"

# ─────────────────────────────────────────────────────────────────────────────
# ESM3-open (~1.4B params; batch=2 to fit MIG A100)
# ─────────────────────────────────────────────────────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "WARNING: HF_TOKEN not found — skipping esm3_open (gated model)."
else
    MODEL_KEY="esm3_open"
    ESM3_MODEL="esm3-open"
    BATCH=2
    EMB_DIR="$PROJECT/datasets/human_virus/data/embeddings/$MODEL_KEY"
    mkdir -p "$EMB_DIR"

    echo ""
    echo "========================================"
    echo "Model: $MODEL_KEY  (batch=$BATCH)"
    echo "========================================"

    for LABEL in viral nonviral; do
        for SPLIT in train val test; do
            FASTA="$PROC_DIR/${LABEL}_${SPLIT}.faa"
            OUT="$EMB_DIR/${LABEL}_${SPLIT}.npz"

            if [ -f "$OUT" ]; then
                echo "[$(date)] Already exists: $OUT — skipping"
                continue
            fi

            echo "[$(date)] Extracting: $MODEL_KEY / $LABEL / $SPLIT"
            python "$PROJECT/scripts/extract_embeddings_esm3.py" \
                --fasta      "$FASTA" \
                --outfile    "$OUT" \
                --model      "$ESM3_MODEL" \
                --cache_dir  "$HF_HOME" \
                --batch_size "$BATCH" \
                --device     cuda \
                --max_len    1022
        done
    done
    echo "[$(date)] $MODEL_KEY complete."; ls -lh "$EMB_DIR/"
fi

echo ""
echo "[$(date)] ESMC / ESM3-open embeddings complete."
