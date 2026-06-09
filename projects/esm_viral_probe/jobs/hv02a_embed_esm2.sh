#!/usr/bin/env bash
# ============================================================
# Human Virus experiment — Step 2a: ESM2 embeddings  (requires GPU)
#
# Extracts mean-pool embeddings for ESM2 variants:
#   esm2_8m    (320-dim)  esm2_35m   (480-dim)
#   esm2_150m  (640-dim)  esm2_650m  (1280-dim)
#
# Input:  datasets/human_virus/data/processed/
# Output: datasets/human_virus/data/embeddings/{model_key}/
#
# Run after hv01_preprocess.sh.
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repository root}/projects/esm_viral_probe"
PROC_DIR="$PROJECT/datasets/human_virus/data/processed"

export TRANSFORMERS_CACHE="$PROJECT/data/hf_cache"
export HF_HOME="$PROJECT/data/hf_cache"
mkdir -p "$TRANSFORMERS_CACHE"

# Activate the project environment first: conda activate beyond-nativeness

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

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

# ---- Model definitions: key, HF-ID, batch_size ----
declare -A MODELS
MODELS["esm2_8m"]="facebook/esm2_t6_8M_UR50D"
MODELS["esm2_35m"]="facebook/esm2_t12_35M_UR50D"
MODELS["esm2_150m"]="facebook/esm2_t30_150M_UR50D"
MODELS["esm2_650m"]="facebook/esm2_t33_650M_UR50D"

declare -A BATCH_SIZES
BATCH_SIZES["esm2_8m"]=64
BATCH_SIZES["esm2_35m"]=48
BATCH_SIZES["esm2_150m"]=32
BATCH_SIZES["esm2_650m"]=16

for MODEL_KEY in esm2_8m esm2_35m esm2_150m esm2_650m; do
    HF_MODEL="${MODELS[$MODEL_KEY]}"
    BATCH="${BATCH_SIZES[$MODEL_KEY]}"
    EMB_DIR="$PROJECT/datasets/human_virus/data/embeddings/$MODEL_KEY"
    mkdir -p "$EMB_DIR"

    echo ""
    echo "========================================"
    echo "Model: $MODEL_KEY  ($HF_MODEL)"
    echo "  Batch size: $BATCH  |  Output: $EMB_DIR"
    echo "========================================"

    for LABEL in viral nonviral; do
        for SPLIT in train val test; do
            FASTA="$PROC_DIR/${LABEL}_${SPLIT}.faa"
            OUT="$EMB_DIR/${LABEL}_${SPLIT}.npz"

            if [ -f "$OUT" ]; then
                echo "[$(date)] Already exists: $OUT — skipping"
                continue
            fi

            echo ""
            echo "[$(date)] Extracting: $MODEL_KEY / $LABEL / $SPLIT"
            python "$PROJECT/scripts/extract_embeddings.py" \
                --fasta      "$FASTA" \
                --outfile    "$OUT" \
                --model      "$HF_MODEL" \
                --cache_dir  "$TRANSFORMERS_CACHE" \
                --batch_size "$BATCH" \
                --device     cuda \
                --max_len    1022
        done
    done

    echo ""
    echo "[$MODEL_KEY] Embedding files:"
    ls -lh "$EMB_DIR/"
done

echo ""
echo "[$(date)] All ESM2 embeddings complete."
