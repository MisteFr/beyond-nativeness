#!/usr/bin/env bash
# ============================================================
# Human Virus experiment — Step 2a2: ESM2-3B mean-pool embeddings  (requires GPU)
#   facebook/esm2_t36_3B_UR50D (3B params, 2560-dim)
#   Loaded in bf16 to keep VRAM low; outputs cast to fp32 for probe.
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repository root}/projects/esm_viral_probe"
PROC_DIR="$PROJECT/datasets/human_virus/data/processed"
MODEL_KEY="esm2_3b"
HF_MODEL="facebook/esm2_t36_3B_UR50D"
BATCH=4
EMB_DIR="$PROJECT/datasets/human_virus/data/embeddings/$MODEL_KEY"

export TRANSFORMERS_CACHE="$PROJECT/data/hf_cache"
export HF_HOME="$PROJECT/data/hf_cache"
mkdir -p "$EMB_DIR" "$TRANSFORMERS_CACHE"

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

echo ""
echo "Model: $MODEL_KEY  ($HF_MODEL)"
echo "  Batch size: $BATCH  |  dtype: bf16  |  Output: $EMB_DIR"

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
            --max_len    1022 \
            --dtype      bf16
    done
done

echo ""
echo "[$(date)] ESM2-3B human_virus embeddings complete."
echo "Embedding files:"
ls -lh "$EMB_DIR/"
