#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# Step 2a: Extract embeddings for the random/shuffled control
#          sequences using GPU models: esm2_8m, esm2_650m, esm3_open
#
# Requires a GPU. esm3_open is gated and requires an HF token
# (accept the EvolutionaryScale license on HuggingFace once).
#
# Input:  data/{random_uniform,shuffled_viral,shuffled_nonviral}.faa
# Output: data/embeddings/{model}/{random_uniform,shuffled_viral,shuffled_nonviral}.npz
# ============================================================

# Activate the project environment first: conda activate beyond-nativeness

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repo root}/projects/esm_random_ood"
PROBE_PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe"
SCRIPTS="$PROJECT/scripts"
DATA="$PROJECT/data"

export TRANSFORMERS_CACHE="$PROBE_PROJECT/data/hf_cache"
export HF_HOME="$PROBE_PROJECT/data/hf_cache"
export HF_TOKEN=$(cat ${HOME}/.cache/huggingface/token 2>/dev/null || true)

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (not available)"

FASTAS=("random_uniform" "shuffled_viral" "shuffled_nonviral")

# ---------------------------------------------------------------------------
# ESM2 variants (HuggingFace)
# ---------------------------------------------------------------------------
declare -A ESM2_MODELS=(
    ["esm2_8m"]="facebook/esm2_t6_8M_UR50D"
    ["esm2_650m"]="facebook/esm2_t33_650M_UR50D"
)
declare -A ESM2_BATCH=(
    ["esm2_8m"]=64 ["esm2_650m"]=16
)

for MODEL_KEY in esm2_8m esm2_650m; do
    HF_MODEL="${ESM2_MODELS[$MODEL_KEY]}"
    BATCH="${ESM2_BATCH[$MODEL_KEY]}"
    OUT_DIR="$DATA/embeddings/$MODEL_KEY"
    mkdir -p "$OUT_DIR"

    for FASTA_NAME in "${FASTAS[@]}"; do
        OUT="$OUT_DIR/${FASTA_NAME}.npz"
        FASTA="$DATA/${FASTA_NAME}.faa"

        if [ -f "$OUT" ]; then
            echo "[$(date)] [$MODEL_KEY/$FASTA_NAME] Already exists — skipping"
            continue
        fi

        echo ""
        echo "========================================"
        echo "[$MODEL_KEY] $FASTA_NAME  batch=$BATCH"
        echo "========================================"
        python "$SCRIPTS/extract_embeddings.py" \
            --fasta      "$FASTA" \
            --outfile    "$OUT" \
            --model      "$HF_MODEL" \
            --cache_dir  "$TRANSFORMERS_CACHE" \
            --batch_size "$BATCH" \
            --device     cuda \
            --max_len    1022
        echo "[$(date)] [$MODEL_KEY/$FASTA_NAME] Done."; ls -lh "$OUT"
    done
done

# ---------------------------------------------------------------------------
# ESM3-open (gated, requires HF token)
# ---------------------------------------------------------------------------
echo ""
echo "Checking EvolutionaryScale ESM package..."
if ! python -c "from esm.models.esm3 import ESM3" 2>/dev/null; then
    pip install esm httpx --quiet
fi

MODEL_KEY="esm3_open"
OUT_DIR="$DATA/embeddings/$MODEL_KEY"
mkdir -p "$OUT_DIR"

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set — skipping esm3_open"
else
    for FASTA_NAME in "${FASTAS[@]}"; do
        OUT="$OUT_DIR/${FASTA_NAME}.npz"
        FASTA="$DATA/${FASTA_NAME}.faa"

        if [ -f "$OUT" ]; then
            echo "[$(date)] [$MODEL_KEY/$FASTA_NAME] Already exists — skipping"
            continue
        fi

        echo ""
        echo "========================================"
        echo "[$MODEL_KEY] $FASTA_NAME  batch=2"
        echo "========================================"
        python "$SCRIPTS/extract_embeddings_esm3.py" \
            --fasta      "$FASTA" \
            --outfile    "$OUT" \
            --model      "esm3-open" \
            --cache_dir  "$HF_HOME" \
            --batch_size 2 \
            --device     cuda \
            --max_len    1022
        echo "[$(date)] [$MODEL_KEY/$FASTA_NAME] Done."; ls -lh "$OUT"
    done
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "[$(date)] All GPU embedding extraction complete."
echo ""
echo "Summary:"
for MODEL_KEY in esm2_8m esm2_650m esm3_open; do
    for FASTA_NAME in "${FASTAS[@]}"; do
        OUT="$DATA/embeddings/$MODEL_KEY/${FASTA_NAME}.npz"
        if [ -f "$OUT" ]; then
            python3 -c "
import numpy as np
d = np.load('$OUT', allow_pickle=True)
print(f'  $MODEL_KEY/$FASTA_NAME: {d[\"embeddings\"].shape}')
"
        else
            echo "  $MODEL_KEY/$FASTA_NAME: MISSING"
        fi
    done
done
