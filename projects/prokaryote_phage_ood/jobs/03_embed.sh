#!/usr/bin/env bash
set -euo pipefail

# Activate the project environment first: conda activate beyond-nativeness
# Requires a GPU.

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/prokaryote_phage_ood"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
# Reuse the existing embedding script from esm_viral_probe
EMBED_SCRIPT="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/scripts/extract_embeddings_esmc.py"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

cd "$PROJECT"

for GROUP in bacteria archaea phage; do
    FASTA="data/processed/${GROUP}_clean.faa"
    OUTFILE="data/embeddings/esmc_600m/${GROUP}.npz"

    if [ ! -f "$FASTA" ]; then
        echo "SKIP: $FASTA not found"
        continue
    fi

    if [ -f "$OUTFILE" ]; then
        echo "SKIP: $OUTFILE already exists"
        continue
    fi

    echo "=== Extracting ESMC-600M embeddings: $GROUP ==="
    python "$EMBED_SCRIPT" \
        --fasta     "$FASTA" \
        --outfile   "$OUTFILE" \
        --model     esmc_600m \
        --cache_dir "$HF_CACHE" \
        --batch_size 8 \
        --device    cuda
    echo ""
done

echo "=== Done ==="
