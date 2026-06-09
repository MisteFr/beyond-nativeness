#!/usr/bin/env bash

# Extract OOD embeddings for ESMC 600M and ESMC 6B via the EvolutionaryScale Forge API.
# Embeds:
#   (a) probe training splits (viral/nonviral × train/val/test) -> esm_viral_probe embeddings dir
#   (b) OOD sequences (random_uniform, shuffled_viral, shuffled_nonviral) -> esm_random_ood embeddings dir

set -euo pipefail

# ---- Paths ----------------------------------------------------------------
OOD_PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm_random_ood"
PROBE_PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe"

PROBE_FASTA_DIR="${PROBE_PROJECT}/datasets/human_virus/data/processed"
PROBE_EMB_DIR="${PROBE_PROJECT}/datasets/human_virus/data/embeddings"
OOD_FASTA_DIR="${OOD_PROJECT}/data"
OOD_EMB_DIR="${OOD_PROJECT}/data/embeddings"

SCRIPT="${OOD_PROJECT}/scripts/extract_embeddings_esmc_api.py"

FORGE_TOKEN="${FORGE_TOKEN:?Set FORGE_TOKEN to your EvolutionaryScale Forge API key (see docs/forge_api_setup.md)}"

# ---- Environment ----------------------------------------------------------

mkdir -p "${OOD_PROJECT}/logs"

# ---- Model loop -----------------------------------------------------------
# model_name → directory key
declare -A MODELS
MODELS["esmc-600m-2024-12"]="esmc_600m"
MODELS["esmc-6b-2024-12"]="esmc_6b"

for MODEL_NAME in "esmc-600m-2024-12" "esmc-6b-2024-12"; do
    MODEL_KEY="${MODELS[$MODEL_NAME]}"
    echo "========================================================"
    echo "Model: ${MODEL_NAME}  (key: ${MODEL_KEY})"
    echo "========================================================"

    # ---- (a) Probe training splits ----------------------------------------
    mkdir -p "${PROBE_EMB_DIR}/${MODEL_KEY}"
    for PREFIX in viral nonviral; do
        for SPLIT in train val test; do
            OUTFILE="${PROBE_EMB_DIR}/${MODEL_KEY}/${PREFIX}_${SPLIT}.npz"
            if [ -f "${OUTFILE}" ]; then
                echo "  [skip] ${PREFIX}_${SPLIT} already exists"
                continue
            fi
            echo "  Embedding ${PREFIX}_${SPLIT} ..."
            python "${SCRIPT}" \
                --fasta   "${PROBE_FASTA_DIR}/${PREFIX}_${SPLIT}.faa" \
                --outfile "${OUTFILE}" \
                --token   "${FORGE_TOKEN}" \
                --model   "${MODEL_NAME}"
        done
    done

    # ---- (b) OOD sequences -------------------------------------------------
    mkdir -p "${OOD_EMB_DIR}/${MODEL_KEY}"
    for SEQ_TYPE in random_uniform shuffled_viral shuffled_nonviral; do
        OUTFILE="${OOD_EMB_DIR}/${MODEL_KEY}/${SEQ_TYPE}.npz"
        if [ -f "${OUTFILE}" ]; then
            echo "  [skip] ${SEQ_TYPE} already exists"
            continue
        fi
        echo "  Embedding ${SEQ_TYPE} ..."
        python "${SCRIPT}" \
            --fasta   "${OOD_FASTA_DIR}/${SEQ_TYPE}.faa" \
            --outfile "${OUTFILE}" \
            --token   "${FORGE_TOKEN}" \
            --model   "${MODEL_NAME}"
    done

    echo "  Done: ${MODEL_KEY}"
done

echo "All ESMC embeddings complete."
