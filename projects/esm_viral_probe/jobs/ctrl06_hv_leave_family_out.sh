#!/usr/bin/env bash
# ============================================================
# Human Virus controls — Exp 6: leave-one-family-out evaluation
#
# Step 1 maps viral accessions to NCBI family (family_metadata.tsv);
# Step 2 trains a probe on all other families and evaluates on the
# held-out family, for each model with cached embeddings.
#
# Output: datasets/human_virus/data/controls/leave_family_out/family_metadata.tsv
#         datasets/human_virus/results/leave_family_out_summary.json
#
# Requires the cached embeddings from hv02* and the NCBI taxonomy dump
# under data/taxonomy/ (nodes.dmp, names.dmp).
# ============================================================

set -euo pipefail

cd "${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repository root}/projects/esm_viral_probe"

# Activate the project environment first: conda activate beyond-nativeness

FAMILY_META=datasets/human_virus/data/controls/leave_family_out/family_metadata.tsv

echo "=== Step 1: prepare_family_splits.py ==="
python scripts/prepare_family_splits.py \
    --source_meta ${BEYOND_NATIVENESS_ROOT}/projects/human_virus_dataset/data/processed/human_virus_clean.tsv \
    --taxdir      data/taxonomy \
    --out         "${FAMILY_META}"

echo ""
echo "=== Step 2: eval_leave_family_out.py ==="
python scripts/eval_leave_family_out.py \
    --project_dir datasets/human_virus \
    --family_meta "${FAMILY_META}" \
    --min_family_size 50 \
    --threshold 0.5

echo ""
echo "=== ctrl06 DONE ==="
