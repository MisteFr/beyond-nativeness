#!/usr/bin/env bash
# ============================================================
# Human Virus controls — Exp 1: shallow sequence-statistics baselines
#
# Trains length-only, AA-composition, and length+composition
# classifiers on the Human Virus dataset as ablation controls.
#
# Output: datasets/human_virus/results/baseline/{length_only,
#         aa_composition,length_plus_composition}/test_results.json
#         (merged into datasets/human_virus/results/baseline/summary.json)
#
# Requires the processed FASTA splits from hv01_preprocess.sh.
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repository root}/projects/esm_viral_probe"
SCRIPTS="$PROJECT/scripts"
HV_ROOT="$PROJECT/datasets/human_virus"

# Activate the project environment first: conda activate beyond-nativeness

# Validate inputs
for SPLIT in train val test; do
    for LABEL in viral nonviral; do
        FASTA="$HV_ROOT/data/processed/${LABEL}_${SPLIT}.faa"
        if [ ! -f "$FASTA" ]; then
            echo "ERROR: Missing required FASTA: $FASTA"
            exit 1
        fi
    done
done
echo "All 6 processed FASTAs found."

mkdir -p "$HV_ROOT/results/baseline"

echo ""
echo "[$(date)] Running baseline classifiers..."
python "$SCRIPTS/baseline_classifiers.py" \
    --data_dir "$HV_ROOT/data/processed" \
    --outdir   "$HV_ROOT/results/baseline"

echo ""
echo "[$(date)] Baseline classifiers complete."
echo ""
echo "Results:"
for FS in length_only aa_composition length_plus_composition; do
    RESULT_FILE="$HV_ROOT/results/baseline/$FS/test_results.json"
    if [ -f "$RESULT_FILE" ]; then
        python3 -c "
import json
with open('$RESULT_FILE') as f:
    r = json.load(f)
lin = r['linear']
print(f'  $FS: AUC-ROC={lin[\"auc_roc\"]:.4f}  AUC-PR={lin[\"auc_pr\"]:.4f}  F1={lin[\"f1\"]:.4f}  MCC={lin[\"mcc\"]:.4f}')
"
    else
        echo "  $FS: not found"
    fi
done
