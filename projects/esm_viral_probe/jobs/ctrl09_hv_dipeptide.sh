#!/usr/bin/env bash
# ============================================================
# Human Virus controls — Exp 9: dipeptide-composition baseline
#
# Trains a 400-D dipeptide-composition classifier (frequency of each
# ordered adjacent AA pair) as an extension to ctrl01's 20-D AA-comp
# baseline. This is the natural "second moment" of the AA distribution
# and quantifies how much local pair co-occurrence adds over marginal
# composition.
#
# Reuses scripts/baseline_classifiers.py via its --feature_sets flag
# so existing length_only / aa_composition / length_plus_composition
# results are preserved; summary.json is merged in place.
#
# Output: datasets/human_virus/results/baseline/dipeptide_composition/test_results.json
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
echo "[$(date)] Running dipeptide-composition baseline..."
python "$SCRIPTS/baseline_classifiers.py" \
    --data_dir      "$HV_ROOT/data/processed" \
    --outdir        "$HV_ROOT/results/baseline" \
    --feature_sets  "dipeptide_composition"

echo ""
echo "[$(date)] Dipeptide baseline complete."
echo ""
echo "All baseline results (merged summary.json):"
SUMMARY="$HV_ROOT/results/baseline/summary.json"
if [ -f "$SUMMARY" ]; then
    python3 -c "
import json
with open('$SUMMARY') as f:
    s = json.load(f)
print(f'  {\"Feature set\":<30}  {\"AUC-ROC\":>8}  {\"AUC-PR\":>8}  {\"F1\":>8}  {\"MCC\":>8}')
print('  ' + '-'*62)
for fs, m in s.items():
    print(f'  {fs:<30}  {m[\"auc_roc\"]:>8.4f}  {m[\"auc_pr\"]:>8.4f}  {m[\"f1\"]:>8.4f}  {m[\"mcc\"]:>8.4f}')
"
fi
