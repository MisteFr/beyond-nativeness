#!/usr/bin/env bash
# ============================================================
# Human Virus controls — Exp 3: human-only negatives
#
# Re-evaluates the trained probes using ONLY Homo sapiens proteins
# as the negative class. The hardest control for a dataset-construction
# artifact.
#
# Requires: trained probes / test predictions for each model (from hv03)
#
# Output: datasets/human_virus/results/{model}/human_neg_results.json
#         datasets/human_virus/results/human_neg_summary.json
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repository root}/projects/esm_viral_probe"
SCRIPTS="$PROJECT/scripts"
HV_ROOT="$PROJECT/datasets/human_virus"
NONVIRAL_FASTA="$PROJECT/data/nonviral/uniprot_nonviral.faa"

# Activate the project environment first: conda activate beyond-nativeness

if [ ! -f "$NONVIRAL_FASTA" ]; then
    echo "ERROR: Missing SwissProt FASTA: $NONVIRAL_FASTA"
    exit 1
fi
echo "SwissProt FASTA found: $(wc -l < "$NONVIRAL_FASTA") lines"

echo ""
echo "[$(date)] Running human-only negatives evaluation..."
python "$SCRIPTS/human_negatives_eval.py" \
    --project_dir    "$HV_ROOT" \
    --nonviral_fasta "$NONVIRAL_FASTA"

echo ""
echo "[$(date)] Human-only negatives evaluation complete."
echo ""
echo "Summary:"
SUMMARY="$HV_ROOT/results/human_neg_summary.json"
if [ -f "$SUMMARY" ]; then
    python3 -c "
import json
with open('$SUMMARY') as f:
    s = json.load(f)
print(f'  {\"Model\":<20}  {\"Full AUC\":>9}  {\"Human AUC\":>10}  {\"Delta\":>7}  {\"N human\":>8}')
print('  ' + '-'*60)
for model, r in s.items():
    full = r.get('full_auc') or float('nan')
    human = r.get('human_auc', float('nan'))
    delta = human - full if r.get('full_auc') else float('nan')
    n = r.get('n_human', 0)
    print(f'  {model:<20}  {full:>9.4f}  {human:>10.4f}  {delta:>+7.4f}  {n:>8,}')
"
fi
