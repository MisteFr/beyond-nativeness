#!/usr/bin/env bash
# ============================================================
# Unified analysis: PPL + LL AUC across all models, compared
# with probe AUC from esm_viral_probe. Produces comparison_summary.json.
# ============================================================

set -euo pipefail

# Activate the project environment first: conda activate beyond-nativeness

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repo root}/projects/esm_zeroshot_ppl"

python "$PROJECT/scripts/analyze_zeroshot_ppl.py" \
    --results_dir "$PROJECT/results" \
    --probe_dir   "${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe/datasets/human_virus/results" \
    --out_dir     "$PROJECT/results/figures"

echo ""
echo "Completed."
ls -lh "$PROJECT/results/figures/" 2>/dev/null || echo "  (empty)"
