#!/usr/bin/env bash
# ============================================================
# Human Virus experiment — Step 3: train probes + evaluate
#
# Trains a linear probe and evaluates on the test set for all available models.
# Skips any model whose embedding files are not yet present.
#
# Run after the hv02a/b/c embedding jobs.
# Outputs: datasets/human_virus/results/<model>/test_results.json
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repository root}/projects/esm_viral_probe"
SCRIPTS="$PROJECT/scripts"
HV_ROOT="$PROJECT/datasets/human_virus"

# Activate the project environment first: conda activate beyond-nativeness

ALL_MODELS=(esm2_8m esm2_35m esm2_150m esm2_650m esm2_3b esm2_15b esmc_300m esmc_600m esmc_6b esm3_small esm3_open esm3_medium esm3_large)

for MODEL_KEY in "${ALL_MODELS[@]}"; do
    EMB_DIR="$HV_ROOT/data/embeddings/$MODEL_KEY"
    OUT_DIR="$HV_ROOT/results/$MODEL_KEY"
    mkdir -p "$OUT_DIR"

    # Skip if any embedding file is missing
    REQUIRED=(viral_train.npz nonviral_train.npz viral_val.npz nonviral_val.npz viral_test.npz nonviral_test.npz)
    MISSING=false
    for f in "${REQUIRED[@]}"; do
        if [ ! -f "$EMB_DIR/$f" ]; then
            echo "[$(date)] [$MODEL_KEY] SKIP: Missing $EMB_DIR/$f"
            MISSING=true
            break
        fi
    done
    [ "$MISSING" = true ] && continue

    echo ""
    echo "========================================"
    echo "[$MODEL_KEY] Training + Evaluating"
    echo "  Embeddings: $EMB_DIR"
    echo "  Results:    $OUT_DIR"
    echo "========================================"

    echo "[$(date)] Training linear probe..."
    python "$SCRIPTS/train_probe.py" \
        --viral_train    "$EMB_DIR/viral_train.npz" \
        --nonviral_train "$EMB_DIR/nonviral_train.npz" \
        --viral_val      "$EMB_DIR/viral_val.npz" \
        --nonviral_val   "$EMB_DIR/nonviral_val.npz" \
        --outdir         "$OUT_DIR"

    echo "[$(date)] Evaluating on test set..."
    python "$SCRIPTS/evaluate.py" \
        --viral_test    "$EMB_DIR/viral_test.npz" \
        --nonviral_test "$EMB_DIR/nonviral_test.npz" \
        --results_dir   "$OUT_DIR"

    echo "[$(date)] [$MODEL_KEY] Done."
    ls -lh "$OUT_DIR/"
done

echo ""
echo "[$(date)] All models trained and evaluated."
echo ""
echo "Summary of results:"
for MODEL_KEY in "${ALL_MODELS[@]}"; do
    RESULT_FILE="$HV_ROOT/results/$MODEL_KEY/test_results.json"
    if [ -f "$RESULT_FILE" ]; then
        python3 -c "
import json
with open('$RESULT_FILE') as f:
    r = json.load(f)
lin = r['linear']
print(f'  $MODEL_KEY: AUC-ROC={lin[\"auc_roc\"]:.4f}  AUC-PR={lin[\"auc_pr\"]:.4f}  F1={lin[\"f1\"]:.4f}  MCC={lin[\"mcc\"]:.4f}')
"
    else
        echo "  $MODEL_KEY: not found"
    fi
done
