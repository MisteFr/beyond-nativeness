#!/usr/bin/env bash
# ============================================================
# Cross-architecture nativeness — viral probe, step 1: train + evaluate
#
# Reuses the ESM viral-probe scripts VERBATIM (train_probe.py / evaluate.py)
# on the already-extracted non-ESM embeddings, which share the byte-identical
# npz format (keys: embeddings[N,D] f32, accessions[N]) and the SAME human-virus
# train/val/test splits. Labels are assigned by file (viral->1, nonviral->0).
#
# Outputs per model -> results/<model>/probe/:
#   linear_probe.pkl  scaler.pkl  train_metrics.json  probe_config.json
#   test_results.json  test_preds_linear.npz
#
# CPU only; expects the score jobs to have written the embeddings already.
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT}/projects/cross_architecture_nativeness"
ESM="${BEYOND_NATIVENESS_ROOT}/projects/esm_viral_probe"

# evaluate.py does `from train_probe import load_embeddings`, so the ESM scripts
# dir must be importable.
export PYTHONPATH="$ESM/scripts:${PYTHONPATH:-}"

# Activate the project environment first: conda activate beyond-nativeness

# Full scaling ladder consumed by paper_figures/scripts/appfig_crossarch_probe.py
# (the 2-panel probe-vs-PPL scaling figure): ProGen2 151M/764M/2.7B/6.4B +
# EvoDiff 38M/640M, plus the single-point ProtT5-XL. The 6 human-virus split
# embeddings for each key are produced by the committed score jobs
# (score_progen2_base.sh, score_progen2_scale.sh with MODEL_KEY=progen2_{small,
# large,xlarge}, score_evodiff_640m.sh, score_evodiff_38m.sh, score_prott5_xl.sh);
# any key whose embeddings are absent is skipped gracefully below.
MODELS=(progen2_small progen2_base progen2_large progen2_xlarge evodiff_oadm_38m evodiff_oadm_640m prott5_xl)

for MODEL_KEY in "${MODELS[@]}"; do
    EMB_DIR="$PROJECT/data/embeddings/$MODEL_KEY"
    OUT_DIR="$PROJECT/results/$MODEL_KEY/probe"
    mkdir -p "$OUT_DIR"

    # Skip if any of the 6 human-virus split embeddings is missing
    REQUIRED=(viral_train.npz nonviral_train.npz viral_val.npz nonviral_val.npz viral_test.npz nonviral_test.npz)
    MISSING=false
    for f in "${REQUIRED[@]}"; do
        if [ ! -f "$EMB_DIR/$f" ]; then
            echo "[$(date)] [$MODEL_KEY] SKIP: missing $EMB_DIR/$f"
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
    python "$ESM/scripts/train_probe.py" \
        --viral_train    "$EMB_DIR/viral_train.npz" \
        --nonviral_train "$EMB_DIR/nonviral_train.npz" \
        --viral_val      "$EMB_DIR/viral_val.npz" \
        --nonviral_val   "$EMB_DIR/nonviral_val.npz" \
        --outdir         "$OUT_DIR"

    echo "[$(date)] Evaluating on test set..."
    python "$ESM/scripts/evaluate.py" \
        --viral_test    "$EMB_DIR/viral_test.npz" \
        --nonviral_test "$EMB_DIR/nonviral_test.npz" \
        --results_dir   "$OUT_DIR"

    echo "[$(date)] [$MODEL_KEY] Done."
    ls -lh "$OUT_DIR/"
done

echo ""
echo "[$(date)] Summary of test AUCs:"
for MODEL_KEY in "${MODELS[@]}"; do
    RESULT_FILE="$PROJECT/results/$MODEL_KEY/probe/test_results.json"
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
echo "[$(date)] probe01 complete."
