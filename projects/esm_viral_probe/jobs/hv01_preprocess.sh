#!/usr/bin/env bash
# ============================================================
# Human Virus experiment — Step 1: preprocess
#
# Runs preprocess.py (MMseqs2 cluster-aware 60/20/20 split)
# on the curated human-virus protein dataset.
#
# Viral input:    human_virus_clean.faa (5,815 seqs, already curated)
#                 Filters to [50–1022 aa], <5% non-std AA → 5,203 usable
# Non-viral:      uniprot_nonviral.faa (reused, subsampled to 5,203 to match viral)
# n_samples=5203 → balanced 50/50 dataset (preprocess.py applies n_samples per class)
#
# Output: datasets/human_virus/data/processed/
# ============================================================

set -euo pipefail

PROJECT="${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repository root}/projects/esm_viral_probe"
SCRIPTS="$PROJECT/scripts"
THREADS=8

VIRAL="${BEYOND_NATIVENESS_ROOT}/projects/human_virus_dataset/data/processed/human_virus_clean.faa"
NONVIRAL="$PROJECT/data/nonviral/uniprot_nonviral.faa"
OUTDIR="$PROJECT/datasets/human_virus/data/processed"

mkdir -p "$OUTDIR"

# Activate the project environment first: conda activate beyond-nativeness

# ---- Validate inputs ----
echo ""
echo "Validating input files..."
for f in "$VIRAL" "$NONVIRAL"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing required file: $f"
        exit 1
    fi
done
echo "  Viral:    $VIRAL"
echo "  Nonviral: $NONVIRAL"
echo "  All required input files found."
echo ""
echo "Viral input sequence count: $(grep -c '^>' "$VIRAL")"

# MMseqs2 must be pre-installed in the conda env
echo "MMseqs2 version: $(mmseqs version 2>/dev/null || echo 'NOT FOUND — install on login node first')"
command -v mmseqs &>/dev/null || { echo "ERROR: mmseqs not found"; exit 1; }

# ---- Run preprocessing (filter, cluster, split) ----
echo ""
echo "========================================"
echo "Preprocessing (MMseqs2 cluster-aware split)"
echo "  Viral:    $VIRAL"
echo "  Nonviral: $NONVIRAL"
echo "  Output:   $OUTDIR"
echo "  n_samples: 999999 (use all sequences passing length/quality filters)"
echo "========================================"

# n_samples=5203 = exact viral count after length/quality filtering (50–1022 aa, <5% non-std AA).
# preprocess.py applies n_samples independently per class, so non-viral is also subsampled
# to 5203 → balanced 50/50 dataset. Viral count determined from hv01 first run (2026-03-12).
python "$SCRIPTS/preprocess.py" \
    --viral     "$VIRAL" \
    --nonviral  "$NONVIRAL" \
    --outdir    "$OUTDIR" \
    --n_samples 5203 \
    --threads   "$THREADS" \
    --seed      42

echo ""
echo "[$(date)] Preprocessing complete."
echo ""
echo "Output directory:"
ls -lh "$OUTDIR/"
echo ""
echo "Split counts:"
for LABEL in viral nonviral; do
    for SPLIT in train val test; do
        F="$OUTDIR/${LABEL}_${SPLIT}.faa"
        if [ -f "$F" ]; then
            N=$(grep -c '^>' "$F" || echo 0)
            echo "  ${LABEL}_${SPLIT}: $N sequences"
        fi
    done
done
