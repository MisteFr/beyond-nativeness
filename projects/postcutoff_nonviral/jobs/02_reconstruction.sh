#!/usr/bin/env bash
set -euo pipefail

cd ${BEYOND_NATIVENESS_ROOT}/projects/postcutoff_nonviral

# Activate the project environment first: conda activate beyond-nativeness

export PYTHONUNBUFFERED=1

echo "=== Post-cutoff ESMC-600M Masked Reconstruction ==="

python scripts/02_run_masked_reconstruction.py \
    --fasta data/postcutoff_nonviral_filtered.faa \
    --out_dir results/esmc_600m \
    --model esmc_600m \
    --batch_size 4 \
    --mask_rate 0.15 \
    --n_seeds 3

echo ""
echo "=== Done ==="
