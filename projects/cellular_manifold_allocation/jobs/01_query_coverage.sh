#!/usr/bin/env bash
set -euo pipefail

# Stage 1 of the reproduction pipeline: produces results/coverage_taxa.tsv
# (Table 1) by querying the UniProt REST API. CPU-only, no GPU.

# Activate the project environment first: conda activate beyond-nativeness

cd "${BEYOND_NATIVENESS_ROOT}/projects/cellular_manifold_allocation"

echo "Started at $(date -Iseconds)"
python3 -u scripts/01_query_coverage.py
echo "Finished at $(date -Iseconds)"
