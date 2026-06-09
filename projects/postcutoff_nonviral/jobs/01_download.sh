#!/usr/bin/env bash
set -euo pipefail

cd ${BEYOND_NATIVENESS_ROOT}/projects/postcutoff_nonviral

# Activate the project environment first: conda activate beyond-nativeness

export PYTHONUNBUFFERED=1

echo "=== Post-cutoff nonviral protein download ==="

python scripts/01_download_postcutoff.py

echo ""
echo "=== Done ==="
