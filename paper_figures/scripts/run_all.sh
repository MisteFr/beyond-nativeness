#!/usr/bin/env bash
# Render the three main paper figures.
#
# Each script is self-contained: it re-applies the house style, loads the
# committed summary data from <repo>/data/figure_data/, and writes
# ${name}.{pdf,png} into <repo>/figures/. No GPU or API access is required.
#
# Prerequisites:
#   pip install -r requirements.txt        # numpy / pandas / scipy / sklearn / matplotlib
#
# Optional overrides:
#   BN_FIGURE_DATA   point at a regenerated projects/*/results tree instead of data/figure_data/
#   BN_FIGURES_DIR   write rendered figures somewhere other than <repo>/figures/

set -euo pipefail

cd "$(dirname "$0")"

for script in fig_pca_and_ppl_strip.py fig_family_nativization_esmc.py fig4_scaling_divergence.py; do
  echo "--- ${script} ---"
  python3 "${script}"
done

echo "--- rendered ---"
ls -1 ../../figures/fig_pca_and_ppl_strip.pdf \
      ../../figures/fig_family_nativization_esmc.pdf \
      ../../figures/fig4_scaling_divergence.pdf
