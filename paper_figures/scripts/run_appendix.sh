#!/usr/bin/env bash
# Render the nine appendix figures of the camera-ready paper:
#   - per-family nativization across ESM2/ESM3        (appfig_family_nativization_all)
#   - joint PCA + PPL across ESM2 and ESM3            (appfig_pca_ppl_esm2_esm3)
#   - low-FPR TPR comparison                          (appfig_tpr_at_low_fpr)
#   - post-release non-viral control                 (appfig_postrelease_control)
#   - human-only negatives control (Ctrl 3)          (appfig_human_negatives)
#   - leave-one-viral-family-out control (Ctrl 6)    (appfig_leave_family_out)
#   - cross-architecture PCA + PPL (ProGen2/EvoDiff) (appfig_crossarch_pca_ppl)
#   - cross-architecture per-family nativization     (appfig_family_nativization_nonesm)
#   - cross-architecture probe vs PPL scaling        (appfig_crossarch_probe)
#
# Prerequisites: same as run_all.sh — renders from the committed summary data in
# data/figure_data/ (no GPU/API needed); `pip install -r requirements.txt` first.

set -euo pipefail

cd "$(dirname "$0")"

FIGS=(
  appfig_family_nativization_all.py
  appfig_pca_ppl_esm2_esm3.py
  appfig_tpr_at_low_fpr.py
  appfig_postrelease_control.py
  appfig_human_negatives.py
  appfig_leave_family_out.py
  appfig_crossarch_pca_ppl.py
  appfig_family_nativization_nonesm.py
  appfig_crossarch_probe.py
)

for script in "${FIGS[@]}"; do
  echo "--- ${script} ---"
  python3 "${script}"
done

echo "--- rendered ---"
ls -1 ../../figures/appfig_family_nativization_all.pdf \
      ../../figures/appfig_pca_ppl_esm2_esm3.pdf \
      ../../figures/appfig_tpr_at_low_fpr.pdf \
      ../../figures/appfig_postrelease_control.pdf \
      ../../figures/appfig_human_negatives.pdf \
      ../../figures/appfig_leave_family_out.pdf \
      ../../figures/appfig_crossarch_pca_ppl.pdf \
      ../../figures/appfig_family_nativization_nonesm.pdf \
      ../../figures/appfig_crossarch_probe.pdf
