#!/usr/bin/env bash
# Render every paper figure (3 main + 9 appendix + 1 cross-architecture companion)
# from the committed summary data in data/figure_data/.
#
#   pip install -r requirements.txt
#   ./render_figures.sh
#
# Outputs land in figures/ as {name}.pdf and {name}.png.
set -euo pipefail
cd "$(dirname "$0")"

bash paper_figures/scripts/run_all.sh
bash paper_figures/scripts/run_appendix.sh

echo
echo "Done. Rendered figures are in: $(pwd)/figures/"
