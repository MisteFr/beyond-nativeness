# Paper Figures

This directory contains the scripts that render the paper figures from the
summary tables and precomputed PCA coordinates in `data/figure_data/`.

## Layout

```text
paper_figures/
+-- scripts/
|   +-- _common.py             # model registries, colors, data loading
|   +-- _sfp.py                # local scientific-figure-pro helper
|   +-- fig*.py                # main figure scripts
|   +-- appfig_*.py            # appendix figure scripts
|   +-- _*.tsv, _*.json        # small derived sidecars
|   +-- run_all.sh             # render the main set
|   +-- run_appendix.sh        # render appendix figures
+-- README.md
```

By default, the scripts read committed summaries from
`${BN_FIGURE_DATA:-data/figure_data}`. To render from fresh regeneration outputs,
set `BN_FIGURE_DATA` to the relevant results tree and keep
`BEYOND_NATIVENESS_ROOT` pointed at the repository root.

## Figure scripts

- `fig_pca_and_ppl_strip.py`: PCA view of the nativeness axis and group-level PPL.
- `fig_family_nativization_esmc.py`: per-family movement along the axis with ESMC scale.
- `fig4_scaling_divergence.py`: probe and zero-shot PPL behavior across model scale.
- `appfig_pca_ppl_esm2_esm3.py`: PCA comparison across ESMC-600M, ESM2-650M, and ESM3-open.
- `appfig_family_nativization_all.py`: per-family scaling for ESM2 and ESM3.
- `appfig_postrelease_control.py`: post-release non-viral perplexity control.
- `appfig_tpr_at_low_fpr.py`: TPR at strict false-positive rates.
- `appfig_crossarch_*.py` and `appfig_family_nativization_nonesm.py`: non-ESM appendix figures.

## Run

```bash
conda activate beyond-nativeness
export BEYOND_NATIVENESS_ROOT="$(git rev-parse --show-toplevel)"
bash paper_figures/scripts/run_all.sh
bash paper_figures/scripts/run_appendix.sh
```

Rendered PDFs and PNGs are written to `<repo>/figures/`.
