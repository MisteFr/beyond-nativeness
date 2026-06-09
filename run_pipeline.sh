#!/usr/bin/env bash
# Pipeline for Beyond Nativeness.
#
# To just reproduce the FIGURES (no GPU/API needed), you do not need this file:
#   pip install -r requirements.txt && ./render_figures.sh
#
# Prerequisites for regeneration:
#   conda env create -f environment.yml -n beyond-nativeness && conda activate beyond-nativeness
#   export BEYOND_NATIVENESS_ROOT="$PWD"
#   export FORGE_TOKEN="<your EvolutionaryScale Forge API key>"   # docs/forge_api_setup.md
#   export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"         # + huggingface-cli login for ESM3-open
#
# Read docs/reproduction_guide.md first; each projects/<module>/README.md documents
# that module's inputs, outputs, and run order.

set -euo pipefail
: "${BEYOND_NATIVENESS_ROOT:?Set BEYOND_NATIVENESS_ROOT to the repo root}"
cd "${BEYOND_NATIVENESS_ROOT}"

cat <<'EOF'
=========================================================================
 Beyond Nativeness — regeneration pipeline (guided)
=========================================================================
Run the stages below in order, waiting for each to finish before the next.
GPU/Forge stages are noted. Outputs land under projects/<module>/results/ and
projects/<module>/data/embeddings/; point BN_FIGURE_DATA at that tree (or copy
it into data/figure_data/) to re-render the figures from fresh results.

-------------------------------------------------------------------------------
Stage 0 — Render figures from the committed summary data (CPU, ~1 min)
-------------------------------------------------------------------------------
  pip install -r requirements.txt
  ./render_figures.sh

-------------------------------------------------------------------------------
Stage 1 — Pretraining-coverage counts (Table 1) — CPU
-------------------------------------------------------------------------------
  bash projects/cellular_manifold_allocation/jobs/01_query_coverage.sh

-------------------------------------------------------------------------------
Stage 2 — Dataset assembly (FASTAs are vendored in data/; optional) — CPU
-------------------------------------------------------------------------------
  # Curated human-virus pool (Table 2)
  bash projects/human_virus_dataset/jobs/01_download.sh
  bash projects/human_virus_dataset/jobs/02_process.sh
  bash projects/human_virus_dataset/jobs/03_clean_normalize.sh
  # Eight tree-of-life groups
  bash projects/prokaryote_phage_ood/jobs/01_download.sh
  bash projects/prokaryote_phage_ood/jobs/02_clean_filter.sh
  bash projects/prokaryote_phage_ood/jobs/12_download_expanded.sh
  bash projects/prokaryote_phage_ood/jobs/13_clean_filter_expanded.sh
  # Shuffled / random controls
  bash projects/esm_random_ood/jobs/01_generate_random.sh
  # Non-viral negatives + human viral/cellular classification split (MMseqs2)
  bash projects/esm_viral_probe/data/download_nonviral.sh
  bash projects/esm_viral_probe/jobs/hv01_preprocess.sh

-------------------------------------------------------------------------------
Stage 3 — Embeddings on the human classification split (GPU + Forge)
-------------------------------------------------------------------------------
  bash projects/esm_viral_probe/jobs/hv02a_embed_esm2.sh        # ESM2 8M-650M
  bash projects/esm_viral_probe/jobs/hv02a2_embed_esm2_3b.sh    # ESM2-3B
  bash projects/esm_viral_probe/jobs/hv02a3_embed_esm2_15b.sh   # ESM2-15B
  bash projects/esm_viral_probe/jobs/hv02b_embed_esmc.sh        # ESMC-300M/600M + ESM3-open
  bash projects/esm_viral_probe/jobs/hv02c_embed_forge.sh       # ESM3-S/M/L + ESMC-6B (Forge)

-------------------------------------------------------------------------------
Stage 4 — Linear probe + baselines + controls (CPU) — Figs 3, 8, 9
-------------------------------------------------------------------------------
  bash projects/esm_viral_probe/jobs/hv03_train_eval.sh
  bash projects/esm_viral_probe/jobs/ctrl01_hv_baselines.sh
  bash projects/esm_viral_probe/jobs/ctrl09_hv_dipeptide.sh
  bash projects/esm_viral_probe/jobs/ctrl03_hv_human_neg.sh
  bash projects/esm_viral_probe/jobs/ctrl06_hv_leave_family_out.sh

-------------------------------------------------------------------------------
Stage 5 — Masked-reconstruction PPL (GPU + Forge) — Figs 1, 2, 4, 6
-------------------------------------------------------------------------------
  # ESMC-600M + ESM3-open + Forge models, human pool + controls:
  for j in projects/esm3_masked_reconstruction/jobs/*.sh; do bash "$j"; done
  # Zero-shot PPL classifier across all 13 ESM models:
  for j in projects/esm_zeroshot_ppl/jobs/run_esm2_*.sh \
           projects/esm_zeroshot_ppl/jobs/run_esmc_*.sh; do bash "$j"; done
  bash projects/esm_zeroshot_ppl/jobs/run_analysis.sh

-------------------------------------------------------------------------------
Stage 6 — Tree-of-life group embeddings + per-group PPL (GPU) — Figs 1, 6
-------------------------------------------------------------------------------
  bash projects/prokaryote_phage_ood/jobs/03_embed.sh
  bash projects/prokaryote_phage_ood/jobs/14_embed_expanded.sh
  for j in projects/prokaryote_phage_ood/jobs/04?_recon_*.sh \
           projects/prokaryote_phage_ood/jobs/15?_recon_*.sh; do bash "$j"; done
  bash projects/prokaryote_phage_ood/jobs/30_embed_ppl_esm2_650m_phage_ood.sh
  bash projects/prokaryote_phage_ood/jobs/30_embed_ppl_esm3_open_phage_ood.sh
  # Control embeddings for the appendix PCA:
  bash projects/esm_random_ood/jobs/02a_embed_gpu.sh
  bash projects/esm_random_ood/jobs/02c_embed_esmc_api.sh

-------------------------------------------------------------------------------
Stage 7 — Post-release non-viral control (App. Fig 4) — CPU + 1 GPU
-------------------------------------------------------------------------------
  bash projects/postcutoff_nonviral/jobs/01_download.sh
  bash projects/postcutoff_nonviral/jobs/02_reconstruction.sh

-------------------------------------------------------------------------------
Stage 8 — Cross-architecture (non-ESM): ProGen2 / EvoDiff / ProtT5 (GPU)
-------------------------------------------------------------------------------
  # EvoDiff needs a one-time separate venv — see that module's README.
  bash projects/cross_architecture_nativeness/jobs/score_progen2_base.sh
  for k in progen2_small progen2_large progen2_xlarge; do
    MODEL_KEY=$k bash projects/cross_architecture_nativeness/jobs/score_progen2_scale.sh
  done
  bash projects/cross_architecture_nativeness/jobs/score_prott5_xl.sh
  bash projects/cross_architecture_nativeness/jobs/score_evodiff_38m.sh
  bash projects/cross_architecture_nativeness/jobs/score_evodiff_640m.sh
  bash projects/cross_architecture_nativeness/jobs/score_evodiff_elbo.sh
  bash projects/cross_architecture_nativeness/jobs/probe01_train_eval.sh

-------------------------------------------------------------------------------
Stage 9 — Re-render the figures from the fresh results
-------------------------------------------------------------------------------
  BN_FIGURE_DATA="$PWD/projects" ./render_figures.sh
  # (projects/*/results mirror the data/figure_data/ layout the figures expect)
EOF
