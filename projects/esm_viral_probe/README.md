# ESM Viral Probe

This project rebuilds the human viral/cellular classification experiment. It
creates the homology-controlled split, extracts mean-pooled final-layer
embeddings for the 13 ESM-family models, trains a logistic-regression probe for
each model, and runs the probe controls used in the paper.

The committed figures render from summary data, so this directory is only needed
when regenerating those summaries from sequences and model calls.

## Inputs

- `${BEYOND_NATIVENESS_ROOT}/projects/human_virus_dataset/data/processed/human_virus_clean.faa`:
  curated human-infecting viral proteins.
- `data/nonviral/uniprot_nonviral.faa`: UniProtKB/Swiss-Prot non-viral proteins,
  downloaded by `data/download_nonviral.sh`.
- `data/taxonomy/`: NCBI taxonomy dump (`nodes.dmp`, `names.dmp`), needed only
  for leave-one-family-out.
- `HF_TOKEN`: needed for the gated `esm3_open` model.
- `FORGE_TOKEN`: needed for Forge-hosted ESMC/ESM3 models.

## Outputs

All outputs are under `datasets/human_virus/`.

- `results/<model>/test_results.json`: probe metrics for
  `esm2_8m`, `esm2_35m`, `esm2_150m`, `esm2_650m`, `esm2_3b`, `esm2_15b`,
  `esmc_300m`, `esmc_600m`, `esmc_6b`, `esm3_open`, `esm3_small`,
  `esm3_medium`, and `esm3_large`.
- `results/baseline/summary.json`: length, amino-acid composition,
  length+composition, and dipeptide baselines.
- `results/human_neg_summary.json`: human-only negative control.
- `results/leave_family_out_summary.json`: leave-one-family-out control.
- `data/controls/leave_family_out/family_metadata.tsv`: accession-to-family map.

## Run

```bash
conda activate beyond-nativeness
export BEYOND_NATIVENESS_ROOT="$(git rev-parse --show-toplevel)"
```

1. `data/download_nonviral.sh`: download and filter non-viral negatives.
2. `jobs/hv01_preprocess.sh`: build the MMseqs2 cluster-aware 60/20/20 split.
   MMseqs2 must be on `PATH`.
3. Extract embeddings:
   - `jobs/hv02a_embed_esm2.sh`: ESM2 8M/35M/150M/650M.
   - `jobs/hv02a2_embed_esm2_3b.sh`: ESM2-3B.
   - `jobs/hv02a3_embed_esm2_15b.sh`: ESM2-15B.
   - `jobs/hv02b_embed_esmc.sh`: ESMC-300M and ESM3-open.
   - `jobs/hv02c_embed_forge.sh`: ESM3-small and ESM3-medium through Forge.
   - Forge-hosted `esmc_600m`, `esmc_6b`, and `esm3_large` follow the same
     split pattern with `scripts/extract_embeddings_forge.py` and
     `scripts/extract_embeddings_esmc.py`.
4. `jobs/hv03_train_eval.sh`: train and evaluate probes for all models with
   available embeddings.
5. Controls and baselines:
   - `jobs/ctrl01_hv_baselines.sh` and `jobs/ctrl09_hv_dipeptide.sh`
   - `jobs/ctrl03_hv_human_neg.sh`
   - `jobs/ctrl06_hv_leave_family_out.sh`

Embedding jobs require a CUDA GPU or Forge API access. Preprocessing, probes,
and baselines run on CPU once embeddings are available.
