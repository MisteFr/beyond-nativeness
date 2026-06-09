# ESM3 / ESMC Masked Reconstruction

This project computes masked-reconstruction perplexity for the human viral and
non-viral split, plus the shuffled/random controls. It covers ESMC-600M and
ESM3-open locally, and ESMC-6B plus ESM3-small/medium/large through the Forge
API.

Each run masks 15% of amino-acid positions and averages per-sequence perplexity
over three mask seeds. The scale summary marks viral sequences that fall below
the per-model non-viral P95 threshold.

## Inputs

- `data/{viral,nonviral}_{train,val,test}.faa`: local split for ESMC-600M and
  Forge human-pool runs.
- Processed split FASTAs from `esm_viral_probe` for the ESM3-open human-pool run.
- Control FASTAs from `esm_random_ood`: `shuffled_viral.faa`,
  `shuffled_nonviral.faa`, and `random_uniform.faa`.
- `BEYOND_NATIVENESS_ROOT`, pointing at the repository root.
- CUDA for local runs; `FORGE_TOKEN` for API runs; `HF_HOME`/`HF_TOKEN` as needed
  for local Hugging Face model access.

## Outputs

Under `results/`:

- `esmc_600m/per_sequence_results.tsv`: ESMC-600M human pool.
- `per_sequence_results.tsv`: ESM3-open human pool.
- `esmc_600m/{shuffled_viral,shuffled_nonviral,random_uniform}/per_sequence_results.tsv`:
  ESMC-600M controls.
- `{shuffled_viral,shuffled_nonviral,random_uniform}/per_sequence_results.tsv`:
  ESM3-open controls.
- `esmc_6b/`, `esm3_medium/`, `esm3_large/`, `esm3_small_api/`: Forge tables for
  human-pool and control runs.
- `lowppl_by_scale/04_consistent_vs_scale.tsv` and `per_model_lowppl.tsv`:
  low-perplexity flags across scales.

## Run

```bash
conda activate beyond-nativeness
export BEYOND_NATIVENESS_ROOT="$(git rev-parse --show-toplevel)"
```

1. Human-pool PPL, local GPU:
   - `jobs/run_esmc_600m.sh`
   - `jobs/run_reconstruction.sh`
2. Human-pool PPL, Forge:
   - `jobs/run_esmc_6b_api.sh`
   - `jobs/run_esm3_medium_api.sh`
   - `jobs/run_esm3_large_api.sh`
   - `jobs/run_esm3small_api.sh`
3. OOD controls:
   - Local GPU: `jobs/run_shuffled_viral_esmc_600m.sh`,
     `jobs/run_shuffled_nonviral_esmc_600m.sh`, `jobs/run_random_esmc_600m.sh`,
     `jobs/run_shuffled_viral_esm3_open.sh`,
     `jobs/run_shuffled_nonviral_esm3_open.sh`, `jobs/run_random_esm3_open.sh`
   - Forge: matching `run_{shuffled_viral,shuffled_nonviral,random}_*` scripts
     for ESMC-6B and ESM3-small/medium/large.
4. Aggregate the scale table:
   - `python scripts/analyze_lowppl_by_scale.py`

## Scripts

- `scripts/run_masked_reconstruction_esmc.py`: ESMC local human-pool run.
- `scripts/run_masked_reconstruction.py`: ESM3-open local human-pool run.
- `scripts/run_masked_reconstruction_forge.py`: Forge human-pool runs.
- `scripts/run_masked_reconstruction_api.py`: ESM3-small Forge human-pool run.
- `scripts/run_masked_reconstruction_single_fasta.py`: single-FASTA controls.
- `scripts/analyze_lowppl_by_scale.py`: scale aggregation.
