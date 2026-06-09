# Cross-Architecture Nativeness

This project repeats the nativeness analysis on non-ESM protein language models.
The goal is to check whether the PCA/PPL axis and viral-probe behavior are
specific to the ESM family or appear across different training objectives.

Models:

- ProGen2 (151M-6.4B): autoregressive, causal perplexity.
- EvoDiff OA-DM (38M, 640M): discrete diffusion, order-agnostic ELBO
  pseudo-perplexity.
- ProtT5-XL (3B): span denoising.

For each model, the scripts compute per-sequence scores and mean-pooled
embeddings over the same biological/control pools used in the appendix. The
ProGen2/EvoDiff ladder is also used for the linear viral/cellular probe.

## Inputs

- Human-virus split, tree-of-life FASTAs, and shuffled/random controls from
  `data/`, `esm_viral_probe`, `prokaryote_phage_ood`, and `esm_random_ood`.
- `BEYOND_NATIVENESS_ROOT`, pointing at the repository root.
- A CUDA GPU.
- A separate EvoDiff virtualenv at `envs/evodiff_venv`; EvoDiff dependencies
  conflict with the main environment.

## Outputs

Under `results/`:

- `<model>/per_sequence_results.tsv` and per-group subdirectories for
  `progen2_{small,base,large,xlarge}`, `evodiff_oadm_38m(_elbo)`,
  `evodiff_oadm_640m(_elbo)`, and `prott5_xl`.
- `<model>/probe/test_results.json`: linear-probe metrics.

## Run

```bash
conda activate beyond-nativeness
export BEYOND_NATIVENESS_ROOT="$(git rev-parse --show-toplevel)"
```

Forge is not used here; all model families are open.

1. ProGen2:
   - `jobs/score_progen2_base.sh`
   - `jobs/score_progen2_scale.sh` with `MODEL_KEY=progen2_small`,
     `progen2_large`, or `progen2_xlarge`
2. EvoDiff:
   - `jobs/score_evodiff_38m.sh`
   - `jobs/score_evodiff_640m.sh`
   - `jobs/score_evodiff_elbo.sh`
3. ProtT5:
   - `jobs/score_prott5_xl.sh`
4. Probe:
   - `jobs/probe01_train_eval.sh`

## Scripts

- `scripts/score_progen2.py`: autoregressive score and embeddings.
- `scripts/score_evodiff.py`: ELBO score and embeddings.
- `scripts/score_prott5.py`: span-denoising score and embeddings.
- `scripts/nat_io.py`: FASTA/pool loading and accession handling.
