# Zero-Shot Perplexity Classifier

This project computes masked-reconstruction perplexity on the human
viral/cellular split for ESM2 and ESMC checkpoints. It then reports zero-shot AUC
using `-PPL` as the viral score.

ESM3 and ESMC-6B perplexities are produced by `esm3_masked_reconstruction` and
exposed here through symlinks. Together, these results provide the PPL baseline
and the family/group perplexities used by the figures.

## Inputs

- Human viral/cellular split FASTAs from `esm_viral_probe`.
- Tree-of-life FASTAs from `prokaryote_phage_ood` for per-group OOD PPL.
- `BEYOND_NATIVENESS_ROOT`, pointing at the repository root.
- A CUDA GPU. ESM2-3B and ESM2-15B need high-memory GPUs.

## Outputs

Under `results/`:

- `<model>/per_sequence_results.tsv` and `<model>/summary.json` for
  `esm2_{8m,35m,150m,650m,3b,15b}` and `esmc_{300m,600m}`.
- Per-group OOD perplexity tables from the `run_esm2_*_oodgroups` jobs.
- Symlinks for `esm3_*`, `esmc_6b`, and fine-tuned model directories. The
  targets live in `esm3_masked_reconstruction/results/`; do not remove them.

## Run

```bash
conda activate beyond-nativeness
export BEYOND_NATIVENESS_ROOT="$(git rev-parse --show-toplevel)"
```

1. Human-split perplexity:
   - `jobs/run_esm2_8m.sh` through `jobs/run_esm2_15b.sh`
   - `jobs/run_esmc_300m.sh`, `jobs/run_esmc_600m.sh`
2. Per-group perplexity for family-nativization figures:
   - `jobs/run_esm2_650m_oodgroups.sh`
   - `jobs/run_esm2_15b_oodgroups.sh`
3. Aggregate AUCs:
   - `jobs/run_analysis.sh`

## Scripts

- `scripts/run_masked_reconstruction_esm2.py`: ESM2 per-sequence PPL.
- `scripts/run_masked_reconstruction_esmc.py`: ESMC per-sequence PPL.
- `scripts/run_masked_reconstruction_esm2_oodgroups.py`: tree-of-life group PPL.
- `scripts/analyze_zeroshot_ppl.py`: AUC aggregation.
