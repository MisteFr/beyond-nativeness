# Post-Cutoff Non-Viral Control

This project checks whether high viral perplexity is just a novelty effect. It
downloads reviewed non-viral Swiss-Prot proteins created after the ESMC training
cutoff, then scores them with ESMC-600M using the same masked-reconstruction
protocol as the main analysis.

These post-cutoff non-viral proteins still receive low perplexity, close to the
pre-cutoff non-viral pool and far from viral proteins. The result feeds the
post-release control figure.

## Inputs

- UniProtKB Swiss-Prot REST query: reviewed, not `taxonomy_id:10239`, created on
  or after `2025-01-01`, length 50-1022.
- ESMC-600M weights, fetched on first use by the EvolutionaryScale `esm` package.

## Outputs

- `data/postcutoff_nonviral.faa.gz`: raw FASTA.
- `data/postcutoff_nonviral.tsv.gz`: accession, organism, and creation date.
- `data/postcutoff_nonviral_filtered.faa`: quality-filtered, length-stratified
  model input.
- `results/esmc_600m/per_sequence_results.tsv`: per-sequence PPL and recovery.
- `results/esmc_600m/summary.json`: aggregate PPL/recovery summary.

## Run

```bash
conda activate beyond-nativeness
export BEYOND_NATIVENESS_ROOT="$(git rev-parse --show-toplevel)"
```

Run from the repository root or this project directory.

1. `jobs/01_download.sh`: download and filter post-cutoff non-viral proteins.
   CPU-only, network required.
2. `jobs/02_reconstruction.sh`: run ESMC-600M masked reconstruction on the
   filtered FASTA. GPU recommended.

Each job wraps one Python script in `scripts/`, and those scripts can also be run
directly.
