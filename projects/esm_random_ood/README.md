# Random and Shuffled Controls

This project builds the non-biological control pools used as OOD anchors in the
PCA figures, then extracts their mean-pooled embeddings.

Controls:

- `random_uniform`: length-matched sequences with residues sampled uniformly from
  the 20 standard amino acids.
- `shuffled_viral`: human-viral proteins with residue positions permuted.
- `shuffled_nonviral`: non-viral proteins with residue positions permuted.

The shuffled controls preserve length and amino-acid composition but remove
positional structure. The FASTAs are also included under `../../data/controls/`.

## Inputs

- Human-virus and non-viral processed splits from `esm_viral_probe`.
- `BEYOND_NATIVENESS_ROOT`, pointing at the repository root.
- A CUDA GPU for ESM2 and ESM3-open embeddings.
- `FORGE_TOKEN` for ESMC-600M embeddings through the EvolutionaryScale Forge API.

## Outputs

- `data/<control>.faa`: generated control FASTAs.
- `data/embeddings/<model>/<control>.npz`: mean-pooled embeddings for
  `esm2_650m`, `esm3_open`, and `esmc_600m`.

## Run

```bash
conda activate beyond-nativeness
export BEYOND_NATIVENESS_ROOT="$(git rev-parse --show-toplevel)"
```

1. Generate FASTAs:
   - `jobs/01_generate_random.sh`
2. Embed controls:
   - `jobs/02a_embed_gpu.sh`: ESM2-650M and ESM3-open on a local GPU.
   - `jobs/02c_embed_esmc_api.sh`: ESMC-600M through Forge.

## Scripts

- `scripts/generate_random_sequences.py`: random and shuffled sequence pools.
- `scripts/extract_embeddings.py`: ESM2 embeddings.
- `scripts/extract_embeddings_esm3.py`: ESM3-open embeddings.
- `scripts/extract_embeddings_esmc_api.py`: ESMC embeddings through Forge.
