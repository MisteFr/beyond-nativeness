# Tree-of-Life Groups

This project builds the non-human biological groups used in the PCA and
per-group perplexity analyses: bacteria, archaea, plants, fungi, insects,
bacteriophage, plant viruses, and invertebrate viruses.

It also runs masked-reconstruction perplexity and mean-pooled embeddings for
ESMC-600M, ESM2-650M, and ESM3-open. These results feed the tree-of-life panels
in the main and appendix figures.

The assembled FASTAs are included in `data/processed/` here and in
`../../data/prokaryote_phage_ood/`. Exact UniProt queries are listed in
`../../data/README.md`.

## Inputs

- `BEYOND_NATIVENESS_ROOT`, pointing at the repository root.
- Network access to UniProt/NCBI only when regenerating FASTAs.
- A CUDA GPU for embedding and PPL jobs. Several jobs reuse entry points from
  `esm_viral_probe` and `esm_zeroshot_ppl`.

## Outputs

Under `results/`:

- `masked_reconstruction/<group>_ppl.tsv`: ESMC-600M per-group perplexity.
- `masked_reconstruction_esm2_650m/<group>/per_sequence_results.tsv`: ESM2-650M.
- `masked_reconstruction_esm3_open/<group>/per_sequence_results.tsv`: ESM3-open.
- `data/embeddings/<model>/<group>.npz`: mean-pooled embeddings.

## Run

Activate the environment and export the repo root first:

```bash
conda activate beyond-nativeness
export BEYOND_NATIVENESS_ROOT="$(git rev-parse --show-toplevel)"
```

1. Assemble FASTAs, optional because they are already included:
   - `jobs/01_download.sh`, `jobs/02_clean_filter.sh` for bacteria, archaea, and phage.
   - `jobs/12_download_expanded.sh`, `jobs/13_clean_filter_expanded.sh` for plants,
     fungi, insects, plant viruses, and invertebrate viruses.
2. Run ESMC-600M embeddings and perplexity:
   - `jobs/03_embed.sh`, `jobs/14_embed_expanded.sh`
   - `jobs/04a_recon_bacteria.sh`, `jobs/04b_recon_archaea.sh`, `jobs/04c_recon_phage.sh`
   - `jobs/15a_recon_plants.sh` through `jobs/15e_recon_invertebrate_virus.sh`
3. Run ESM2-650M and ESM3-open group embeddings/perplexity:
   - `jobs/30_embed_ppl_esm2_650m_phage_ood.sh`
   - `jobs/30_embed_ppl_esm3_open_phage_ood.sh`

## Scripts

- `scripts/01_download_uniprot.py`: host/lineage-filtered UniProt/NCBI download.
- `scripts/02_clean_filter.py`: length/composition filtering, deduplication, subsampling.
- `scripts/04_masked_reconstruction.py`: ESMC per-group masked reconstruction.
- `scripts/utils.py`: shared FASTA I/O and filters.
