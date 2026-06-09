# Data

This directory contains the sequence inputs and compact figure inputs used by the
paper. The FASTAs are checked in so the regeneration pipeline does not need to
hit UniProt or NCBI unless you want to rebuild the datasets from scratch. The
`figure_data/` files are the small summaries needed to redraw the figures on a
CPU-only machine.

To rebuild a FASTA, use the corresponding project under `projects/`. Each one
has the download, filtering, and validation scripts used to produce the files
below.

## `human_virus/`

Curated human-infecting viral proteins.

- `human_virus_clean.faa`: 5,815 unique sequences after exact deduplication of
  6,664 raw entries. Of these, 5,204 pass the standard ESM length filter
  (50-1022 aa).
- `human_virus_clean.tsv`: accession, source, protein name, organism, taxid,
  length, percent non-standard amino acids, and length-filter status.

The pool is the union of:

1. UniProt Swiss-Prot, 6,044 reviewed entries:

   ```text
   reviewed:true AND virus_host_id:9606 AND (existence:1 OR existence:2 OR existence:3)
   ```

2. NCBI RefSeq `NP_` entries, 620 sequences from a manual NCBI Virus export under
   a human-host filter.

Pipeline: `projects/human_virus_dataset/scripts/`.

## `prokaryote_phage_ood/`

Eight tree-of-life groups used in Fig. 1 and appendix Fig. 6.

| File | Count | UniProt query (PE1-3, KW-0181 excluded) |
|---|---:|---|
| `bacteria_clean.faa` | 5,000 | `taxonomy_id:2` |
| `archaea_clean.faa` | 17,379 | `taxonomy_id:2157` |
| `plants_clean.faa` | 5,000 | `taxonomy_id:33090` |
| `fungi_clean.faa` | 5,000 | `taxonomy_id:4751` |
| `insects_clean.faa` | 5,000 | `taxonomy_id:50557` |
| `phage_clean.faa` | 1,262 | `taxonomy_id:10239`, then filtered to phage-relevant lineages or organism names containing `phage` |
| `plant_virus_clean.faa` | 954 | `taxonomy_id:10239 AND virus_host_id:33090` |
| `invertebrate_virus_clean.faa` | 1,395 | `taxonomy_id:10239 AND virus_host_id:6656 AND NOT virus_host_id:9606` |

All groups use the same local filters: length in `[50, 1022]`, less than 5%
non-standard amino acids, exact-sequence deduplication by SHA-256, and a random
subsample to 5,000 sequences with seed 42 when needed.

Pipeline:
`projects/prokaryote_phage_ood/scripts/{01_download_uniprot.py,02_clean_filter.py}`.

## `controls/`

Biologically meaningless controls used as OOD anchors.

- `random_uniform.faa`: 5,000 sequences. Residues are sampled i.i.d. from the 20
  standard amino acids; lengths follow the empirical human-virus and non-viral
  pools.
- `shuffled_viral.faa`: 5,203 human-viral sequences with residue positions
  permuted. Length and composition are preserved; positional structure is not.
- `shuffled_nonviral.faa`: 5,197 shuffled non-viral sequences from the processed
  `esm_viral_probe` split.

Pipeline: `projects/esm_random_ood/scripts/generate_random_sequences.py`.

## `figure_data/`

Small, committed inputs for figure rendering.

- `<module>/results/.../*.tsv` and `*.json`: per-sequence perplexities, probe AUC
  summaries, and family metadata, laid out like the regeneration outputs under
  `projects/<module>/results/`.
- `pca_coords/*.npz`: precomputed display coordinates, variance explained,
  Spearman correlations, and sequence counts for the PCA scatter figures.

Figure scripts read this directory through `BN_FIGURE_DATA`, defaulting to
`data/figure_data/`. To point the scripts at regenerated outputs, set
`BN_FIGURE_DATA` to the matching `projects/<module>/results` tree.
