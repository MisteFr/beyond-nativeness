# Human Virus Dataset

This project builds the curated human-infecting viral protein pool used as the
viral positive class. It combines reviewed UniProtKB/Swiss-Prot entries with a
manual NCBI Virus export, keeps highly curated `NP_` RefSeq accessions from the
NCBI side, normalizes both sources, and exact-deduplicates the result.

The final pool has 5,815 unique sequences. The validation report also records
the family counts used in Table 2.

## Inputs

- UniProtKB REST query, fetched live:

  ```text
  reviewed:true AND virus_host_id:9606 AND (existence:1 OR existence:2 OR existence:3)
  ```

- `sequences-2.fasta`: manual NCBI Virus web export. The NCBI `[Host]` field is
  not searchable through Entrez, so this file is included here.

## Outputs

- `data/processed/human_virus_clean.faa`: merged, normalized, deduplicated FASTA.
- `data/processed/human_virus_clean.tsv`: accession, source, protein name,
  organism, taxid, length, non-standard amino-acid fraction, and length-filter
  status.
- `data/processed/ncbi_human_np.{faa,tsv}`: intermediate NCBI `NP_` subset.
- `data/stats_report.txt`: length distribution, non-standard residue counts, and
  per-family/per-organism composition.

The final FASTA is also committed under `data/human_virus/` for downstream use.

## Run

Set `BEYOND_NATIVENESS_ROOT` to the repository root and activate the environment:

```bash
conda activate beyond-nativeness
```

No GPU is needed. The first step needs network access to UniProt.

1. `jobs/01_download.sh`: download UniProt FASTA and metadata, and import the
   manual NCBI export.
2. `jobs/02_process.sh`: keep NCBI `NP_` accessions and write the validation
   report.
3. `jobs/03_clean_normalize.sh`: merge sources, normalize records,
   exact-deduplicate, and write `human_virus_clean.{faa,tsv}`.

The same steps can be run directly from `scripts/0*.py` if you need custom
arguments.
