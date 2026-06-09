# Cellular Manifold Allocation

This project produces the UniProt/UniRef coverage audit used in Table 1. It
queries UniProtKB and UniRef for cellular groups, viral groups, and a per-family
human-virus breakdown, then records sequence counts, reviewed counts, UniRef50
clusters, UniRef90 clusters, and per-protein coverage ratios.

Every request is logged with the literal URL and raw `x-total-results` value so
the table can be traced back to the API calls.

## Inputs

- Live access to the UniProt REST API:
  - `https://rest.uniprot.org/uniprotkb/search`
  - `https://rest.uniprot.org/uniref/search`


## Outputs

- `results/coverage_taxa.tsv`: Table 1 counts and coverage ratios.
- `results/human_viral_by_family.tsv`: per-family rows used for the
  `human_viral_family_sum` entry.
- `logs/uniprot_queries.jsonl`: endpoint, query string, resolved URL, and raw
  total for each API call.

## Run


```bash
python3 scripts/01_query_coverage.py
```

This is CPU-only. It does require network access to UniProt.
