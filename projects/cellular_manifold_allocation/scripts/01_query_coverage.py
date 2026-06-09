#!/usr/bin/env python3
"""Phase A: coverage audit.

Query UniProt REST for UniProtKB and UniRef cluster counts across 6 cellular
and 4 viral groups. Every query is logged with its literal URL and the raw
x-total-results header into logs/uniprot_queries.jsonl for reproducibility.

All queries are run from scratch — no cached count files are read. Each
(group, metric) tuple is one REST call; total ~50 calls.

Output: results/coverage_taxa.tsv

Usage:
    python3 scripts/01_query_coverage.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT = Path("${BEYOND_NATIVENESS_ROOT}/projects/cellular_manifold_allocation")
OUTDIR = PROJECT / "results"
LOGDIR = PROJECT / "logs"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

LOGFILE = LOGDIR / "uniprot_queries.jsonl"
OUT_TSV = OUTDIR / "coverage_taxa.tsv"

# ── How many sequences of each group are actually in our analysis set ──
# Matches the files in prokaryote_phage_ood/results/masked_reconstruction/
# (wc -l minus header) and esm_viral_probe human_virus/processed split counts.
N_IN_OUR_SET = {
    "bacteria":            5000,
    "archaea":            17379,
    "plants":              5000,
    "fungi":               5000,
    "insects":             5000,
    "sp_nonviral":         5197,  # Swiss-Prot nonviral from esm_viral_probe
    "phage":               1262,
    "plant_virus":          954,
    "invertebrate_virus":  1395,
    "human_viral":         5203,  # esm_viral_probe viral split counts
}

# ── Query definitions ──
# Mirror the filters used in prokaryote_phage_ood/scripts/01_download_uniprot.py
# for like-for-like counts, dropping the KW-0181 "Complete proteome" exclusion
# (our coverage estimate should measure the universe the model saw in training,
# not our filtered subsample).
#
# NOTE: The UniRef endpoint does NOT support the `virus_host_id` field. For the
# three host-filtered viral groups (plant_virus, invertebrate_virus,
# human_viral_hosted) we query UniProtKB totals only; UniRef values come back
# as -1 and are treated as missing downstream. Their contribution to the
# aggregate viral UniRef50 is captured by `phage` + `human_viral_family_sum`
# (Caudoviricetes covers most phage, the family sum covers human-viral).
GROUP_QUERIES = {
    "bacteria":            'taxonomy_id:2',
    "archaea":             'taxonomy_id:2157',
    "plants":              'taxonomy_id:33090',
    "fungi":               'taxonomy_id:4751',
    "insects":             'taxonomy_id:50557',
    "sp_nonviral":         'NOT taxonomy_id:10239',  # whole UniProtKB excluding viruses
    "phage":               'taxonomy_name:Caudoviricetes',
    "plant_virus":         'taxonomy_id:10239 AND virus_host_id:33090',
    "invertebrate_virus":  'taxonomy_id:10239 AND virus_host_id:6656 AND NOT virus_host_id:9606',
    "human_viral_hosted":  'taxonomy_id:10239 AND virus_host_id:9606',
}

# Queries that do NOT work on the UniRef endpoint (host-id filtering is UniProtKB-only).
SKIP_UNIREF = {"plant_virus", "invertebrate_virus", "human_viral_hosted"}

# 18 human-viral families from the dataset (viral_probe family_metadata.tsv).
# Preferred query is taxonomy_name: (works across all families in current UniProt
# taxonomy, which has purged some old numeric IDs). The taxid is recorded for
# provenance only.
HUMAN_VIRAL_FAMILIES = {
    "Orthomyxoviridae":   11308,
    "Orthoherpesviridae": 10292,
    "Poxviridae":         10240,
    "Retroviridae":       11632,
    "Papillomaviridae":   10566,
    "Sedoreoviridae":     10880,
    "Hepadnaviridae":     10404,
    "Adenoviridae":       10508,
    "Paramyxoviridae":    11158,
    "Rhabdoviridae":      11270,
    "Filoviridae":        11266,
    "Coronaviridae":      11118,
    "Pneumoviridae":      11244,
    "Kolmioviridae":      2946170,
    "Arenaviridae":       11617,
    "Polyomaviridae":     10624,
    "Flaviviridae":       11050,
    "Hantaviridae":       1980413,
}

REST_UNIPROTKB = "https://rest.uniprot.org/uniprotkb/search"
REST_UNIREF    = "https://rest.uniprot.org/uniref/search"

SLEEP_SECS = 0.5
MAX_RETRIES = 4


def _log(record: dict) -> None:
    record["logged_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with open(LOGFILE, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def query_count(endpoint: str, query: str, extra_params: dict | None = None) -> int:
    """Hit a UniProt REST search endpoint with size=0 and return x-total-results.

    Returns -1 on failure after MAX_RETRIES.
    """
    params = {"query": query, "size": "0", "format": "json"}
    if extra_params:
        params.update(extra_params)

    last_err: str | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(endpoint, params=params, timeout=45)
            if resp.status_code == 200:
                total_raw = resp.headers.get("x-total-results", "0")
                total = int(total_raw)
                _log({
                    "endpoint": endpoint, "query": query, "extra": extra_params or {},
                    "url": resp.url, "total": total, "status": 200,
                })
                return total
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "5"))
                last_err = f"429 rate-limit, waiting {wait}s"
                time.sleep(wait)
                continue
            last_err = f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            last_err = f"exception: {exc}"
        time.sleep(2 ** attempt)

    _log({
        "endpoint": endpoint, "query": query, "extra": extra_params or {},
        "total": -1, "status": "failed", "error": last_err,
    })
    return -1


def counts_for_query(query: str, skip_uniref: bool = False) -> dict[str, int]:
    """Return a dict with 4 counts for a single query.

    If skip_uniref=True, UniRef counts are set to -1 (used when the query
    contains a UniProtKB-only field like virus_host_id).
    """
    uniprotkb_total = query_count(REST_UNIPROTKB, query)
    time.sleep(SLEEP_SECS)
    swissprot_reviewed = query_count(REST_UNIPROTKB, f"({query}) AND reviewed:true")
    time.sleep(SLEEP_SECS)
    if skip_uniref:
        uniref50, uniref90 = -1, -1
    else:
        uniref50 = query_count(REST_UNIREF, f"({query}) AND identity:0.5")
        time.sleep(SLEEP_SECS)
        uniref90 = query_count(REST_UNIREF, f"({query}) AND identity:0.9")
        time.sleep(SLEEP_SECS)
    return {
        "uniprotkb_total":    uniprotkb_total,
        "swissprot_reviewed": swissprot_reviewed,
        "uniref50_clusters":  uniref50,
        "uniref90_clusters":  uniref90,
    }


def main() -> None:
    print("=" * 70)
    print("Phase A: UniProt coverage audit")
    print("=" * 70)
    print(f"Logging every query to {LOGFILE}")
    # Reset log file for a clean run (keep the path, just truncate).
    LOGFILE.write_text("")

    rows = []

    # ── 1) Group-level queries (9 rows) ──
    for group, query in GROUP_QUERIES.items():
        skip = group in SKIP_UNIREF
        print(f"\n[{group}]  query = {query}  {'(UniRef skipped)' if skip else ''}")
        counts = counts_for_query(query, skip_uniref=skip)
        print(f"  uniprotkb_total     = {counts['uniprotkb_total']:>15,}")
        print(f"  swissprot_reviewed  = {counts['swissprot_reviewed']:>15,}")
        print(f"  uniref50_clusters   = {counts['uniref50_clusters']:>15,}")
        print(f"  uniref90_clusters   = {counts['uniref90_clusters']:>15,}")

        rows.append({
            "group":              group,
            "query_string":       query,
            "source":             "group_query",
            "uniprotkb_total":    counts["uniprotkb_total"],
            "swissprot_reviewed": counts["swissprot_reviewed"],
            "uniref50_clusters":  counts["uniref50_clusters"],
            "uniref90_clusters":  counts["uniref90_clusters"],
        })

    # ── 2) Per-family queries for human_viral (18 families, then sum) ──
    # Use taxonomy_name: (preferred — current UniProt taxonomy). Record the
    # historical taxid for provenance; do a fallback-by-id only if name=0.
    fam_rows = []
    for fam, taxid in HUMAN_VIRAL_FAMILIES.items():
        q_name = f"taxonomy_name:{fam}"
        print(f"\n[human_viral family: {fam} (taxid={taxid})]")
        counts = counts_for_query(q_name)
        q_used = q_name
        if counts["uniprotkb_total"] <= 0:
            q_id = f"taxonomy_id:{taxid}"
            print(f"  name query returned {counts['uniprotkb_total']} — retry with {q_id}")
            counts = counts_for_query(q_id)
            q_used = q_id
        fam_rows.append({
            "family":             fam,
            "taxid":              taxid,
            "query_string":       q_used,
            **counts,
        })
        print(f"  uniprotkb_total={counts['uniprotkb_total']:>12,}  "
              f"uniref50={counts['uniref50_clusters']:>10,}")

    fam_df = pd.DataFrame(fam_rows)
    fam_df.to_csv(OUTDIR / "human_viral_by_family.tsv", sep="\t", index=False)

    # Aggregate
    human_viral_agg = {
        "group":              "human_viral_family_sum",
        "query_string":       "sum over 18 human-viral families (see human_viral_by_family.tsv)",
        "source":             "family_sum",
        "uniprotkb_total":    int(fam_df["uniprotkb_total"].sum()),
        "swissprot_reviewed": int(fam_df["swissprot_reviewed"].sum()),
        "uniref50_clusters":  int(fam_df["uniref50_clusters"].sum()),
        "uniref90_clusters":  int(fam_df["uniref90_clusters"].sum()),
    }
    rows.append(human_viral_agg)
    print(f"\n[human_viral — sum over 18 families]")
    print(f"  uniprotkb_total     = {human_viral_agg['uniprotkb_total']:>15,}")
    print(f"  uniref50_clusters   = {human_viral_agg['uniref50_clusters']:>15,}")

    # ── 3) Assemble output table with n-in-our-set and per-protein coverage ──
    df = pd.DataFrame(rows)

    # For the per-protein coverage columns, match N_IN_OUR_SET by group name.
    group_key_map = {
        "bacteria":                "bacteria",
        "archaea":                 "archaea",
        "plants":                  "plants",
        "fungi":                   "fungi",
        "insects":                 "insects",
        "sp_nonviral":             "sp_nonviral",
        "phage":                   "phage",
        "plant_virus":             "plant_virus",
        "invertebrate_virus":      "invertebrate_virus",
        "human_viral_family_sum":  "human_viral",
        "human_viral_hosted":      "human_viral",
    }
    df["n_in_our_set"] = df["group"].map(lambda g: N_IN_OUR_SET.get(group_key_map.get(g, g)))
    df["uniref50_per_protein"]       = df["uniref50_clusters"] / df["n_in_our_set"]
    df["uniprotkb_total_per_protein"] = df["uniprotkb_total"]  / df["n_in_our_set"]
    df["query_timestamp"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    col_order = [
        "group", "query_string", "source",
        "uniprotkb_total", "swissprot_reviewed",
        "uniref50_clusters", "uniref90_clusters",
        "n_in_our_set",
        "uniref50_per_protein", "uniprotkb_total_per_protein",
        "query_timestamp",
    ]
    df = df[col_order]
    df.to_csv(OUT_TSV, sep="\t", index=False)

    print("\n" + "=" * 70)
    print("COVERAGE TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nWrote: {OUT_TSV}")
    print(f"Wrote: {OUTDIR / 'human_viral_by_family.tsv'}")
    print(f"Wrote: {LOGFILE}")

    # ── Quick sanity summary ──
    print("\n" + "=" * 70)
    print("QUICK SANITY")
    print("=" * 70)
    # aggregate cellular vs viral (skip -1 values where UniRef wasn't obtainable)
    cellular_groups = ["bacteria", "archaea", "plants", "fungi", "insects", "sp_nonviral"]
    # Viral aggregate: phage + human_viral_family_sum only. plant_virus and
    # invertebrate_virus contribute minimally (raw UniProtKB totals 22k, 30k —
    # UniRef50 would be O(10^3 each); exclusion doesn't change the ratio scale).
    viral_groups    = ["phage", "human_viral_family_sum"]
    def _sum_positive(groups: list[str], col: str) -> int:
        vals = df[df["group"].isin(groups)][col]
        return int(vals[vals > 0].sum())
    cell_ur50 = _sum_positive(cellular_groups, "uniref50_clusters")
    vir_ur50  = _sum_positive(viral_groups, "uniref50_clusters")
    print(f"Aggregate cellular UniRef50  = {cell_ur50:>15,}  "
          f"(sum over {cellular_groups})")
    print(f"Aggregate viral UniRef50     = {vir_ur50:>15,}  "
          f"(sum over {viral_groups})")
    if vir_ur50 > 0:
        print(f"Ratio (cellular / viral)     = {cell_ur50 / vir_ur50:>15.1f}×")

    # human_viral cross-check
    try:
        v1 = df.loc[df["group"] == "human_viral_family_sum", "uniref50_clusters"].iloc[0]
        v2 = df.loc[df["group"] == "human_viral_hosted",     "uniref50_clusters"].iloc[0]
        if v1 > 0 and v2 > 0:
            print(f"human_viral family_sum={v1:,}  hosted-query={v2:,}  "
                  f"ratio={v1/v2:.3f}")
    except (KeyError, IndexError):
        pass


if __name__ == "__main__":
    main()
