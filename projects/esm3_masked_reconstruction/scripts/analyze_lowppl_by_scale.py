#!/usr/bin/env python3
"""Analyse how low-perplexity viral sequences evolve with model scale.

For each model (ordered by parameter count), identify viral sequences whose
perplexity falls below the nonviral P95 threshold.

Outputs (results/lowppl_by_scale/):
  04_consistent_vs_scale.tsv   — sequences classified by how many models flag them
  per_model_lowppl.tsv         — per-sequence × per-model flag
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_DIR    = ROOT / "data"
OUT_DIR     = RESULTS_DIR / "lowppl_by_scale"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model catalogue (ordered by parameter count) ───────────────────────────
MODELS = [
    dict(name="ESMC 600M",     params="0.6B",  tsv=RESULTS_DIR / "esmc_600m"     / "per_sequence_results.tsv"),
    dict(name="ESM3 Open",     params="1.4B",  tsv=RESULTS_DIR / "per_sequence_results.tsv"),
    dict(name="ESMC 6B",       params="6B",    tsv=RESULTS_DIR / "esmc_6b"       / "per_sequence_results.tsv"),
    dict(name="ESM3 Medium",   params="7B",    tsv=RESULTS_DIR / "esm3_medium"   / "per_sequence_results.tsv"),
    dict(name="ESM3 Large",    params="98B",   tsv=RESULTS_DIR / "esm3_large"    / "per_sequence_results.tsv"),
]

# ── Optional metadata ──────────────────────────────────────────────────────
UNIPROT_TSV = Path("${BEYOND_NATIVENESS_ROOT}/projects/human_virus_dataset/data/raw/uniprot/human_virus.tsv.gz")

def _parse_family_from_lineage(lineage: str) -> str:
    """Extract virus family (*viridae) from UniProt taxonomic lineage string."""
    if pd.isna(lineage):
        return "Unknown"
    m = re.search(r'(\w+(?:viridae|dae))\s*\(family\)', lineage, re.IGNORECASE)
    if m:
        return m.group(1)
    return "Unknown"

def _strip_sp_prefix(acc: str) -> str:
    """Convert 'sp|P12345|NAME_HUMAN' or 'tr|...' → 'P12345', pass plain IDs through."""
    if pd.isna(acc):
        return acc
    m = re.match(r'(?:sp|tr|ref)\|([^|]+)\|', str(acc))
    return m.group(1) if m else str(acc)

def _load_uniprot_meta():
    try:
        df = pd.read_csv(UNIPROT_TSV, sep="\t", compression="gzip",
                         usecols=lambda c: c in
                             {"Entry", "Taxonomic lineage", "Organism", "Protein names"})
        df = df.rename(columns={
            "Entry": "accession",
            "Taxonomic lineage": "lineage",
            "Organism": "organism",
            "Protein names": "protein_name",
        })
        df["family"] = df["lineage"].apply(_parse_family_from_lineage)
        return df.drop_duplicates("accession").set_index("accession")
    except Exception as e:
        print(f"  [warn] Could not load UniProt metadata: {e}")
        return pd.DataFrame()

# ── Load all models ────────────────────────────────────────────────────────
print("Loading per-sequence results …")
frames = {}
for m in MODELS:
    df = pd.read_csv(m["tsv"], sep="\t")
    # Normalise accession format: strip 'sp|P12345|NAME' → 'P12345'
    df["accession"] = df["accession"].apply(_strip_sp_prefix)
    frames[m["name"]] = df
    print(f"  {m['name']:14s}  n={len(df):,}")

# ── Per-model thresholds (nonviral P95) ────────────────────────────────────
thresholds = {}
for m in MODELS:
    df = frames[m["name"]]
    nv_ppl = df.loc[df["label"] == "nonviral", "mean_perplexity"].values
    if len(nv_ppl) == 0:
        nv_ppl = df.loc[df["label"] == "non-viral", "mean_perplexity"].values
    thresholds[m["name"]] = float(np.percentile(nv_ppl, 95))
    print(f"  {m['name']:14s}  nonviral P95 = {thresholds[m['name']]:.2f}")

# ── Build viral-only wide dataframe ───────────────────────────────────────
print("\nBuilding viral perplexity matrix …")

# collect viral accessions present in all models
viral_sets = []
for m in MODELS:
    df = frames[m["name"]]
    vmask = df["label"].isin(["viral"])
    viral_sets.append(set(df.loc[vmask, "accession"].values))

common_viral = viral_sets[0]
for s in viral_sets[1:]:
    common_viral = common_viral & s
print(f"  Viral sequences common to all models: {len(common_viral):,}")

wide_rows = []
for m in MODELS:
    df = frames[m["name"]]
    vmask = df["label"].isin(["viral"]) & df["accession"].isin(common_viral)
    sub = df.loc[vmask, ["accession", "mean_perplexity"]].set_index("accession")
    sub = sub.rename(columns={"mean_perplexity": m["name"]})
    wide_rows.append(sub)

ppl_wide = pd.concat(wide_rows, axis=1)  # shape: (n_viral, n_models)
print(f"  Matrix shape: {ppl_wide.shape}")

# ── Binary low-ppl flags (below per-model nonviral P95) ───────────────────
flag_wide = pd.DataFrame(
    {m["name"]: ppl_wide[m["name"]] < thresholds[m["name"]] for m in MODELS},
    index=ppl_wide.index,
)
flag_wide["n_models_low"] = flag_wide.sum(axis=1)

# ── Save per-sequence flag table ───────────────────────────────────────────
flag_wide.to_csv(OUT_DIR / "per_model_lowppl.tsv", sep="\t")

# ── Summary TSV: per-sequence consistency ─────────────────────────────────
meta = _load_uniprot_meta()
summary = flag_wide.copy()
summary["mean_ppl_all_models"] = ppl_wide.mean(axis=1)
# Add metadata
if not meta.empty:
    for col in ["family", "organism", "protein_name"]:
        if col in meta.columns:
            summary[col] = meta[col].reindex(summary.index)

summary.to_csv(OUT_DIR / "04_consistent_vs_scale.tsv", sep="\t")

# Print summary table
print("\n=== Low-ppl viral sequences by number of models flagging them ===")
for n in range(len(MODELS) + 1):
    cnt = (summary["n_models_low"] == n).sum()
    pct = cnt / len(summary) * 100
    print(f"  Flagged by {n}/{len(MODELS)} models: {cnt:4d} ({pct:.1f}%)")

print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
