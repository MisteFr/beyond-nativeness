#!/usr/bin/env python3
"""
_io.py — shared I/O + pool definitions for the cross-architecture nativeness study.

Single source of truth for:
  * the sequence pools (same FASTAs the ESM appendix used, so results are comparable),
  * the on-disk output contract the figure (`appfig_pca_ppl_nonesm.py`) reads:
        embeddings -> data/embeddings/{model_key}/{emb_name}.npz   keys: embeddings[N,D] f32, accessions[N]
        human PPL  -> results/{model_key}/per_sequence_results.tsv (cols incl. accession,label,mean_perplexity)
        group PPL  -> results/{model_key}/{group}/per_sequence_results.tsv (cols accession,mean_perplexity,...)

Each scorer (score_progen2/evodiff/prott5.py) only implements (a) embedding and
(b) per-sequence perplexity for its architecture; everything else lives here so the
three scorers emit byte-identical file layouts.

Accession convention (matches every ESM script): first whitespace token of the FASTA
header, sequence upper-cased. Because embeddings and PPL for a given pool come from the
SAME FASTA, accessions join exactly with no prefix stripping.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
# Resolves to <repo>/projects by default; override with BEYOND_NATIVENESS_ROOT
# (mirrors paper_figures/scripts/_common.py). nat_io.py lives at
# projects/cross_architecture_nativeness/scripts/, so parents[3] is the repo root.
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
LAB  = Path(os.environ.get("BEYOND_NATIVENESS_ROOT", _DEFAULT_REPO_ROOT)) / "projects"
HV   = LAB / "esm_viral_probe/datasets/human_virus/data/processed"
PH   = LAB / "prokaryote_phage_ood/data/processed"
SH   = LAB / "prokaryote_phage_ood/data/shuffled"

ROOT     = LAB / "cross_architecture_nativeness"
EMB_ROOT = ROOT / "data/embeddings"
PPL_ROOT = ROOT / "results"

# Phage/cellular OOD groups (cellular = native, the three *_virus/phage = viral).
PHAGE_GROUPS   = ["bacteria", "archaea", "phage", "fungi", "plants",
                  "insects", "plant_virus", "invertebrate_virus"]
VIRAL_PHAGE    = {"phage", "plant_virus", "invertebrate_virus"}
CONTROL_GROUPS = ["shuffled_viral", "shuffled_nonviral", "random_uniform"]


# ---------------------------------------------------------------------------
# FASTA parsing (identical convention to the ESM pipeline)
# ---------------------------------------------------------------------------
def parse_fasta(path) -> list[tuple[str, str]]:
    """Return list of (accession, sequence). Accession = first whitespace token."""
    records, header, parts = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header.split()[0], "".join(parts)))
                header, parts = line[1:], []
            else:
                parts.append(line.upper())
    if header is not None:
        records.append((header.split()[0], "".join(parts)))
    return records


# ---------------------------------------------------------------------------
# Pool spec — the single task list every scorer iterates
# ---------------------------------------------------------------------------
def human_tasks() -> Iterator[dict]:
    for label in ["viral", "nonviral"]:
        for split in ["train", "val", "test"]:
            yield dict(kind="human", fasta=HV / f"{label}_{split}.faa",
                       label=label, split=split, emb_name=f"{label}_{split}", group=None)


def phage_tasks() -> Iterator[dict]:
    for g in PHAGE_GROUPS:
        yield dict(kind="phage", fasta=PH / f"{g}_clean.faa",
                   label=("viral" if g in VIRAL_PHAGE else "cellular"),
                   split=None, emb_name=g, group=g)


def control_tasks() -> Iterator[dict]:
    fmap = {
        "shuffled_viral":    SH / "shuffled_viral.faa",
        "shuffled_nonviral": SH / "shuffled_nonviral.faa",
        "random_uniform":    SH / "random_uniform.faa",
    }
    for g in CONTROL_GROUPS:
        yield dict(kind="control", fasta=fmap[g], label=g, split=None, emb_name=g, group=g)


def all_tasks(which: str = "all") -> list[dict]:
    """which ∈ {all, human, phage, controls} OR a comma-separated list of individual
    group names (e.g. 'shuffled_viral' or 'archaea,plants') — the latter lets a job
    process one or a few groups so the work can be split across parallel SLURM jobs
    (finer than the human/phage/controls granularity). Additive: the coarse selectors
    behave exactly as before."""
    sel = {
        "human":    list(human_tasks()),
        "phage":    list(phage_tasks()),
        "controls": list(control_tasks()),
    }
    if which == "all":
        return sel["human"] + sel["phage"] + sel["controls"]
    if which in sel:
        return sel[which]
    # fine-grained: comma-separated phage/control group names
    by_group = {t["group"]: t for t in (sel["phage"] + sel["controls"]) if t["group"]}
    wanted = [w.strip() for w in which.split(",") if w.strip()]
    bad = [w for w in wanted if w not in by_group]
    if bad:
        raise ValueError(f"unknown pool selector(s) {bad!r}; valid: "
                         f"all/human/phage/controls or groups {sorted(by_group)}")
    return [by_group[w] for w in wanted]


# ---------------------------------------------------------------------------
# Output paths + writers (the contract the figure depends on)
# ---------------------------------------------------------------------------
def emb_path(model_key: str, emb_name: str) -> Path:
    return EMB_ROOT / model_key / f"{emb_name}.npz"


def human_ppl_path(model_key: str) -> Path:
    return PPL_ROOT / model_key / "per_sequence_results.tsv"


def group_ppl_path(model_key: str, group: str) -> Path:
    return PPL_ROOT / model_key / group / "per_sequence_results.tsv"


def save_embeddings(model_key: str, emb_name: str,
                    embeddings: np.ndarray, accessions) -> Path:
    p = emb_path(model_key, emb_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p,
                        embeddings=np.asarray(embeddings, dtype=np.float32),
                        accessions=np.asarray(accessions))
    return p


def write_tsv(path: Path, rows: list[dict]) -> Path:
    """Write list-of-dicts as TSV; header = keys of first row (must be consistent)."""
    if not rows:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[k]) for k in header) + "\n")
    return path


def write_ppl(model_key: str, human_rows: list[dict],
              group_rows: dict[str, list[dict]]) -> None:
    """human_rows -> one combined TSV; group_rows[group] -> per-group TSV."""
    if human_rows:
        write_tsv(human_ppl_path(model_key), human_rows)
    for group, rows in group_rows.items():
        if rows:
            write_tsv(group_ppl_path(model_key, group), rows)


# ---------------------------------------------------------------------------
# Pooling helper (mean over kept positions); used by every scorer
# ---------------------------------------------------------------------------
def mean_pool(hidden, keep_mask):
    """hidden [B,L,D] torch tensor, keep_mask [B,L] (1 = keep) -> [B,D] float32 ndarray."""
    m = keep_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    return pooled.float().cpu().numpy()
