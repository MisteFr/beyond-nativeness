"""Shared utilities, palette, and data loaders for the paper figures.

Every figure script imports from here so that house style, group colors, and
model-family ordering are identical across the set. No figure-level logic lives
in this module — only loaders and constants.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Paths --------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
# Output directory for rendered figures (override with BN_FIGURES_DIR).
FIGURES_DIR = Path(os.environ.get("BN_FIGURES_DIR", REPO_ROOT / "figures"))

# Figure inputs are the compact, vendored per-figure summaries under
# data/figure_data/ (per-sequence PPL tables, probe AUC JSONs, family metadata).
# Everything needed to render the paper figures is committed there, so a clean
# clone renders on a plain CPU machine with no GPU and no API access.
#
# To render against a freshly regenerated pipeline instead, point BN_FIGURE_DATA
# at a tree laid out like projects/<name>/results (see docs/reproduction_guide.md).
LAB_ROOT = Path(os.environ.get("BN_FIGURE_DATA", REPO_ROOT / "data" / "figure_data"))

# Precomputed 2-D PCA coordinates for the three PCA-scatter figures, committed so
# those figures render without the large embedding matrices. Always under the
# repository (independent of BN_FIGURE_DATA).
PCA_CACHE_DIR = REPO_ROOT / "data" / "figure_data" / "pca_coords"

PROBE_HV = LAB_ROOT / "esm_viral_probe/datasets/human_virus/results"
MASKED_RECON = LAB_ROOT / "esm3_masked_reconstruction/results"
ZEROSHOT = LAB_ROOT / "esm_zeroshot_ppl/results"
PHAGE_OOD = LAB_ROOT / "prokaryote_phage_ood/results"
POSTCUT = LAB_ROOT / "postcutoff_nonviral/results"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- Vendored scientific-figure-pro module -----------------------------------

_SFP_PATH = HERE / "_sfp.py"


def load_sfp():
    spec = importlib.util.spec_from_file_location("scientific_figure_pro", _SFP_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SFP = load_sfp()
PALETTE = SFP.PALETTE

# Model-family colors for grouped bars (appendix control figures).
FAMILY_COLOR = {
    "Baseline": PALETTE["neutral"],
    "ESM2": PALETTE["blue_secondary"],
    "ESMC": PALETTE["green_3"],
    "ESM3": PALETTE["red_strong"],
}


def apply_style(font_size: int = 13, axes_linewidth: float = 1.8) -> None:
    SFP.apply_publication_style(SFP.FigureStyle(font_size=font_size, axes_linewidth=axes_linewidth))


def finalize(fig, name: str, pad: float = 0.1, formats=("pdf", "png")) -> None:
    """Save a figure as PDF/PNG (and optionally SVG) under <repo>/figures/."""
    base = FIGURES_DIR / name
    SFP.finalize_figure(fig, str(base), formats=list(formats), dpi=300, pad=pad, close=True)

# --- Model registries --------------------------------------------------------

# Ordered by parameter count within each family.
MODEL_FAMILIES: dict[str, list[tuple[str, str, int]]] = {
    "ESM2":  [
        ("esm2_8m",   "8M",   int(8e6)),
        ("esm2_35m",  "35M",  int(35e6)),
        ("esm2_150m", "150M", int(150e6)),
        ("esm2_650m", "650M", int(650e6)),
        ("esm2_3b",   "3B",   int(3e9)),
        ("esm2_15b",  "15B",  int(15e9)),
    ],
    "ESMC": [
        ("esmc_300m", "300M", int(300e6)),
        ("esmc_600m", "600M", int(600e6)),
        ("esmc_6b",   "6B",   int(6e9)),
    ],
    "ESM3": [
        ("esm3_small",  "small",  int(1.4e9)),
        ("esm3_open",   "open",   int(1.4e9)),
        ("esm3_medium", "medium", int(7e9)),
        ("esm3_large",  "large",  int(98e9)),
    ],
}

ALL_MODELS_ORDERED = [m for fam in ("ESM2", "ESMC", "ESM3") for m, _, _ in MODEL_FAMILIES[fam]]


def pretty(model_key: str) -> str:
    """Human-readable model label, e.g. 'esmc_600m' -> 'ESMC-600M'."""
    for fam, lst in MODEL_FAMILIES.items():
        for mk, label, _ in lst:
            if mk == model_key:
                return f"{fam}-{label}"
    return model_key


# --- Data loaders ------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def probe_auc(model_key: str) -> float:
    obj = load_json(PROBE_HV / model_key / "test_results.json")
    return obj["linear"]["auc_roc"] if "linear" in obj else obj["auc_roc"]


def baseline_aucs() -> dict[str, float]:
    return {
        k: v["auc_roc"] for k, v in load_json(PROBE_HV / "baseline/summary.json").items()
    }


def human_neg_summary() -> dict:
    """Ctrl 3 — per-model full (multi-kingdom) vs human-only-negative AUC.

    Shape: {model_key: {full_auc, human_auc, n_human, frac_human}}. The
    human-only negatives are the test-split non-viral sequences with OX=9606.
    """
    return load_json(PROBE_HV / "human_neg_summary.json")


def leave_family_out_summary() -> dict:
    """Ctrl 6 — leave-one-viral-family-out held-out AUC.

    Shape: {min_family_size, threshold, families: [...],
    models: {model_key: {family: {auc_roc, sensitivity, n_test}}}}.
    """
    return load_json(PROBE_HV / "leave_family_out_summary.json")


# Probe dir → masked-reconstruction PPL dir mapping. Only the ESM3 "small"
# variant differs (probe uses the HF snapshot, PPL was computed via the ESM3
# API endpoint).
_PPL_DIR_ALIAS = {"esm3_small": "esm3_small_api"}


def per_seq_ppl(model_key: str) -> pd.DataFrame:
    """Per-sequence masked-reconstruction PPL for one model (human-virus set).

    Columns guaranteed: accession, label, mean_perplexity.
    """
    alias = _PPL_DIR_ALIAS.get(model_key, model_key)
    return pd.read_csv(ZEROSHOT / alias / "per_sequence_results.tsv", sep="\t")


def esmc_600m_per_seq() -> pd.DataFrame:
    return pd.read_csv(MASKED_RECON / "esmc_600m/per_sequence_results.tsv", sep="\t")


# --- PCA coordinate cache ----------------------------------------------------
#
# The three PCA-scatter figures fit a 2-component PCA on large per-model
# embedding matrices. Those matrices are too big to ship, so each figure caches
# its *projected* output here: per display-group 2-D coordinates (already
# PC1-oriented) plus the per-model summary stats (variance explained, rho, n).
# When the cache is present the figure renders from these few-MB files with no
# embeddings; when absent, the figure rebuilds it from embeddings (regeneration).

def load_pca_cache(fig_name: str, model_key: str):
    """Return (group_coords, meta) or None if the cache is absent.

    group_coords: {group_key: (Z[n, 2], ppl[n])}
    meta: {"evr": (pc1_var, pc2_var), "rho": float, "n": int,
           "rho_within": {group_key: float | None}}
    """
    path = PCA_CACHE_DIR / f"{fig_name}__{model_key}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    groups = sorted(k[: -len("__Z")] for k in d.files if k.endswith("__Z"))
    group_coords = {g: (d[f"{g}__Z"], d[f"{g}__ppl"]) for g in groups}
    rho_within = {g: (float(d[f"rhw__{g}"]) if f"rhw__{g}" in d.files else None)
                  for g in groups}
    meta = {
        "evr": (float(d["evr"][0]), float(d["evr"][1])),
        "rho": float(d["rho"]),
        "n": int(d["n"]),
        "rho_within": rho_within,
    }
    return group_coords, meta


def save_pca_cache(fig_name: str, model_key: str, group_coords: dict,
                   meta: dict) -> None:
    """Persist a PCA figure's projected coordinates + summary stats."""
    PCA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arrs: dict = {}
    for g, (Z, ppl) in group_coords.items():
        arrs[f"{g}__Z"] = np.asarray(Z, dtype=np.float32)
        arrs[f"{g}__ppl"] = np.asarray(ppl, dtype=np.float32)
    arrs["evr"] = np.asarray(meta["evr"], dtype=np.float64)
    arrs["rho"] = np.asarray(meta["rho"], dtype=np.float64)
    arrs["n"] = np.asarray(meta["n"], dtype=np.int64)
    for g, v in meta.get("rho_within", {}).items():
        if v is not None:
            arrs[f"rhw__{g}"] = np.asarray(v, dtype=np.float64)
    np.savez_compressed(PCA_CACHE_DIR / f"{fig_name}__{model_key}.npz", **arrs)
