#!/usr/bin/env python3
"""
analyze_zeroshot_ppl.py
=======================
Unified analysis of zero-shot viral classification across all models.

Two classifiers:
  - PPL classifier:  higher PPL → viral (length-normalized)
  - LL classifier:   lower (more negative) total LL → viral (length-dependent)

Cross-references probe AUC from esm_viral_probe for comparison.

Produces 6 publication-quality figures + comparison_summary.json.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
PRETRAINED_MODELS = [
    {"key": "esm2_8m",        "name": "ESM2-8M",        "family": "ESM2",  "params_M": 8},
    {"key": "esm2_35m",       "name": "ESM2-35M",       "family": "ESM2",  "params_M": 35},
    {"key": "esm2_150m",      "name": "ESM2-150M",      "family": "ESM2",  "params_M": 150},
    {"key": "esm2_650m",      "name": "ESM2-650M",      "family": "ESM2",  "params_M": 650},
    {"key": "esm2_3b",        "name": "ESM2-3B",        "family": "ESM2",  "params_M": 3000},
    {"key": "esm2_15b",       "name": "ESM2-15B",       "family": "ESM2",  "params_M": 15000},
    {"key": "esmc_300m",      "name": "ESMC-300M",      "family": "ESMC",  "params_M": 300},
    {"key": "esmc_600m",      "name": "ESMC-600M",      "family": "ESMC",  "params_M": 600},
    {"key": "esm3_open",      "name": "ESM3 Open (1.4B)",  "family": "ESM3",  "params_M": 1400},
    {"key": "esm3_medium",    "name": "ESM3 Medium (7B)", "family": "ESM3",  "params_M": 7000},
    {"key": "esm3_large",     "name": "ESM3 Large (98B)", "family": "ESM3",  "params_M": 98000},
    {"key": "esmc_6b",        "name": "ESMC-6B",        "family": "ESMC",  "params_M": 6000},
]

# Models with corresponding probe results
PROBE_KEY_MAP = {
    "esm2_8m": "esm2_8m",
    "esm2_35m": "esm2_35m",
    "esm2_150m": "esm2_150m",
    "esm2_650m": "esm2_650m",
    "esm2_3b": "esm2_3b",
    "esm2_15b": "esm2_15b",
    "esmc_300m": "esmc_300m",
    "esmc_600m": "esmc_600m",
    "esmc_6b": "esmc_6b",
    "esm3_open": "esm3_open",
    "esm3_medium": "esm3_medium",
    "esm3_large": "esm3_large",
}

FT_MODELS = [
    {"key": "esmc_600m_ft_viral_lr1e4", "name": "ESMC-600M FT lr1e-4", "base": "esmc_600m"},
    {"key": "esmc_600m_ft_viral_lr2e4", "name": "ESMC-600M FT lr2e-4", "base": "esmc_600m"},
]

# Color palettes by family. ESM2 ordered light→dark by model size (8M → 15B).
FAMILY_COLORS = {
    "ESM2": ["#AEC6E8", "#6BAED6", "#2171B5", "#08306B", "#041B43", "#010A20"],  # blues
    "ESMC": ["#ff7f0e", "#ffa74d", "#ffc98a"],              # oranges
    "ESM3": ["#2ca02c", "#5cc35c", "#8dd68d", "#bee9be"],   # greens
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tsv(tsv_path: str) -> list[dict]:
    """Load per-sequence results TSV. Returns list of dicts."""
    rows = []
    with open(tsv_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            row = {
                "accession": r["accession"],
                "label": r["label"],
                "split": r.get("split", "unknown"),
                "length": int(r["length"]),
                "mean_perplexity": float(r["mean_perplexity"]),
                "mean_recovery_rate": float(r["mean_recovery_rate"]),
            }
            # Log-likelihood: use exact value if available, else approximate
            if "mean_log_likelihood" in r:
                row["mean_log_likelihood"] = float(r["mean_log_likelihood"])
            else:
                # Approximate: LL = -log(PPL) * n_masked
                # n_masked ≈ ceil(0.15 * length) but at least 1
                n_masked = max(1, math.ceil(0.15 * row["length"]))
                row["mean_log_likelihood"] = -math.log(row["mean_perplexity"]) * n_masked

            if "n_masked" in r:
                row["n_masked"] = int(r["n_masked"])
            else:
                row["n_masked"] = max(1, math.ceil(0.15 * row["length"]))

            rows.append(row)
    return rows


def compute_metrics(rows: list[dict]) -> dict:
    """Compute AUC, Mann-Whitney, means for viral vs nonviral."""
    viral_ppl = [r["mean_perplexity"] for r in rows if r["label"] == "viral"]
    nonviral_ppl = [r["mean_perplexity"] for r in rows if r["label"] == "nonviral"]
    viral_ll = [r["mean_log_likelihood"] for r in rows if r["label"] == "viral"]
    nonviral_ll = [r["mean_log_likelihood"] for r in rows if r["label"] == "nonviral"]

    if not viral_ppl or not nonviral_ppl:
        return None

    # Binary labels (viral=1)
    labels = [1 if r["label"] == "viral" else 0
              for r in rows if r["label"] in ("viral", "nonviral")]
    ppl_scores = [r["mean_perplexity"]
                  for r in rows if r["label"] in ("viral", "nonviral")]
    ll_scores = [-r["mean_log_likelihood"]
                 for r in rows if r["label"] in ("viral", "nonviral")]

    auc_ppl = roc_auc_score(labels, ppl_scores)
    auc_ll = roc_auc_score(labels, ll_scores)

    # Optimal PPL threshold by Youden's J (TPR - FPR), restricted to finite PPL values
    fpr, tpr, thr = roc_curve(labels, ppl_scores)
    finite = np.isfinite(thr)
    j = tpr[finite] - fpr[finite]
    best = int(np.argmax(j))
    ppl_optimal_threshold = float(thr[finite][best])
    ppl_optimal_tpr = float(tpr[finite][best])
    ppl_optimal_fpr = float(fpr[finite][best])

    mwu_ppl = stats.mannwhitneyu(viral_ppl, nonviral_ppl, alternative="two-sided")
    mwu_ll = stats.mannwhitneyu(viral_ll, nonviral_ll, alternative="two-sided")

    return {
        "n_viral": len(viral_ppl),
        "n_nonviral": len(nonviral_ppl),
        "viral_mean_ppl": float(np.mean(viral_ppl)),
        "viral_std_ppl": float(np.std(viral_ppl)),
        "nonviral_mean_ppl": float(np.mean(nonviral_ppl)),
        "nonviral_std_ppl": float(np.std(nonviral_ppl)),
        "viral_mean_ll": float(np.mean(viral_ll)),
        "viral_std_ll": float(np.std(viral_ll)),
        "nonviral_mean_ll": float(np.mean(nonviral_ll)),
        "nonviral_std_ll": float(np.std(nonviral_ll)),
        "auc_ppl": auc_ppl,
        "auc_ll": auc_ll,
        "ppl_optimal_threshold": ppl_optimal_threshold,
        "ppl_optimal_tpr": ppl_optimal_tpr,
        "ppl_optimal_fpr": ppl_optimal_fpr,
        "mwu_ppl_pvalue": float(mwu_ppl.pvalue),
        "mwu_ll_pvalue": float(mwu_ll.pvalue),
    }


def load_probe_auc(probe_dir: str, model_key: str) -> float | None:
    """Load probe AUC-ROC from esm_viral_probe results."""
    json_path = Path(probe_dir) / model_key / "test_results.json"
    if not json_path.exists():
        return None
    with open(json_path) as fh:
        d = json.load(fh)
    return d.get("linear", {}).get("auc_roc")


# ---------------------------------------------------------------------------
# Figure styling (scientific-figure-pro conventions)
# ---------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def get_model_color(model_info: dict, idx_in_family: int) -> str:
    family = model_info["family"]
    colors = FAMILY_COLORS.get(family, ["#999999"])
    return colors[min(idx_in_family, len(colors) - 1)]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_ppl_kde(all_data: dict, out_dir: Path):
    """PPL KDE overlay for all pretrained models."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    family_idx = {}
    for m in PRETRAINED_MODELS:
        if m["key"] not in all_data:
            continue
        fam = m["family"]
        family_idx.setdefault(fam, 0)
        color = get_model_color(m, family_idx[fam])
        family_idx[fam] += 1

        rows = all_data[m["key"]]
        viral_ppl = [r["mean_perplexity"] for r in rows if r["label"] == "viral"]
        nonviral_ppl = [r["mean_perplexity"] for r in rows if r["label"] == "nonviral"]

        for ax, vals, title in [(axes[0], nonviral_ppl, "Nonviral"),
                                (axes[1], viral_ppl, "Viral")]:
            vals_clip = [min(v, 40) for v in vals]
            ax.hist(vals_clip, bins=80, density=True, alpha=0.35,
                    color=color, label=m["name"], histtype="stepfilled")

    for ax, title in [(axes[0], "Nonviral"), (axes[1], "Viral")]:
        ax.set_xlabel("Perplexity")
        ax.set_title(title)
        ax.set_xlim(0, 35)
    axes[0].set_ylabel("Density")
    axes[1].legend(loc="upper right", fontsize=7)

    fig.suptitle("Zero-Shot PPL Distributions by Model", fontsize=13, y=1.02)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig1_ppl_kde_all_models.{ext}")
    plt.close(fig)
    print(f"  Saved fig1_ppl_kde_all_models")


def fig2_auc_bar(all_metrics: dict, out_dir: Path):
    """Grouped bar chart: probe AUC vs PPL AUC, organised by model family."""
    # Define family order and within-family order
    family_order = ["ESM2", "ESMC", "ESM3"]
    family_models = {fam: [] for fam in family_order}
    for m in PRETRAINED_MODELS:
        if m["key"] in all_metrics and m["family"] in family_models:
            family_models[m["family"]].append(m)

    # Build x positions with gaps between families
    gap = 0.8  # extra space between family groups
    width = 0.35
    positions = []
    names = []
    families_for_label = []  # (center_x, family_name)
    pos = 0
    for fam in family_order:
        models = family_models[fam]
        if not models:
            continue
        group_start = pos
        for m in models:
            positions.append(pos)
            names.append(m["name"])
            pos += 1
        group_end = pos - 1
        families_for_label.append(((group_start + group_end) / 2, fam))
        pos += gap  # gap before next family

    x = np.array(positions)
    ppl_aucs = []
    probe_aucs = []
    for fam in family_order:
        for m in family_models[fam]:
            ppl_aucs.append(all_metrics[m["key"]]["auc_ppl"])
            probe_aucs.append(all_metrics[m["key"]].get("probe_auc"))

    fig, ax = plt.subplots(figsize=(12, 5))

    # Probe AUC where available
    probe_vals = [v if v is not None else 0 for v in probe_aucs]
    bars = ax.bar(x - width / 2, probe_vals, width, label="Probe AUC", color="#1f77b4", alpha=0.8)
    for bar, v in zip(bars, probe_aucs):
        if v is None:
            bar.set_alpha(0)

    ppl_bars = ax.bar(x + width / 2, ppl_aucs, width, label="PPL AUC",
                      color="#2ca02c", alpha=0.8)

    # Annotate optimal PPL threshold (Youden's J) above each green bar
    ppl_thresholds = []
    for fam in family_order:
        for m in family_models[fam]:
            ppl_thresholds.append(all_metrics[m["key"]].get("ppl_optimal_threshold"))
    for bar, thr in zip(ppl_bars, ppl_thresholds):
        if thr is None or not np.isfinite(thr):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{thr:.2f}",
                ha="center", va="bottom", fontsize=7, color="#1b5e1b")

    ax.set_ylabel("AUC-ROC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.axhline(0.5, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.legend(loc="lower right")
    ax.set_title("Zero-Shot Classification: PPL vs Probe AUC\n"
                 "(PPL bar labels: optimal threshold)")

    # Add family labels below the x-axis tick labels
    for center_x, fam_name in families_for_label:
        ax.annotate(fam_name, xy=(center_x, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -38), textcoords="offset points",
                    ha="center", va="top", fontsize=11, fontweight="bold")

    # Draw light vertical separators between families
    sep_positions = []
    pos = 0
    for i, fam in enumerate(family_order):
        models = family_models[fam]
        if not models:
            continue
        pos += len(models)
        if i < len(family_order) - 1:
            sep_positions.append(pos - 1 + gap / 2)
        pos += gap
    for sx in sep_positions:
        ax.axvline(sx, color="#cccccc", ls="--", lw=0.8, alpha=0.6)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)  # extra room for family labels
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig2_auc_bar_chart.{ext}")
    plt.close(fig)
    print(f"  Saved fig2_auc_bar_chart")


def fig3_probe_vs_zeroshot(all_metrics: dict, out_dir: Path):
    """Scatter: probe AUC (x) vs zero-shot PPL AUC (y)."""
    models = [m for m in PRETRAINED_MODELS
              if m["key"] in all_metrics and all_metrics[m["key"]].get("probe_auc") is not None]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

    family_idx = {}
    for m in models:
        fam = m["family"]
        family_idx.setdefault(fam, 0)
        color = get_model_color(m, family_idx[fam])
        family_idx[fam] += 1

        probe_auc = all_metrics[m["key"]]["probe_auc"]
        zs_auc = all_metrics[m["key"]]["auc_ppl"]

        ax.scatter(probe_auc, zs_auc, color=color, s=60, zorder=5, edgecolors="k", lw=0.5)
        ax.annotate(m["name"], (probe_auc, zs_auc),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    lims = [0.8, 1.01]
    ax.plot(lims, lims, "k--", alpha=0.3, lw=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Probe AUC-ROC")
    ax.set_ylabel("Zero-Shot PPL AUC")
    ax.set_title("Probe vs PPL AUC")
    ax.set_aspect("equal")

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig3_probe_vs_zeroshot.{ext}")
    plt.close(fig)
    print(f"  Saved fig3_probe_vs_zeroshot")


def fig4_scaling(all_metrics: dict, out_dir: Path):
    """AUC vs parameter count (log scale), PPL and probe only."""
    models = [m for m in PRETRAINED_MODELS if m["key"] in all_metrics]

    params = [m["params_M"] for m in models]
    ppl_aucs = [all_metrics[m["key"]]["auc_ppl"] for m in models]

    probe_models = [m for m in models if all_metrics[m["key"]].get("probe_auc") is not None]
    probe_params = [m["params_M"] for m in probe_models]
    probe_aucs = [all_metrics[m["key"]]["probe_auc"] for m in probe_models]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(params, ppl_aucs, "o-", color="#2ca02c", label="PPL AUC", ms=6)
    if probe_aucs:
        ax.plot(probe_params, probe_aucs, "^-", color="#1f77b4", label="Probe AUC", ms=6)

    # Annotate
    for m in models:
        x = m["params_M"]
        y = all_metrics[m["key"]]["auc_ppl"]
        ax.annotate(m["name"], (x, y), textcoords="offset points",
                    xytext=(5, -10), fontsize=6, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.75, 1.02)
    ax.axhline(0.5, color="gray", ls="--", lw=0.5, alpha=0.3)
    ax.legend()
    ax.set_title("Zero-Shot Classification: Scaling with Model Size")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}M" if x < 1000 else f"{x/1000:.0f}B"))

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig4_scaling.{ext}")
    plt.close(fig)
    print(f"  Saved fig4_scaling")


def fig5_mean_ppl(all_metrics: dict, out_dir: Path):
    """Grouped bars: viral vs nonviral mean PPL per model."""
    models = [m for m in PRETRAINED_MODELS if m["key"] in all_metrics]

    names = [m["name"] for m in models]
    viral_ppl = [all_metrics[m["key"]]["viral_mean_ppl"] for m in models]
    viral_std = [all_metrics[m["key"]]["viral_std_ppl"] for m in models]
    nonviral_ppl = [all_metrics[m["key"]]["nonviral_mean_ppl"] for m in models]
    nonviral_std = [all_metrics[m["key"]]["nonviral_std_ppl"] for m in models]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, nonviral_ppl, width, yerr=nonviral_std, capsize=3,
           label="Nonviral", color="#1f77b4", alpha=0.8)
    ax.bar(x + width / 2, viral_ppl, width, yerr=viral_std, capsize=3,
           label="Viral", color="#d62728", alpha=0.8)

    ax.set_ylabel("Mean Perplexity")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_title("Mean Masked Reconstruction Perplexity by Class")

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig5_mean_ppl_comparison.{ext}")
    plt.close(fig)
    print(f"  Saved fig5_mean_ppl_comparison")


def fig6_ft_comparison(all_metrics: dict, out_dir: Path):
    """FT model comparison: baseline vs lr1e4 vs lr2e4."""
    ft_keys = ["esmc_600m", "esmc_600m_ft_viral_lr1e4", "esmc_600m_ft_viral_lr2e4"]
    ft_names = ["Baseline", "FT lr1e-4", "FT lr2e-4"]

    available = [k for k in ft_keys if k in all_metrics]
    if len(available) < 2:
        print("  Skipping fig6 — insufficient FT data")
        return

    names = [ft_names[ft_keys.index(k)] for k in available]
    ppl_aucs = [all_metrics[k]["auc_ppl"] for k in available]
    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # AUC comparison
    ax = axes[0]
    ax.bar(x, ppl_aucs, width, label="PPL AUC", color="#2ca02c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=0.5)
    ax.legend()
    ax.set_title("Zero-Shot PPL AUC: Effect of Fine-Tuning")

    # Mean PPL comparison
    ax = axes[1]
    viral_ppl = [all_metrics[k]["viral_mean_ppl"] for k in available]
    nonviral_ppl = [all_metrics[k]["nonviral_mean_ppl"] for k in available]
    ax.bar(x - width / 2, nonviral_ppl, width, label="Nonviral", color="#1f77b4", alpha=0.8)
    ax.bar(x + width / 2, viral_ppl, width, label="Viral", color="#d62728", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean Perplexity")
    ax.legend()
    ax.set_title("Mean PPL: Effect of Fine-Tuning")

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"fig6_ft_comparison.{ext}")
    plt.close(fig)
    print(f"  Saved fig6_ft_comparison")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True,
                        help="Root results directory with model subdirs")
    parser.add_argument("--probe_dir", required=True,
                        help="esm_viral_probe results dir")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for figures and summary")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    # ---- Load all data ----
    all_data = {}   # key -> list[dict] (per-sequence rows)
    all_metrics = {}  # key -> dict (aggregated metrics)

    all_models = PRETRAINED_MODELS + FT_MODELS
    for m in all_models:
        tsv_path = results_dir / m["key"] / "per_sequence_results.tsv"
        if not tsv_path.exists():
            print(f"  SKIP {m['key']}: {tsv_path} not found")
            continue

        rows = load_tsv(str(tsv_path))
        metrics = compute_metrics(rows)
        if metrics is None:
            print(f"  SKIP {m['key']}: no viral/nonviral data")
            continue

        all_data[m["key"]] = rows
        all_metrics[m["key"]] = metrics
        print(f"  Loaded {m['key']}: {metrics['n_viral']} viral, "
              f"{metrics['n_nonviral']} nonviral, "
              f"PPL AUC={metrics['auc_ppl']:.4f}, "
              f"LL AUC={metrics['auc_ll']:.4f}")

    # ---- Load probe AUCs ----
    for key, probe_key in PROBE_KEY_MAP.items():
        if key not in all_metrics:
            continue
        probe_auc = load_probe_auc(args.probe_dir, probe_key)
        if probe_auc is not None:
            all_metrics[key]["probe_auc"] = probe_auc
            print(f"  Probe AUC for {key}: {probe_auc:.4f}")

    if not all_metrics:
        print("ERROR: No models loaded. Check results_dir.")
        sys.exit(1)

    # ---- Generate figures ----
    print("\nGenerating figures...")
    fig1_ppl_kde(all_data, out_dir)
    fig2_auc_bar(all_metrics, out_dir)
    fig3_probe_vs_zeroshot(all_metrics, out_dir)
    fig4_scaling(all_metrics, out_dir)
    fig5_mean_ppl(all_metrics, out_dir)
    fig6_ft_comparison(all_metrics, out_dir)

    # ---- Save summary JSON ----
    summary = {}
    for m in all_models:
        if m["key"] not in all_metrics:
            continue
        entry = {
            "name": m["name"],
            "family": m.get("family", "FT"),
            "params_M": m.get("params_M", None),
        }
        entry.update(all_metrics[m["key"]])
        summary[m["key"]] = entry

    json_path = out_dir / "comparison_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved: {json_path}")

    # ---- Print summary table ----
    print("\n" + "=" * 100)
    print(f"{'Model':<22s} {'Family':<6s} {'Params':>8s} "
          f"{'V PPL':>8s} {'NV PPL':>8s} "
          f"{'PPL AUC':>8s} {'LL AUC':>8s} {'Probe AUC':>10s}")
    print("-" * 100)

    for m in all_models:
        if m["key"] not in all_metrics:
            continue
        d = all_metrics[m["key"]]
        params_str = (f"{m['params_M']}M" if m.get('params_M') and m['params_M'] < 1000
                      else f"{m['params_M']/1000:.0f}B" if m.get('params_M')
                      else "—")
        probe_str = f"{d['probe_auc']:.4f}" if d.get("probe_auc") else "—"
        print(f"{m['name']:<22s} {m.get('family','FT'):<6s} {params_str:>8s} "
              f"{d['viral_mean_ppl']:>8.2f} {d['nonviral_mean_ppl']:>8.2f} "
              f"{d['auc_ppl']:>8.4f} {d['auc_ll']:>8.4f} {probe_str:>10s}")
    print("=" * 100)


if __name__ == "__main__":
    main()
