"""Appendix figure: post-cutoff non-viral sequence-novelty control.

This figure supports the paragraph in §4.1 that distinguishes pretraining
coverage from evolutionary regime as explanations for the cellular-to-viral
perplexity shift. If the shift were a simple memorization / exposure effect,
then non-viral proteins created *after* the model's training-data cutoff — and
therefore unlikely to have been seen by the model — should behave more like
viral proteins than like pre-cutoff non-viral proteins. They do not.

Data source
-----------
ESMC-600M was released as ``esmc-600m-2024-12``; training data was frozen
several months earlier. We use 2025-01-01 as a conservative cutoff and pull
1,723 reviewed Swiss-Prot non-viral entries whose ``date_created`` falls on
or after that date (length 50-1022, <5%% non-standard AAs, dedup by sha256).
Their masked-reconstruction PPL is computed under the same recipe as the
rest of the paper (15%% mask, 3 seeds).

Two panels matching the paper's house style:
  A — overlaid smooth KDE of per-sequence PPL for the three groups
  B — box + jittered strip per group on the same colormap used in Fig 1B,
      median and n annotated per row.

Outputs:
  ../figures/appfig_postrelease_control.{pdf,png}
  _appfig_postrelease_control.json (numbers locked into the LaTeX text)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats as sstats
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

import _common as c


# --- Paths -------------------------------------------------------------------

POSTCUT_TSV  = c.POSTCUT / "esmc_600m/per_sequence_results.tsv"
PRECUT_TSV   = c.MASKED_RECON / "esmc_600m/per_sequence_results.tsv"


# --- Colormap (matches main Fig 1B per-group strip) -------------------------

VMIN, VMAX = 2.0, 25.0
_BRG = mpl.colormaps["brg"]
CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "BlueVioletRed", _BRG(np.linspace(0.0, 0.5, 256))
)
_NORM = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)


def color_for_ppl(ppl: float):
    return CMAP(_NORM(float(np.clip(ppl, VMIN, VMAX))))


# --- Groups ------------------------------------------------------------------

GROUPS = [
    ("precutoff_nonviral",  "Pre-release non-viral",  c.PALETTE["blue_main"]),
    ("postcutoff_nonviral", "Post-release non-viral", c.PALETTE["blue_secondary"]),
    ("viral",               "Human viral",            c.PALETTE["red_strong"]),
]


def load_groups() -> dict[str, np.ndarray]:
    pre = pd.read_csv(PRECUT_TSV, sep="\t")
    post = pd.read_csv(POSTCUT_TSV, sep="\t")
    return {
        "precutoff_nonviral":  pre.loc[pre["label"] == "nonviral",
                                       "mean_perplexity"].to_numpy(dtype=float),
        "postcutoff_nonviral": post["mean_perplexity"].to_numpy(dtype=float),
        "viral":               pre.loc[pre["label"] == "viral",
                                       "mean_perplexity"].to_numpy(dtype=float),
    }


def compute_stats(data: dict[str, np.ndarray]) -> dict:
    pre, post, vir = data["precutoff_nonviral"], data["postcutoff_nonviral"], data["viral"]

    def _auc(a: np.ndarray, b: np.ndarray) -> float:
        # Label 1 = "viral", score = -PPL. Report max(x, 1-x) so higher = better
        # separation regardless of sign, matching the paper's convention.
        y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
        s = np.concatenate([-a, -b])
        raw = roc_auc_score(y, s)
        return float(max(raw, 1.0 - raw))

    pooled_std = float(np.sqrt((pre.var() + post.var()) / 2.0))
    cohens_d   = float((post.mean() - pre.mean()) / pooled_std) if pooled_std > 0 else 0.0

    mwu_post_pre = sstats.mannwhitneyu(post, pre, alternative="two-sided")
    mwu_post_vir = sstats.mannwhitneyu(post, vir, alternative="two-sided")
    mwu_pre_vir  = sstats.mannwhitneyu(pre, vir, alternative="two-sided")

    return {
        "precutoff_nonviral":  {"n": int(len(pre)),
                                "mean_ppl": float(pre.mean()),
                                "median_ppl": float(np.median(pre)),
                                "std_ppl": float(pre.std(ddof=0))},
        "postcutoff_nonviral": {"n": int(len(post)),
                                "mean_ppl": float(post.mean()),
                                "median_ppl": float(np.median(post)),
                                "std_ppl": float(post.std(ddof=0))},
        "viral":               {"n": int(len(vir)),
                                "mean_ppl": float(vir.mean()),
                                "median_ppl": float(np.median(vir)),
                                "std_ppl": float(vir.std(ddof=0))},
        "cohens_d_post_vs_pre":  cohens_d,
        "auc_postcut_vs_viral":  _auc(post, vir),
        "auc_precut_vs_viral":   _auc(pre, vir),
        "mannwhitney_post_vs_pre_p":   float(mwu_post_pre.pvalue),
        "mannwhitney_post_vs_viral_p": float(mwu_post_vir.pvalue),
        "mannwhitney_pre_vs_viral_p":  float(mwu_pre_vir.pvalue),
    }


# --- Panel A: KDE overlay ----------------------------------------------------

def panel_kde(ax, data: dict[str, np.ndarray]) -> None:
    x_grid = np.linspace(0.0, 30.0, 800)
    for key, label, col in GROUPS:
        vals = data[key]
        kde = gaussian_kde(vals, bw_method=0.30)
        y = kde(x_grid)
        ax.plot(x_grid, y, color=col, lw=2.4,
                label=f"{label} (n={len(vals):,})", zorder=3)
        ax.fill_between(x_grid, y, color=col, alpha=0.18, zorder=2)

    ax.set_xlim(0.0, 28.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Masked-reconstruction PPL")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


# --- Panel B: box + strip ----------------------------------------------------

def panel_strip(ax, data: dict[str, np.ndarray], rng,
                max_points: int = 400) -> None:
    # Sort by median descending so the viral row sits at top, matching Fig 1B.
    stats_rows = []
    for key, label, _ in GROUPS:
        vals = data[key]
        stats_rows.append({"key": key, "label": label, "vals": vals,
                           "n": len(vals), "median": float(np.median(vals))})
    stats_rows.sort(key=lambda s: -s["median"])

    y_positions = np.arange(len(stats_rows))

    for y, s in zip(y_positions, stats_rows):
        col = color_for_ppl(s["median"])
        vals = s["vals"]

        if len(vals) > max_points:
            sample = vals[rng.choice(len(vals), size=max_points, replace=False)]
        else:
            sample = vals
        jitter = rng.uniform(-0.22, 0.22, size=len(sample))
        ax.scatter(np.clip(sample, VMIN, VMAX), y + jitter,
                   s=8, color=col, alpha=0.32,
                   edgecolors="none", rasterized=True, zorder=2)

        ax.boxplot(
            [vals], positions=[y], vert=False, widths=0.48,
            whis=(5, 95), showfliers=False, patch_artist=True,
            medianprops=dict(color="white", lw=1.6, zorder=6),
            boxprops=dict(facecolor=col, edgecolor="black", lw=0.9,
                          alpha=0.92, zorder=5),
            whiskerprops=dict(color="black", lw=0.9, zorder=4),
            capprops=dict(color="black", lw=0.9, zorder=4),
        )

        ax.text(26.0, y, f"n={s['n']:,}",
                va="center", ha="right", color="#333", fontsize=10, zorder=7)

    ax.set_xlim(VMIN, 26.2)
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_ylim(-0.6, len(stats_rows) - 0.4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s["label"] for s in stats_rows])
    ax.invert_yaxis()
    ax.set_xlabel("Masked-reconstruction PPL")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for y in y_positions[:-1]:
        ax.axhline(y + 0.5, color="#E0E0E0", lw=0.5, zorder=0)


# --- Main --------------------------------------------------------------------

def main() -> None:
    c.apply_style(font_size=14, axes_linewidth=1.8)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "svg.fonttype": "path",
    })
    rng = np.random.default_rng(7)

    data  = load_groups()
    stats = compute_stats(data)
    print(json.dumps(stats, indent=2))

    fig = plt.figure(figsize=(13.0, 4.6))
    gs = GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28,
                  left=0.065, right=0.985, top=0.92, bottom=0.14)

    ax_kde   = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[0, 1])

    panel_kde(ax_kde, data)
    panel_strip(ax_strip, data, rng)

    for ax, letter, xoff in ((ax_kde, "A", -0.10), (ax_strip, "B", -0.20)):
        ax.text(xoff, 1.04, letter, transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="bottom", ha="right")

    c.finalize(fig, "appfig_postrelease_control")
    print("Saved figures/appfig_postrelease_control.{pdf,png}")

    with open(c.HERE / "_appfig_postrelease_control.json", "w") as fh:
        json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
