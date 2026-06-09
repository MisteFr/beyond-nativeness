"""Per-family nativization of viral proteins under ESMC scaling.

Each line is one viral family; the x-axis is ESMC parameter count (log), and
the y-axis is the fraction of that family's sequences whose masked-reconstruction
PPL has fallen below a native-like threshold (PPL < 5, inside the non-viral IQR).

Top 8 families by ESMC-6B nativization rate are shown individually; all-viral
and all-non-viral population means are plotted as dashed / dotted references.

Source:
  esm_zeroshot_ppl/results/{esmc_300m,esmc_600m,esmc_6b}/per_sequence_results.tsv
  esm3_masked_reconstruction/results/lowppl_by_scale/04_consistent_vs_scale.tsv
      (for family annotations)
"""
from __future__ import annotations

import re

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as c


# --- Constants --------------------------------------------------------------

PPL_THRESHOLD    = 5.0
TOP_N_FAMILIES   = 8
N_SOLID          = 3           # top-N by final % with PPL<5, shown solid
MIN_N_PER_FAMILY = 50

# ESMC family only — absolute PPL scale is only comparable within one
# architecture (different tokenizers across ESM2/ESMC/ESM3).
ALL_MODELS: list[tuple[str, int]] = [
    (mk, params) for mk, _lbl, params in c.MODEL_FAMILIES["ESMC"]
]

FAMILY_META_TSV = (c.MASKED_RECON /
                   "lowppl_by_scale" / "04_consistent_vs_scale.tsv")

# Family colours: distinct hues drawn from the house palette so a single viral
# family reads as a single category across the paper. The "All viral" /
# "All non-viral" references below are rendered in neutral grey so they cannot
# be confused with individual families.
# ColorBrewer 8-class Accent — assigned in legend order (top-down by ESMC-6B
# nativization rank).
FAMILY_COLOR = {
    "Papillomaviridae":   "#7fc97f",
    "Retroviridae":       "#beaed4",
    "Adenoviridae":       "#fdc086",
    "Poxviridae":         "#00838f",
    "Coronaviridae":      "#386cb0",
    "Orthoherpesviridae": "#f0027f",
    "Orthomyxoviridae":   "#bf5b17",
    "Sedoreoviridae":     "#666666",
    "Hepadnaviridae":     "#beaed4",
    "Paramyxoviridae":    "#cccccc",
    "Rhabdoviridae":      "#386cb0",
    "Pneumoviridae":      "#bf5b17",
    "Filoviridae":        "#7fc97f",
    "Flaviviridae":       "#f0027f",
}


# --- Loaders ----------------------------------------------------------------

def _strip_sp(acc) -> str:
    if pd.isna(acc):
        return acc
    m = re.match(r'(?:sp|tr|ref)\|([^|]+)\|', str(acc))
    return m.group(1) if m else str(acc)


def _load_family_meta() -> pd.DataFrame:
    meta = pd.read_csv(FAMILY_META_TSV, sep="\t",
                       usecols=["accession", "family"])
    meta["accession"] = meta["accession"].apply(_strip_sp)
    return meta.drop_duplicates("accession")


def compute_family_pct(
    models: list[tuple[str, int]],
    meta: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.Series, np.ndarray]:
    """Per-model family % with PPL < PPL_THRESHOLD. No cross-model merge.

    Also returns the per-model fraction of the *entire* viral population with
    PPL < PPL_THRESHOLD ("all-viral" reference line).
    """
    # Family counts reference (from the first model's viral set).
    first = c.per_seq_ppl(models[0][0]).copy()
    first["accession"] = first["accession"].apply(_strip_sp)
    first = first[first["label"] == "viral"].merge(meta, on="accession", how="left")
    first["family"] = first["family"].fillna("Unknown")
    fam_counts = first[first["family"] != "Unknown"]["family"].value_counts()
    kept = [f for f in fam_counts.index if fam_counts[f] >= MIN_N_PER_FAMILY]

    pct: dict[str, list[float]] = {fam: [] for fam in kept}
    all_viral_pct: list[float] = []
    for mk, _ in models:
        df = c.per_seq_ppl(mk).copy()
        df["accession"] = df["accession"].apply(_strip_sp)
        df = df[df["label"] == "viral"].merge(meta, on="accession", how="left")
        df["family"] = df["family"].fillna("Unknown")
        all_viral_pct.append(
            float(100.0 * (df["mean_perplexity"] < PPL_THRESHOLD).mean())
            if len(df) else np.nan
        )
        for fam in kept:
            sub = df[df["family"] == fam]["mean_perplexity"]
            pct[fam].append(
                float(100.0 * (sub < PPL_THRESHOLD).mean()) if len(sub) else np.nan
            )
    return (
        {fam: np.asarray(v, dtype=float) for fam, v in pct.items()},
        fam_counts,
        np.asarray(all_viral_pct, dtype=float),
    )


# --- Figure -----------------------------------------------------------------

def main() -> None:
    c.apply_style(font_size=22, axes_linewidth=2.4)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "svg.fonttype": "path",
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

    meta = _load_family_meta()
    fam_pct, fam_counts, all_viral_pct = compute_family_pct(ALL_MODELS, meta)
    kept = list(fam_pct.keys())
    params_arr = np.asarray([p for _, p in ALL_MODELS], dtype=float)

    # Rank by final % with PPL<5 at the largest ESMC model.
    fams_top = sorted(kept, key=lambda f: fam_pct[f][-1], reverse=True)[:TOP_N_FAMILIES]
    solid = fams_top[:N_SOLID]
    dotted = fams_top[N_SOLID:]

    # Compact single-panel canvas; legend goes inside the plot's empty
    # upper-left quadrant (trajectories climb low-left → high-right).
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.94, bottom=0.16)

    ax.set_xscale("log")

    # Pastel fills can wash out on white; outline markers in dark grey and
    # thicken the lines a touch so each trajectory stays readable.
    for fam in dotted:
        col = FAMILY_COLOR.get(fam, "#AAAAAA")
        ax.plot(params_arr, fam_pct[fam], color=col,
                lw=2.6, alpha=1.0, ls=":",
                marker="o", ms=9,
                markeredgecolor="#333333", markeredgewidth=1.0,
                zorder=4)
    for fam in solid:
        col = FAMILY_COLOR.get(fam, "#AAAAAA")
        ax.plot(params_arr, fam_pct[fam], color=col,
                lw=4.0, alpha=1.0, ls="-",
                marker="o", ms=11,
                markeredgecolor="#333333", markeredgewidth=1.2,
                zorder=6)

    # All-viral population reference — black dashed (the Accent palette
    # already includes a grey, so we shift the reference to black).
    ref_color = "#000000"
    ax.plot(params_arr, all_viral_pct, color=ref_color,
            lw=2.4, alpha=0.95, ls="--",
            marker="s", ms=7, zorder=5)

    # Axes
    y_max = max(80.0, 1.1 * float(np.nanmax([v.max() for v in fam_pct.values()])))
    ax.set_xlim(params_arr.min() * 0.5, params_arr.max() * 2.2)
    ax.set_ylim(-2, y_max)
    ax.set_xlabel("ESMC parameters", labelpad=6)
    ax.set_ylabel("% of family with PPL $<$ 5", labelpad=6)
    # Parameter-count ticks (same style as fig4: "300M / 600M / 6B").
    ax.set_xticks(params_arr)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(
        lambda v, _pos: f"{v/1e9:g}B" if v >= 1e9 else f"{int(v/1e6)}M"
    ))
    ax.minorticks_off()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Legend, compact, in-plot (upper-left, where the plot is empty).
    handles: list[mlines.Line2D] = []
    labels: list[str] = []
    for fam in solid:
        handles.append(mlines.Line2D(
            [], [], color=FAMILY_COLOR.get(fam, "#AAAAAA"),
            marker="o", ms=10, lw=3.2, ls="-",
            markeredgecolor="#333333", markeredgewidth=1.2,
        ))
        labels.append(fam)
    for fam in dotted:
        handles.append(mlines.Line2D(
            [], [], color=FAMILY_COLOR.get(fam, "#AAAAAA"),
            marker="o", ms=8, lw=2.0, ls=":",
            markeredgecolor="#333333", markeredgewidth=1.0,
        ))
        labels.append(fam)
    handles.append(mlines.Line2D(
        [], [], color=ref_color, marker="s", ms=8, lw=2.4, ls="--",
    ))
    labels.append("All viral")

    ax.legend(
        handles, labels, loc="upper left", frameon=False,
        fontsize=15, handlelength=2.0, handletextpad=0.6,
        borderaxespad=0.4, labelspacing=0.45,
    )

    c.finalize(fig, "fig_family_nativization_esmc")
    print("Saved figures/fig_family_nativization_esmc.{pdf,png}")


if __name__ == "__main__":
    main()
