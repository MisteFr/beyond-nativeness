"""Appendix companion to `fig_family_nativization_esmc.py`.

Main-paper Fig 2 (`fig_family_nativization_esmc`) shows per-family nativization
across the ESMC scale ladder only. This appendix figure reproduces the same
analysis on the two other ESM families — ESM2 (6 scales) and ESM3 (4 scales) —
so the "scale selectively nativizes specific viral families" claim is supported
across the full ESM family.

Native-like threshold is fixed at PPL < 5 across both panels, matching the
main-text protocol exactly (no per-architecture recalibration).

Sources:
  esm_zeroshot_ppl/results/{model}/per_sequence_results.tsv — per-sequence PPL
  esm_viral_probe/datasets/human_virus/data/controls/leave_family_out/family_metadata.tsv
      — accession → family annotation (viral only)
"""
from __future__ import annotations

import re

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as c


# --- Constants --------------------------------------------------------------

PPL_THRESHOLD    = 5.0           # fixed across both panels (same as main Fig 2)

TOP_N_FAMILIES   = 8             # families plotted (shared across panels)
N_SOLID          = 3             # top-N at largest model (solid lines)
MIN_N_PER_FAMILY = 50

FAMILY_META_TSV = (c.LAB_ROOT /
                   "esm_viral_probe/datasets/human_virus/data/controls/"
                   "leave_family_out/family_metadata.tsv")

# Reuse the main-figure ESMC palette so one colour = one family across the
# whole paper. ColorBrewer 8-class Accent — assigned in legend order
# (top-down by ESMC-6B nativization rank, matching the main figure).
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
    threshold: float,
) -> tuple[dict[str, np.ndarray], pd.Series, np.ndarray]:
    """For each (model, family) return % of that family below a fixed
    threshold. No cross-model merge on sequences."""
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
            float(100.0 * (df["mean_perplexity"] < threshold).mean())
            if len(df) else np.nan
        )
        for fam in kept:
            sub = df[df["family"] == fam]["mean_perplexity"]
            pct[fam].append(
                float(100.0 * (sub < threshold).mean()) if len(sub) else np.nan
            )
    return (
        {fam: np.asarray(v, dtype=float) for fam, v in pct.items()},
        fam_counts,
        np.asarray(all_viral_pct, dtype=float),
    )


# --- Figure -----------------------------------------------------------------

def _family_models(fam: str) -> list[tuple[str, int]]:
    return [(mk, params) for mk, _, params in c.MODEL_FAMILIES[fam]]


def _draw_panel(ax, arch_name: str, models, meta, shared_solid, shared_dotted,
                threshold: float):
    fam_pct, _fam_counts, all_viral_pct = compute_family_pct(models, meta, threshold)
    params_arr = np.asarray([p for _, p in models], dtype=float)

    ax.set_xscale("log")

    for fam in shared_dotted:
        if fam not in fam_pct:
            continue
        col = FAMILY_COLOR.get(fam, "#AAAAAA")
        ax.plot(params_arr, fam_pct[fam], color=col,
                lw=2.4, alpha=1.0, ls=":",
                marker="o", ms=8,
                markeredgecolor="#333333", markeredgewidth=1.0,
                zorder=4)
    for fam in shared_solid:
        if fam not in fam_pct:
            continue
        col = FAMILY_COLOR.get(fam, "#AAAAAA")
        ax.plot(params_arr, fam_pct[fam], color=col,
                lw=3.6, alpha=1.0, ls="-",
                marker="o", ms=10,
                markeredgecolor="#333333", markeredgewidth=1.2,
                zorder=6)

    # Accent's 8th colour is grey, so shift the all-viral reference to black.
    ref_color = "#000000"
    ax.plot(params_arr, all_viral_pct, color=ref_color,
            lw=2.4, alpha=0.95, ls="--",
            marker="s", ms=7, zorder=5)

    ax.set_xticks(params_arr)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(
        lambda v, _pos: f"{v/1e9:g}B" if v >= 1e9 else f"{int(v/1e6)}M"
    ))
    ax.minorticks_off()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.set_xlim(params_arr.min() * 0.5, params_arr.max() * 2.2)
    ax.set_xlabel(f"{arch_name} parameters", labelpad=6)

    return fam_pct


def main() -> None:
    c.apply_style(font_size=18, axes_linewidth=2.2)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "svg.fonttype": "path",
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })
    meta = _load_family_meta()

    # Select the 8 families using the ESMC-6B endpoint (same ranking as main
    # Fig 2), so colour = same family across all paper panels.
    esmc_models = [(mk, params) for mk, _, params in c.MODEL_FAMILIES["ESMC"]]
    fam_pct_esmc, _, _ = compute_family_pct(esmc_models, meta, PPL_THRESHOLD)
    fams_top = sorted(fam_pct_esmc.keys(),
                      key=lambda f: fam_pct_esmc[f][-1],
                      reverse=True)[:TOP_N_FAMILIES]
    shared_solid = fams_top[:N_SOLID]
    shared_dotted = fams_top[N_SOLID:]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4),
                             gridspec_kw=dict(width_ratios=[1.3, 1.1]))
    fig.subplots_adjust(left=0.075, right=0.82, top=0.93, bottom=0.19, wspace=0.22)

    panel_fam_pct = {}
    for ax, arch in zip(axes, ("ESM2", "ESM3")):
        panel_fam_pct[arch] = _draw_panel(
            ax, arch, _family_models(arch), meta,
            shared_solid, shared_dotted,
            PPL_THRESHOLD,
        )

    # Shared y-axis semantics, independent y-limits so small-effect panels
    # (ESM3) aren't flattened by any large jumps.
    axes[0].set_ylabel("% of family with PPL $<$ 5", labelpad=6)
    for ax in axes:
        lo, hi = ax.get_ylim()
        ax.set_ylim(-2, max(80.0, 1.1 * hi))

    # Panel letters
    for ax, letter in zip(axes, ("A", "B")):
        ax.text(-0.13, 1.03, letter, transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="bottom", ha="right")

    # Legend — stack all families + "All viral" reference on the far right.
    handles: list[mlines.Line2D] = []
    labels: list[str] = []
    for fam in shared_solid:
        handles.append(mlines.Line2D(
            [], [], color=FAMILY_COLOR.get(fam, "#AAA"),
            marker="o", ms=10, lw=3.0, ls="-",
            markeredgecolor="#333333", markeredgewidth=1.2,
        ))
        labels.append(fam)
    for fam in shared_dotted:
        handles.append(mlines.Line2D(
            [], [], color=FAMILY_COLOR.get(fam, "#AAA"),
            marker="o", ms=8, lw=2.0, ls=":",
            markeredgecolor="#333333", markeredgewidth=1.0,
        ))
        labels.append(fam)
    handles.append(mlines.Line2D([], [], color="#000000",
                                 marker="s", ms=8, lw=2.4, ls="--"))
    labels.append("All viral")
    fig.legend(handles, labels,
               loc="center right", bbox_to_anchor=(0.998, 0.55),
               frameon=False, fontsize=13,
               handlelength=2.0, handletextpad=0.6,
               labelspacing=0.45, borderaxespad=0.0)

    # Persist the numbers for caption / supplementary table.
    rows = []
    for arch, fam_pct in panel_fam_pct.items():
        for i, (mk, _, params) in enumerate(c.MODEL_FAMILIES[arch]):
            for fam in shared_solid + shared_dotted:
                if fam in fam_pct:
                    rows.append({
                        "architecture": arch,
                        "model": mk,
                        "params": params,
                        "family": fam,
                        "pct_native_like": fam_pct[fam][i],
                    })
    out_tsv = c.HERE / "_appfig_family_nativization.tsv"
    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved numbers: {out_tsv}")

    c.finalize(fig, "appfig_family_nativization_all")
    print("Saved figures/appfig_family_nativization_all.{pdf,png}")


if __name__ == "__main__":
    main()
