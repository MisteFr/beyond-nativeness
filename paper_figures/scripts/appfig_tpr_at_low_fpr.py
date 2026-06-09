"""Appendix Fig - TPR at low FPR for probe and PPL zero-shot.

Probe AUC saturates near 1.0, masking dramatic differences at strict
operating points: at FPR=10^-3 the probe TPR ranges from 0.06 (ESM2-8M)
to 0.83 (ESMC-6B), and PPL zero-shot peaks at 0.27. This figure surfaces
that hidden dynamic range as TPR vs model scale, faceted by family, with
bootstrap 95% CIs.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import _common as c


FPR_TARGETS = (1e-3, 1e-2, 5e-2)
FAMILIES = ("ESM2", "ESMC", "ESM3")


def _load_long() -> pd.DataFrame:
    p = c.HERE / "_tpr_at_low_fpr.tsv"
    if not p.exists():
        raise FileNotFoundError(
            f"Run scripts/_tpr_at_low_fpr.py first to generate {p.name}"
        )
    return pd.read_csv(p, sep="\t")


def _merge_ties(fam: str, df_sub: pd.DataFrame, modality: str, fpr: float):
    """Average across models that share an x-position (ESM3 small/open at 1.4B)."""
    rows = (df_sub.query("modality == @modality and fpr == @fpr")
                  .sort_values("params"))
    if rows.empty:
        return np.array([]), [], np.array([]), np.array([]), np.array([])
    xs, labels, mids, los, his = [], [], [], [], []
    for params, grp in rows.groupby("params", sort=True):
        xs.append(float(params))
        if fam == "ESM3":
            label = "/".join(grp["label"].tolist())
            params_label = _param_label(int(params))
            labels.append(f"{params_label}\n{label}")
        else:
            labels.append(grp["label"].iloc[0])
        mids.append(grp["tpr"].mean())
        los.append(grp["tpr_lo"].mean())
        his.append(grp["tpr_hi"].mean())
    return (np.array(xs), labels, np.array(mids),
            np.array(los), np.array(his))


def _param_label(params: int) -> str:
    if params >= 1e9:
        return f"{params / 1e9:g}B"
    return f"{params / 1e6:g}M"


def panel(ax, df: pd.DataFrame, fam: str, fpr: float, show_legend: bool):
    df_fam = df[df["family"] == fam]
    handles = []

    for modality, color, marker, label_m in (
        ("probe", c.PALETTE["blue_main"], "o", "Embedding probe"),
        ("ppl_zeroshot", c.PALETTE["red_strong"], "s", "PPL zero-shot"),
    ):
        xs, ticks, mids, los, his = _merge_ties(fam, df_fam, modality, fpr)
        if len(xs) == 0:
            continue
        yerr = np.vstack([mids - los, his - mids])
        ax.fill_between(xs, los, his, color=color, alpha=0.12, lw=0, zorder=1)
        h, = ax.plot(xs, mids, marker=marker, ms=10, lw=2.8, color=color,
                     markeredgecolor="white", markeredgewidth=1.0, zorder=3,
                     label=label_m)
        ax.errorbar(xs, mids, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.4,
                    capsize=3, capthick=1.2, alpha=0.9, zorder=2)
        handles.append(h)

    # x-axis ticks from probe (always present)
    xs_p, ticks_p, *_ = _merge_ties(fam, df_fam, "probe", fpr)
    ax.set_xscale("log")
    ax.set_xticks(xs_p)
    ax.set_xticklabels(ticks_p)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(xs_p.min() / 2.0, xs_p.max() * 2.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.grid(axis="y", alpha=0.25, linestyle="--", lw=0.9, zorder=0)
    ax.axhline(1.0, color="0.85", lw=0.8, zorder=0)
    return handles


def main():
    c.apply_style(font_size=14, axes_linewidth=2.0)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "svg.fonttype": "path",
    })
    df = _load_long()

    fig = plt.figure(figsize=(15.0, 9.5))
    gs = fig.add_gridspec(len(FPR_TARGETS), len(FAMILIES),
                          left=0.075, right=0.985,
                          top=0.90, bottom=0.08,
                          hspace=0.45, wspace=0.16)

    handles_for_legend = None
    for r, fpr in enumerate(FPR_TARGETS):
        for cidx, fam in enumerate(FAMILIES):
            ax = fig.add_subplot(gs[r, cidx])
            handles = panel(ax, df, fam, fpr, show_legend=(r == 0 and cidx == 0))
            if handles_for_legend is None and len(handles) == 2:
                handles_for_legend = handles
            if r == 0:
                ax.set_title(fam, loc="left", fontweight="bold", pad=8)
            if cidx == 0:
                pct = fpr * 100
                fpr_str = (f"FPR = {pct:g}%" if pct >= 1
                           else f"FPR = {pct:.1f}%")
                ax.set_ylabel(f"TPR @ {fpr_str}")
            else:
                ax.set_yticklabels([])
            if r == len(FPR_TARGETS) - 1:
                ax.set_xlabel("Model parameters")

    if handles_for_legend is not None:
        fig.legend(handles=handles_for_legend, loc="upper center",
                   bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=False,
                   fontsize=14, handlelength=2.6, columnspacing=3.0)

    c.finalize(fig, "appfig_tpr_at_low_fpr")
    print("Appendix TPR @ low FPR figure rendered.")


if __name__ == "__main__":
    main()
