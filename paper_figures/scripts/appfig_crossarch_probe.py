"""Cross-architecture viral probe vs. PPL ACROSS SCALE (paper §4.3 / Fig. 3 style).

Faithful 2-panel sibling of `fig4_scaling_divergence.py`: same serif house style,
same "Viral / non-viral AUC" axis (0.55-1.00), same blue-circle "Embedding probe" vs
red-square "PPL zero-shot" lines over a grey shallow-feature (baseline) region and a
near-ceiling band, with model scale on a log x-axis. Reproduces the §4.3 result off the
ESM family, across the FULL scale ladder of both non-ESM architectures tested in
`cross_architecture_nativeness`:

  Panel A  ProGen2 (autoregressive)      151M / 764M / 2.7B / 6.4B
  Panel B  EvoDiff OA-DM (diffusion)     38M / 640M

ProtT5-XL is intentionally NOT shown. EvoDiff PPL zero-shot uses the training-faithful
OA-ARDM ELBO (the `*_elbo` keys); the embedding probe is trained per scale on that
scale's mean-pooled embeddings (same human viral/cellular split as the ESM probes).

Data:
  Embedding probe AUC : results/<probe_key>/probe/test_results.json   (linear.auc_roc)
  PPL zero-shot AUC   : results/<ppl_key>/per_sequence_results.tsv     (test split, max(AUC,1-AUC) of -PPL)
  Baseline region     : esm_viral_probe .../baseline/summary.json      (best of length/AA/dipeptide)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import _common as c

CROSS = c.LAB_ROOT / "cross_architecture_nativeness"

# arch -> list of (params, probe_key, ppl_key, ppl_col, label)
LADDERS = {
    "ProGen2": [
        (151e6,  "progen2_small",  "progen2_small",  "mean_perplexity", "151M"),
        (764e6,  "progen2_base",   "progen2_base",   "mean_perplexity", "764M"),
        (2.7e9,  "progen2_large",  "progen2_large",  "mean_perplexity", "2.7B"),
        (6.4e9,  "progen2_xlarge", "progen2_xlarge", "mean_perplexity", "6.4B"),
    ],
    "EvoDiff": [
        (38e6,   "evodiff_oadm_38m",  "evodiff_oadm_38m_elbo",  "mean_perplexity_elbo", "38M"),
        (640e6,  "evodiff_oadm_640m", "evodiff_oadm_640m_elbo", "mean_perplexity_elbo", "640M"),
    ],
}
SUBTITLE = {"ProGen2": "autoregressive", "EvoDiff": "discrete diffusion"}


def probe_auc(key):
    p = CROSS / "results" / key / "probe" / "test_results.json"
    if not p.exists():
        return np.nan
    return float(json.loads(p.read_text())["linear"]["auc_roc"])


def ppl_zeroshot_auc(key, col):
    p = CROSS / "results" / key / "per_sequence_results.tsv"
    if not p.exists():
        return np.nan
    df = pd.read_csv(p, sep="\t")
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    y = (df["label"] == "viral").astype(int).values
    a = roc_auc_score(y, -df[col].values)
    return max(a, 1.0 - a)


def esm_band():
    aucs = {}
    for mk in c.ALL_MODELS_ORDERED:
        try:
            aucs[mk] = c.probe_auc(mk)
        except (FileNotFoundError, KeyError):
            pass
    return min(aucs.values()), max(aucs.values())


def fmt_params(v, _pos=None):
    return f"{v/1e9:g}B" if v >= 1e9 else f"{int(round(v/1e6))}M"


def panel(ax, arch, shallow_top, esm_lo, esm_hi, first):
    rows = LADDERS[arch]
    xs = np.array([r[0] for r in rows], float)
    probe = np.array([probe_auc(r[1]) for r in rows])
    ppl = np.array([ppl_zeroshot_auc(r[2], r[3]) for r in rows])
    print(f"  {arch}: probe={np.round(probe,4)}  ppl={np.round(ppl,4)}")

    xlo, xhi = xs.min() / 1.8, xs.max() * 1.8
    y_low = 0.50
    ax.fill_between([xlo, xhi], y_low, shallow_top, color="0.78", alpha=0.28, lw=0, zorder=0)
    h_base = ax.axhline(shallow_top, color="0.40", lw=1.8, ls="-", zorder=1, label="Baseline")
    ax.fill_between([xlo, xhi], esm_lo, esm_hi, color=c.PALETTE["blue_main"], alpha=0.10, lw=0, zorder=0)

    mp = np.isfinite(probe)
    h_probe, = ax.plot(xs[mp], probe[mp], marker="o", ms=11, lw=3.2,
                       color=c.PALETTE["blue_main"], zorder=3,
                       markeredgecolor="white", markeredgewidth=1.2, label="Embedding probe")
    mz = np.isfinite(ppl)
    h_ppl, = ax.plot(xs[mz], ppl[mz], marker="s", ms=10, lw=3.2,
                     color=c.PALETTE["red_strong"], zorder=3,
                     markeredgecolor="white", markeredgewidth=1.2, label="PPL zero-shot")

    ax.set_title(f"{arch}  ({SUBTITLE[arch]})", loc="left", fontweight="bold", pad=10)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[4] for r in rows])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel(f"{arch} parameters")
    ax.set_ylim(y_low, 1.01)
    ax.set_yticks([0.55, 0.70, 0.85, 1.00])
    ax.grid(axis="y", alpha=0.25, ls="--", lw=0.9, zorder=0)
    if first:
        ax.set_ylabel("Viral / non-viral AUC")
    else:
        ax.set_yticklabels([])
    return (h_probe, h_ppl, h_base), dict(
        params=xs.tolist(), probe=probe.tolist(), ppl_zeroshot=ppl.tolist(),
        labels=[r[4] for r in rows])


def main():
    c.apply_style(font_size=17, axes_linewidth=2.2)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "svg.fonttype": "path",
    })
    base = c.baseline_aucs()
    shallow_top = max(base["aa_composition"], base["dipeptide_composition"])
    esm_lo, esm_hi = esm_band()

    fig = plt.figure(figsize=(11.4, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], left=0.085, right=0.985,
                          top=0.80, bottom=0.165, wspace=0.07)
    handles, stats = None, {}
    for i, arch in enumerate(("ProGen2", "EvoDiff")):
        ax = fig.add_subplot(gs[0, i])
        h, s = panel(ax, arch, shallow_top, esm_lo, esm_hi, first=(i == 0))
        handles = handles or h
        stats[arch] = s
        ax.text(-0.02 if i == 0 else -0.02, 1.12, "AB"[i], transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="bottom", ha="right")

    band = mlines.Line2D([], [], color=c.PALETTE["blue_main"], lw=8, alpha=0.4)
    fig.legend(handles=list(handles) + [band],
               labels=["Embedding probe", "PPL zero-shot", "Baseline",
                       f"ESM probe band ({esm_lo:.3f}–{esm_hi:.3f})"],
               loc="upper center", bbox_to_anchor=(0.54, 0.99), ncol=4, frameon=False,
               fontsize=13, handlelength=2.2, columnspacing=1.8)

    c.finalize(fig, "appfig_crossarch_probe")
    print("Saved figures/appfig_crossarch_probe.{pdf,png}")
    stats["shallow_baseline_top"] = round(float(shallow_top), 4)
    stats["esm_band"] = [round(esm_lo, 4), round(esm_hi, 4)]
    with open(c.HERE / "_appfig_crossarch_probe.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
