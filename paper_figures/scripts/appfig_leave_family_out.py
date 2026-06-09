"""Appendix figure (Ctrl 6) — the viral signal generalizes to entirely unseen
viral families.

Reviewer concern: a probe could memorize family-specific sequence motifs rather
than a general "viral" signal. We test this with leave-one-family-out
evaluation: for each viral family the probe is retrained on all *other*
families and tested only on the held-out family (no sequences from that family
seen in training). Held-out AUC stays high almost everywhere; the only family
that drops is Retroviridae (endogenous-retrovirus entanglement with host
genomes), so the probe captures a family-general viral embedding geometry.

Style: sibling of Fig 5's horizontal bars + the house strip idiom — sans-serif,
model-family colors (ESM2 blue, ESMC green, ESM3 red) shared with Fig 2, no
dense heatmap, no in-plot gridlines.

Single panel — per-model box + jittered strip of the 13 held-out-family AUCs,
one column per pLM (ESM2/ESMC/ESM3 brackets + dotted family separators, the
Fig 2 idiom), boxes coloured by model family, black diamond = the same model's
full within-distribution probe AUC (Fig 2). The point: holding out an entire
viral family barely moves AUC below the full probe. Retroviridae is the hard family — it sits
in the ESMC low whiskers (named in the caption / shown per-family in the
alternative view). The per-family strip view (panel_families, retained below) is
a one-line swap in main() if a by-family x-axis is preferred.

Data source
-----------
esm_viral_probe/datasets/human_virus/results/leave_family_out_summary.json
(Ctrl 6, job 65883058); full-probe AUC from results/{model}/test_results.json.
9 pLMs x 13 families = 117 held-out evaluations.

Outputs:
  ../figures/appfig_leave_family_out.{pdf,png}
  _appfig_leave_family_out.json  (numbers for the LaTeX text)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import _common as c

REF = 0.92  # "almost always >= 0.92" reference line shared by both panels.


def family_of(model_key: str) -> str:
    return c.pretty(model_key).split("-")[0]


def load_grid():
    summ = c.leave_family_out_summary()
    models = [mk for fam in ("ESM2", "ESMC", "ESM3")
              for mk, _, _ in c.MODEL_FAMILIES[fam] if mk in summ["models"]]
    fams = list(summ["families"])
    M = np.full((len(fams), len(models)), np.nan)  # [family, model]
    for j, mk in enumerate(models):
        md = summ["models"][mk]
        for i, fam in enumerate(fams):
            if fam in md:
                M[i, j] = md[fam]["auc_roc"]
    return models, fams, M


def family_brackets(ax, models, y, fs=11):
    """Fig-2-style ESM2/ESMC/ESM3 brackets above per-model columns."""
    fams = [family_of(m) for m in models]
    start = 0
    for i in range(1, len(models) + 1):
        if i == len(models) or fams[i] != fams[start]:
            lo, hi, name = start, i - 1, fams[start]
            col = c.FAMILY_COLOR.get(name, "0.4")
            ax.plot([lo - 0.34, hi + 0.34], [y] * 2, color=col, lw=2.4,
                    solid_capstyle="butt", clip_on=False, zorder=6)
            ax.text((lo + hi) / 2, y + 0.004, name, ha="center", va="bottom",
                    fontsize=fs, color=col, fontweight="bold", clip_on=False)
            start = i


def panel_models(ax, models, M, rng):
    """A — box + strip of held-out AUC across families, one column per model."""
    x = np.arange(len(models))
    for j, mk in enumerate(models):
        col = c.FAMILY_COLOR[family_of(mk)]
        vals = M[:, j][~np.isnan(M[:, j])]

        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(x[j] + jitter, vals, s=20, color=col, alpha=0.55,
                   edgecolors="none", zorder=3)
        ax.boxplot([vals], positions=[x[j]], widths=0.5, vert=True,
                   whis=(0, 100), showfliers=False, patch_artist=True,
                   medianprops=dict(color="white", lw=1.6, zorder=5),
                   boxprops=dict(facecolor=col, edgecolor="black", lw=0.9,
                                 alpha=0.55, zorder=4),
                   whiskerprops=dict(color="black", lw=0.9, zorder=3),
                   capprops=dict(color="black", lw=0.9, zorder=3))

    # Dotted vertical family separators (Fig 2 idiom).
    for edge in (3.5, 5.5):
        ax.axvline(edge, color="0.7", lw=0.8, ls=":", zorder=1)

    full = [c.probe_auc(m) for m in models]
    ax.scatter(x, full, marker="D", s=30, color="black", zorder=6,
               label="Full probe (all families)")

    ax.set_xticks(x)
    ax.set_xticklabels([c.pretty(m) for m in models], rotation=45,
                       ha="right", fontsize=9.5)
    ax.set_ylabel("Held-out AUC-ROC")
    ax.set_ylim(0.84, 1.028)
    ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    family_brackets(ax, models, y=1.008)
    ax.legend(loc="lower left", fontsize=9.5, borderaxespad=0.5)


def panel_families(ax, models, fams, M, rng):
    """B — per-family strip of the 9 per-model AUCs, sorted by mean."""
    order = np.argsort(np.nanmean(M, axis=1))  # ascending: worst at bottom
    fams_s = [fams[i] for i in order]
    M_s = M[order]
    y = np.arange(len(fams_s))

    for i in range(len(fams_s)):
        row = M_s[i]
        for j, mk in enumerate(models):
            if np.isnan(row[j]):
                continue
            col = c.FAMILY_COLOR[family_of(mk)]
            ax.scatter(row[j], y[i] + rng.uniform(-0.12, 0.12), s=22,
                       color=col, alpha=0.75, edgecolors="none", zorder=3)
        m = float(np.nanmean(row))
        ax.plot([m, m], [y[i] - 0.3, y[i] + 0.3], color="black", lw=1.8,
                zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(fams_s, fontsize=9.0)
    ax.set_ylim(-0.6, len(fams_s) - 0.4)
    ax.set_xlabel("Held-out AUC-ROC")
    ax.set_xlim(0.84, 1.005)
    ax.set_xticks([0.85, 0.90, 0.95, 1.00])
    ax.tick_params(axis="y", length=0)

    # Dots are coloured by model family (ESM2/ESMC/ESM3) + a black family-mean
    # tick. Upper-left is empty (top families have no low outliers), so seat the
    # key there.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", ms=7, alpha=0.75,
                      color=c.FAMILY_COLOR[f], label=f)
               for f in ("ESM2", "ESMC", "ESM3")]
    handles.append(Line2D([0], [0], color="black", lw=1.8, label="Family mean"))
    ax.legend(handles=handles, loc="upper left", fontsize=9.5,
              borderaxespad=0.6, handletextpad=0.5, labelspacing=0.4)


def main():
    c.apply_style(font_size=13, axes_linewidth=1.6)
    rng = np.random.default_rng(7)
    models, fams, M = load_grid()

    # Single per-model panel (Fig-2-style model x-axis). To use the by-family
    # view instead, swap for `panel_families(ax, models, fams, M, rng)`.
    fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.2))
    fig.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.975)
    panel_models(ax, models, M, rng)

    c.finalize(fig, "appfig_leave_family_out")

    flat = M[~np.isnan(M)]
    wi, wj = np.unravel_index(np.nanargmin(M), M.shape)
    out = {
        "n_models": len(models), "n_families": len(fams),
        "n_evaluations": int(flat.size),
        "frac_auc_ge_0.92": round(float((flat >= REF).mean()), 4),
        "median_auc": round(float(np.median(flat)), 4),
        "min_auc": round(float(np.nanmin(M)), 4),
        "min_auc_family": fams[wi], "min_auc_model": models[wj],
        "per_model_mean": {m: round(float(np.nanmean(M[:, j])), 4)
                           for j, m in enumerate(models)},
        "per_family_mean": {fams[i]: round(float(np.nanmean(M[i])), 4)
                            for i in range(len(fams))},
    }
    with open(c.HERE / "_appfig_leave_family_out.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print("Saved figures/appfig_leave_family_out.{pdf,png}")


if __name__ == "__main__":
    main()
