"""Appendix figure (Ctrl 3) — the viral probe separates viral proteins from the
human host itself, not just from distant kingdoms.

Reviewer concern: the main probe uses a multi-kingdom Swiss-Prot negative set,
so near-ceiling AUC could reflect a broad "animal vs bacterium/plant/fungus"
contrast rather than a viral-specific signal. We re-evaluate each trained probe
with the negative class restricted to *Homo sapiens* proteins (Swiss-Prot,
OX=9606) — the same host the viral positives infect. Discrimination stays high
for every model >= 35M params, so the probe is not merely a kingdom classifier.

Caveat shown on the figure: held-out human negatives number only N=38 (the
multi-kingdom test pool is ~3.7% human), so the human-only estimate is noisier
than the full-pool estimate, especially for the smallest model.

Style: deliberate sibling of Fig 2 (sans-serif, family brackets at top, dotted
family separators, black-edged bars, frameless legend, no y-grid). Grouped bars
per model: neutral grey = full multi-kingdom negatives (reference, N=1,034),
nonviral blue = human-host-only negatives (N=38, OX=9606).

Data source
-----------
esm_viral_probe/datasets/human_virus/results/human_neg_summary.json
(Ctrl 3, job 65725451; backfilled to ESMC-600M). Predates the ESM2-3B/15B,
ESMC-6B and ESM3-large probes, so 9 pLMs are shown.

Outputs:
  ../figures/appfig_human_negatives.{pdf,png}
  _appfig_human_negatives.json  (numbers for the LaTeX text)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

import _common as c

# Both bars are negative sets, so encode them on a single greyscale ramp (light
# = full multi-kingdom reference, charcoal = the stringent human-only test).
# Keeping the bars colourless leaves the family brackets (blue/green/red) as the
# only hue in the panel, so the bars never collide with the ESM2 family blue.
COL_FULL  = c.PALETTE["neutral"]   # #CFCECE
COL_HUMAN = "#4D4D4D"


def load_rows():
    """Ordered (model_key, family, full_auc, human_auc, n_human) for the 9
    models present in the Ctrl 3 summary, in canonical family/size order."""
    summ = c.human_neg_summary()
    rows = []
    for fam in ("ESM2", "ESMC", "ESM3"):
        for mk, _, _ in c.MODEL_FAMILIES[fam]:
            if mk in summ:
                d = summ[mk]
                rows.append((mk, fam, float(d["full_auc"]),
                             float(d["human_auc"]), int(d["n_human"])))
    return rows


def panel(ax, rows):
    from matplotlib.patches import Patch

    n = len(rows)
    x = np.arange(n)
    w = 0.38
    full  = [r[2] for r in rows]
    human = [r[3] for r in rows]

    ax.bar(x - w / 2, full,  w, color=COL_FULL,  edgecolor="black", lw=0.7, zorder=3)
    b2 = ax.bar(x + w / 2, human, w, color=COL_HUMAN, edgecolor="black", lw=0.7, zorder=3)

    # Annotate human-only AUC (the number that matters for this control) and a
    # smaller grey full-pool reference, mirroring Fig 2's value labels.
    for i, (rect, h, f) in enumerate(zip(b2, human, full)):
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=COL_HUMAN)
        ax.annotate(f"{f:.3f}", xy=(x[i] - w / 2, f),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="0.45")

    # Dotted family separators (Fig 2 idiom).
    fams = [r[1] for r in rows]
    for i in range(1, n):
        if fams[i] != fams[i - 1]:
            ax.axvline(i - 0.5, color="0.7", lw=0.8, ls=":", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([c.pretty(r[0]) for r in rows], rotation=45,
                       ha="right", fontsize=10)
    ax.set_ylabel("AUC-ROC (viral vs. negatives)")
    ax.set_ylim(0.82, 1.07)
    ax.set_yticks([0.85, 0.90, 0.95, 1.00])

    # Family brackets at the top, identical treatment to Fig 2.
    bracket_y = 1.028
    start = 0
    for i in range(1, n + 1):
        if i == n or fams[i] != fams[start]:
            lo, hi, name = start, i - 1, fams[start]
            col = c.FAMILY_COLOR.get(name, "0.4")
            ax.plot([lo - 0.34, hi + 0.34], [bracket_y] * 2, color=col, lw=2.4,
                    solid_capstyle="butt", clip_on=False, zorder=4)
            ax.text((lo + hi) / 2, bracket_y + 0.006, name, ha="center",
                    va="bottom", fontsize=11, color=col, fontweight="bold",
                    clip_on=False)
            start = i

    handles = [Patch(facecolor=COL_FULL,  edgecolor="black", lw=0.7,
                     label="All non-viral (multi-kingdom, N=1,034)"),
               Patch(facecolor=COL_HUMAN, edgecolor="black", lw=0.7,
                     label="Human host proteins only (N=38)")]
    # Bars fill the full plot height, so seat the legend below the axis.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, fontsize=10.5, columnspacing=2.0, handletextpad=0.6)


def main():
    c.apply_style(font_size=13, axes_linewidth=1.6)
    rows = load_rows()

    fig, ax = plt.subplots(1, 1, figsize=(11.0, 5.4))
    fig.subplots_adjust(top=0.9, bottom=0.34, left=0.085, right=0.985)
    panel(ax, rows)
    c.finalize(fig, "appfig_human_negatives")

    out = {
        "n_viral_positives": 1046,
        "n_human_negatives": int(rows[0][4]) if rows else None,
        "models": {r[0]: {"full_auc": round(r[2], 4),
                          "human_auc": round(r[3], 4),
                          "delta": round(r[3] - r[2], 4)} for r in rows},
        "min_human_auc": round(min(r[3] for r in rows), 4),
        "min_human_auc_model": min(rows, key=lambda r: r[3])[0],
    }
    with open(c.HERE / "_appfig_human_negatives.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print("Saved figures/appfig_human_negatives.{pdf,png}")


if __name__ == "__main__":
    main()
