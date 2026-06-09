"""Fig 4 - Zero-shot reconstruction and embedding probes scale differently.

Within each model family, embedding-probe AUC remains near ceiling while
PPL-based zero-shot separation changes substantially with model scale.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import _common as c


def _zeroshot_auc(model_key: str) -> float | None:
    path = c.ZEROSHOT / model_key / "summary.json"
    if not path.exists():
        alt = c.ZEROSHOT / f"{model_key}_api" / "summary.json"
        if alt.exists():
            path = alt
        else:
            return None
    v = json.loads(path.read_text())["auroc_perplexity"]
    return max(v, 1.0 - v)


def _param_label(params: int) -> str:
    if params >= 1e9:
        value = params / 1e9
        return f"{value:g}B"
    return f"{params / 1e6:g}M"


def _tick_label(fam: str, label: str, params: int) -> str:
    if fam == "ESM3":
        return f"{_param_label(params)}\n{label}"
    return label


def _merge_ties(fam, models, probe, ppl):
    """Collapse models with identical param counts into a single averaged point."""
    if fam != "ESM3":
        xs = np.array([float(p) for _, _, p in models])
        labels = [_tick_label(fam, label, p) for _, label, p in models]
        return xs, labels, probe, ppl
    by_param: dict[int, list[int]] = {}
    for idx, (_, _, p) in enumerate(models):
        by_param.setdefault(p, []).append(idx)
    xs, labels, p_out, z_out = [], [], [], []
    for params in sorted(by_param):
        idxs = by_param[params]
        xs.append(float(params))
        merged_label = "/".join(models[i][1] for i in idxs)
        labels.append(_tick_label(fam, merged_label, params))
        p_out.append(np.nanmean([probe[i] for i in idxs]))
        z_out.append(np.nanmean([ppl[i] for i in idxs]))
    return np.array(xs), labels, np.array(p_out), np.array(z_out, dtype=float)


def panel_scaling(fig, gs_block):
    inner = gs_block.subgridspec(1, 3, wspace=0.16)
    handles_for_legend = None
    baseline_vals = c.baseline_aucs()
    shallow_top = max(baseline_vals["aa_composition"],
                      baseline_vals["dipeptide_composition"])
    y_low = 0.50
    for i, fam in enumerate(("ESM2", "ESMC", "ESM3")):
        ax = fig.add_subplot(inner[0, i])
        models = c.MODEL_FAMILIES[fam]
        probe_raw = np.array([c.probe_auc(mk) for mk, _, _ in models])
        ppl_raw = np.array([_zeroshot_auc(mk) for mk, _, _ in models], dtype=float)
        xs, labels, probe, ppl = _merge_ties(fam, models, probe_raw, ppl_raw)

        xlo = xs.min() / 2.0
        xhi = xs.max() * 2.0
        ax.fill_between([xlo, xhi], y_low, shallow_top,
                        color="0.78", alpha=0.28, lw=0, zorder=0)
        h_shallow_line = ax.axhline(shallow_top, color="0.40", lw=1.8,
                                    linestyle="-", zorder=1,
                                    label="Baseline")
        ax.fill_between([xlo, xhi], 0.97, 1.0,
                        color=c.PALETTE["blue_main"], alpha=0.08, lw=0,
                        zorder=0)

        h_probe, = ax.plot(xs, probe, marker="o", ms=11, lw=3.2,
                           color=c.PALETTE["blue_main"], zorder=3,
                           markeredgecolor="white", markeredgewidth=1.2,
                           label="Embedding probe")
        h_ppl, = ax.plot(xs, ppl, marker="s", ms=10, lw=3.2,
                         color=c.PALETTE["red_strong"], zorder=3,
                         markeredgecolor="white", markeredgewidth=1.2,
                         label="PPL zero-shot")
        if handles_for_legend is None:
            handles_for_legend = [h_probe, h_ppl, h_shallow_line]

        ax.set_title(fam, loc="left", fontweight="bold", pad=10)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", which="minor", length=0)
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel("Model parameters")
        ax.set_ylim(y_low, 1.01)
        ax.set_yticks([0.55, 0.70, 0.85, 1.00])
        ax.grid(axis="y", alpha=0.25, linestyle="--", lw=0.9, zorder=0)
        if i == 0:
            ax.set_ylabel("Viral / non-viral AUC")
        else:
            ax.set_yticklabels([])
    return handles_for_legend


def main():
    c.apply_style(font_size=17, axes_linewidth=2.2)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "svg.fonttype": "path",
    })
    fig = plt.figure(figsize=(15.0, 5.8))
    gs = fig.add_gridspec(1, 1, left=0.075, right=0.985,
                          top=0.76, bottom=0.22)
    handles = panel_scaling(fig, gs[0])
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False,
               fontsize=16, handlelength=2.6, columnspacing=3.0)
    c.finalize(fig, "fig4_scaling_divergence")
    print("Fig 4 rendered.")


if __name__ == "__main__":
    main()
