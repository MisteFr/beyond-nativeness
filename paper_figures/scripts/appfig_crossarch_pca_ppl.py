"""Cross-architecture nativeness axis: ESM vs autoregressive vs diffusion.

Paper-styled sibling of `appfig_pca_ppl_esm2_esm3.py` (App. Fig 6). Reproduces
BOTH panels of main Fig 1 (PCA scatter | per-group PPL strip) for three pLMs that
span three DIFFERENT training objectives, on the SAME 13-group pool (n=66,790)
that the ESM appendix used:

  row 1 (A, B): ESMC-600M  — masked LM      (masked-reconstruction PPL)  [reference]
  row 2 (C, D): ProGen2-base — autoregressive (true causal PPL)
  row 3 (E, F): EvoDiff OA-DM — discrete diffusion (order-agnostic ELBO PPL)

This directly addresses the paper's stated open question (Limitations: "whether the
same nativeness-axis structure emerges in autoregressive, diffusion-based, or other
architectures remains an open question"). ProtT5-XL is intentionally NOT shown.

Style is identical to App. Fig 6: serif (Times), RdYlBu_r colormap on a fixed
2-25 axis, white rho box, per-group box+strip sorted by median. The three rows use
three DIFFERENT perplexity definitions (labelled per colorbar); the shared 2-25
colour axis is used only because all three happen to occupy the same numeric band.
The comparable quantity across rows is the WITHIN-MODEL rho(PC1, PPL) and the
group ORDERING (the 5-tier ladder), not absolute PPL.

ESMC-600M loaders read the ESM data tree (esm_viral_probe / prokaryote_phage_ood /
esm_random_ood / esm3_masked_reconstruction), exactly as App. Fig 6. ProGen2 and
EvoDiff read the single-root contract of `cross_architecture_nativeness`; EvoDiff
PPL uses the training-faithful OA-ARDM ELBO column (`mean_perplexity_elbo`).
"""
from __future__ import annotations

import colorsys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import _common as c


# --- Colormap (matches main Fig 1A / App. Fig 6) ----------------------------
VMIN, VMAX = 2.0, 25.0
CMAP = mpl.colormaps["RdYlBu_r"]
_NORM = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)
PALE_SWAP = "#111111"


def color_for_ppl(ppl: float):
    return CMAP(_NORM(float(np.clip(ppl, VMIN, VMAX))))


def readable_text_color(rgba, max_lightness: float = 0.65):
    r, g, b, _ = rgba
    _, l, _ = colorsys.rgb_to_hls(r, g, b)
    return PALE_SWAP if l > max_lightness else rgba


# --- Groups (identical to App. Fig 6) ---------------------------------------
PCA_GROUPS = [
    ("Cellular",            "cellular"),
    ("Viral",               "viral"),
    ("Shuffled\ncellular",  "shuffled_nv"),
    ("Shuffled viral",      "shuffled_v"),
    ("Random uniform",      "random"),
]
KDE_GROUPS = [
    ("nonviral", "Human"), ("bacteria", "Bacteria"), ("archaea", "Archaea"),
    ("plants", "Plants"), ("fungi", "Fungi"), ("insects", "Insects"),
    ("phage", "Bacteriophage"), ("plant_virus", "Plant virus"),
    ("invertebrate_virus", "Invertebrate virus"), ("viral", "Human virus"),
    ("shuffled_nonviral", "Shuffled cellular"), ("shuffled_viral", "Shuffled virus"),
    ("random", "Random uniform"),
]
CELLULAR_PHAGE_GROUPS = ("archaea", "bacteria", "fungi", "insects", "plants")
VIRAL_PHAGE_GROUPS    = ("phage", "plant_virus", "invertebrate_virus")


# =============================================================================
#  Data backends:  "esm" (ESMC-600M, ESM data tree) vs "cross" (ProGen2/EvoDiff)
# =============================================================================

# --- ESM tree paths (from appfig_pca_ppl_esm2_esm3.py) ----------------------
HV_EMB_ROOT     = c.LAB_ROOT / "esm_viral_probe/datasets/human_virus/data/embeddings"
PHAGE_EMB_ROOT  = c.LAB_ROOT / "prokaryote_phage_ood/data/embeddings"
RANDOM_EMB_ROOT = c.LAB_ROOT / "esm_random_ood/data/embeddings"
MASKED_RECON    = c.LAB_ROOT / "esm3_masked_reconstruction/results"
PHAGE_PPL_ROOT  = c.LAB_ROOT / "prokaryote_phage_ood/results"

# --- cross-architecture single-root paths -----------------------------------
CROSS      = c.LAB_ROOT / "cross_architecture_nativeness"
CROSS_EMB  = CROSS / "data/embeddings"
CROSS_PPL  = CROSS / "results"

# Per-model spec: backend + display + colorbar label + EvoDiff PPL remap.
MODELS = [
    dict(key="esmc_600m", backend="esm",
         title="ESMC-600M — masked LM (600M)", cbar="Masked-reconstruction PPL",
         ppl_key="esmc_600m", ppl_col="mean_perplexity"),
    dict(key="progen2_base", backend="cross",
         title="ProGen2-base — autoregressive (764M)", cbar="Causal perplexity",
         ppl_key="progen2_base", ppl_col="mean_perplexity"),
    dict(key="evodiff_oadm_640m", backend="cross",
         title="EvoDiff OA-DM — discrete diffusion (640M)", cbar="OA-ARDM ELBO perplexity",
         ppl_key="evodiff_oadm_640m_elbo", ppl_col="mean_perplexity_elbo"),
]
SPEC = {m["key"]: m for m in MODELS}


def _strip_shuffle_prefix(accs, *, prefix):
    return np.asarray([a.replace(prefix, "", 1) if a.startswith(prefix) else a
                       for a in accs])


# --- ESM PPL loaders ---------------------------------------------------------
def _esm_human_ppl_tsv(mk):
    return MASKED_RECON / "esmc_600m/per_sequence_results.tsv"


def _esm_control_ppl_tsv(mk, group):
    return MASKED_RECON / f"esmc_600m/{group}/per_sequence_results.tsv"


def _esm_phage_ppl_tsv(mk, group):
    return PHAGE_PPL_ROOT / "masked_reconstruction" / f"{group}_ppl.tsv"


# --- cross PPL loaders -------------------------------------------------------
def _cross_human_ppl_tsv(mk):
    return CROSS_PPL / SPEC[mk]["ppl_key"] / "per_sequence_results.tsv"


def _cross_group_ppl_tsv(mk, group):
    return CROSS_PPL / SPEC[mk]["ppl_key"] / group / "per_sequence_results.tsv"


# --- dispatch helpers --------------------------------------------------------
def _ppl_col(mk):
    return SPEC[mk]["ppl_col"]


def _read_ppl_map(tsv_path, col, strip_prefixes=()):
    df = pd.read_csv(tsv_path, sep="\t")
    accs = df["accession"].astype(str)
    for p in strip_prefixes:
        accs = accs.str.replace(p, "", regex=False)
    return dict(zip(accs, df[col]))


def _human_ppl_for_label(mk, label):
    backend = SPEC[mk]["backend"]
    tsv = _esm_human_ppl_tsv(mk) if backend == "esm" else _cross_human_ppl_tsv(mk)
    df = pd.read_csv(tsv, sep="\t")
    sub = df[df["label"] == label]
    return dict(zip(sub["accession"].astype(str), sub[_ppl_col(mk)]))


def _group_ppl_map(mk, group, ctrl=False):
    """group is a kde key. ctrl=True for shuffled/random rows (different dir on ESM)."""
    backend = SPEC[mk]["backend"]
    col = _ppl_col(mk)
    if backend == "esm":
        if ctrl:
            name = "random_uniform" if group == "random" else group
            strip = {"shuffled_viral": ("SHUF_V_",), "shuffled_nonviral": ("SHUF_N_",)}.get(name, ())
            return _read_ppl_map(_esm_control_ppl_tsv(mk, name), col, strip)
        return _read_ppl_map(_esm_phage_ppl_tsv(mk, group), col)
    # cross
    name = "random_uniform" if group == "random" else group
    return _read_ppl_map(_cross_group_ppl_tsv(mk, name), col)


# --- embedding loaders -------------------------------------------------------
def _load_human_embeddings(mk, label):
    root = HV_EMB_ROOT if SPEC[mk]["backend"] == "esm" else CROSS_EMB
    Xs, accs = [], []
    for split in ("train", "val", "test"):
        d = np.load(root / mk / f"{label}_{split}.npz", allow_pickle=True)
        Xs.append(d["embeddings"]); accs.append(d["accessions"])
    return np.concatenate(Xs), np.concatenate(accs)


def _load_phage_embeddings(mk, group):
    root = PHAGE_EMB_ROOT if SPEC[mk]["backend"] == "esm" else CROSS_EMB
    d = np.load(root / mk / f"{group}.npz", allow_pickle=True)
    return d["embeddings"], d["accessions"]


def _load_control_embeddings(mk, name):
    if SPEC[mk]["backend"] == "esm":
        d = np.load(RANDOM_EMB_ROOT / mk / f"{name}.npz", allow_pickle=True)
        accs = d["accessions"]
        if name == "shuffled_viral":
            accs = _strip_shuffle_prefix(accs, prefix="SHUF_V_")
        elif name == "shuffled_nonviral":
            accs = _strip_shuffle_prefix(accs, prefix="SHUF_N_")
        return d["embeddings"], accs
    d = np.load(CROSS_EMB / mk / f"{name}.npz", allow_pickle=True)
    return d["embeddings"], d["accessions"]


def _align(X, accs, ppl_map):
    keep = np.asarray([a in ppl_map and np.isfinite(ppl_map[a]) for a in accs])
    y = np.asarray([ppl_map[a] for a in accs[keep]], dtype=float)
    return X[keep], y


def _group_data(mk, key):
    if key == "cellular":
        Xs, Ys = [], []
        X, acc = _load_human_embeddings(mk, "nonviral")
        Xk, yk = _align(X, acc, _human_ppl_for_label(mk, "nonviral"))
        if len(Xk): Xs.append(Xk); Ys.append(yk)
        for g in CELLULAR_PHAGE_GROUPS:
            X, acc = _load_phage_embeddings(mk, g)
            Xk, yk = _align(X, acc, _group_ppl_map(mk, g))
            if len(Xk): Xs.append(Xk); Ys.append(yk)
        return np.concatenate(Xs), np.concatenate(Ys)
    if key == "viral":
        Xs, Ys = [], []
        X, acc = _load_human_embeddings(mk, "viral")
        Xk, yk = _align(X, acc, _human_ppl_for_label(mk, "viral"))
        if len(Xk): Xs.append(Xk); Ys.append(yk)
        for g in VIRAL_PHAGE_GROUPS:
            X, acc = _load_phage_embeddings(mk, g)
            Xk, yk = _align(X, acc, _group_ppl_map(mk, g))
            if len(Xk): Xs.append(Xk); Ys.append(yk)
        return np.concatenate(Xs), np.concatenate(Ys)
    if key == "shuffled_nv":
        X, acc = _load_control_embeddings(mk, "shuffled_nonviral")
        return _align(X, acc, _group_ppl_map(mk, "shuffled_nonviral", ctrl=True))
    if key == "shuffled_v":
        X, acc = _load_control_embeddings(mk, "shuffled_viral")
        return _align(X, acc, _group_ppl_map(mk, "shuffled_viral", ctrl=True))
    if key == "random":
        X, acc = _load_control_embeddings(mk, "random_uniform")
        return _align(X, acc, _group_ppl_map(mk, "random", ctrl=True))
    raise ValueError(key)


def kde_ppl(mk, key):
    backend = SPEC[mk]["backend"]
    col = _ppl_col(mk)
    if key in ("viral", "nonviral"):
        tsv = _esm_human_ppl_tsv(mk) if backend == "esm" else _cross_human_ppl_tsv(mk)
        df = pd.read_csv(tsv, sep="\t")
        return df[df["label"] == key][col].values
    if key in ("shuffled_viral", "shuffled_nonviral", "random"):
        name = "random_uniform" if key == "random" else key
        tsv = _esm_control_ppl_tsv(mk, name) if backend == "esm" else _cross_group_ppl_tsv(mk, name)
        return pd.read_csv(tsv, sep="\t")[col].values
    tsv = _esm_phage_ppl_tsv(mk, key) if backend == "esm" else _cross_group_ppl_tsv(mk, key)
    return pd.read_csv(tsv, sep="\t")[col].values


# --- Panel drawing (identical idiom to App. Fig 6) ---------------------------
FIG_NAME = "appfig_crossarch_pca_ppl"


def model_pca_groups(mk):
    """Return ({group: (Z[n, 2], ppl[n])}, meta) for one model's PCA panel.

    Loads the committed coordinate cache when present; otherwise fits the PCA on
    that model's embeddings (PC1 oriented so low PPL sits left) and caches it."""
    cached = c.load_pca_cache(FIG_NAME, mk)
    if cached is not None:
        return cached

    group_data = {k: _group_data(mk, k) for _, k in PCA_GROUPS}
    all_X = np.concatenate([group_data[k][0] for _, k in PCA_GROUPS])
    all_ppl = np.concatenate([group_data[k][1] for _, k in PCA_GROUPS])
    pca = PCA(n_components=2, random_state=0).fit(all_X)
    full_Z = pca.transform(all_X)
    finite = np.isfinite(all_ppl)
    rho_signed, _ = spearmanr(full_Z[finite, 0], all_ppl[finite])
    pc1_flip = -1.0 if rho_signed < 0 else 1.0
    full_Z[:, 0] *= pc1_flip
    rho, _ = spearmanr(full_Z[finite, 0], all_ppl[finite])

    rho_within, group_coords, offset = {}, {}, 0
    for _, key in PCA_GROUPS:
        n_g = len(group_data[key][0])
        z_g, y_g = full_Z[offset:offset + n_g, 0], all_ppl[offset:offset + n_g]
        m = np.isfinite(y_g)
        rho_within[key] = float(spearmanr(z_g[m], y_g[m])[0]) if m.sum() >= 10 else None
        group_coords[key] = (full_Z[offset:offset + n_g], group_data[key][1])
        offset += n_g

    meta = {"evr": (float(pca.explained_variance_ratio_[0]),
                    float(pca.explained_variance_ratio_[1])),
            "rho": float(rho), "n": int(finite.sum()), "rho_within": rho_within}
    c.save_pca_cache(FIG_NAME, mk, group_coords, meta)
    return group_coords, meta


def panel_pca(fig, ax, mk, rng, cbar_label, budget_per_group=1500):
    group_coords, meta = model_pca_groups(mk)
    for _, k in PCA_GROUPS:
        Zk, y = group_coords[k]
        print(f"  [{mk}] {k:14s} n={len(Zk):6d}  mean PPL="
              f"{(y.mean() if len(y) else float('nan')):.2f}")

    rho = meta["rho"]
    n_total = meta["n"]
    rho_within = meta["rho_within"]
    print(f"  [{mk}] within-group rho: " +
          ", ".join(f"{k}={v:+.3f}" if v is not None else f"{k}=n/a"
                    for k, v in rho_within.items()))

    plot_Z, plot_ppl, plot_key = [], [], []
    for _, key in PCA_GROUPS:
        Zk, ppl = group_coords[key]
        idx = rng.permutation(len(Zk))[:budget_per_group]
        plot_Z.append(Zk[idx]); plot_ppl.append(ppl[idx]); plot_key.extend([key] * len(idx))
    Z = np.concatenate(plot_Z); ppl = np.concatenate(plot_ppl); plot_key = np.array(plot_key)

    order = rng.permutation(len(Z))
    sc = ax.scatter(Z[order, 0], Z[order, 1], c=np.clip(ppl[order], VMIN, VMAX),
                    cmap=CMAP, s=12, alpha=0.70, edgecolors="white", linewidths=0.25,
                    rasterized=True, vmin=VMIN, vmax=VMAX)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.015)
    cbar.set_label(cbar_label)
    cbar.outline.set_linewidth(0.8)

    entries = []
    for disp, key in PCA_GROUPS:
        m = plot_key == key
        if not m.any():
            continue
        med_ppl = float(np.median(group_coords[key][1])) if len(group_coords[key][1]) else 0.0
        entries.append({"disp": disp, "key": key, "cx": float(np.median(Z[m, 0])),
                        "cy": float(np.median(Z[m, 1])), "med_ppl": med_ppl})
    x_ext, y_ext = np.ptp(Z[:, 0]), np.ptp(Z[:, 1])
    dx_t, dy_t, dy_s = 0.14 * x_ext, 0.09 * y_ext, 0.12 * y_ext
    placed = []
    for e in entries:
        cy = e["cy"]
        while any(abs(e["cx"] - p["cx"]) < dx_t and abs(cy - p["cy"]) < dy_t for p in placed):
            cy += dy_s
        e["cy"] = cy; placed.append(e)
    for e in placed:
        col = color_for_ppl(e["med_ppl"])
        txt = ax.text(e["cx"], e["cy"], e["disp"].replace("\n", " "),
                      fontsize=15, fontweight="bold", ha="center", va="center", color=col,
                      bbox=dict(facecolor="white", edgecolor="#BDBDBD", lw=0.6,
                                boxstyle="round,pad=0.26", alpha=0.92))
        stroke_lw = 0.5 if e["key"] == "viral" else 0.3
        txt.set_path_effects([path_effects.Stroke(linewidth=stroke_lw, foreground="black"),
                              path_effects.Normal()])

    ax.set_xlabel(f"PC1 ({meta['evr'][0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({meta['evr'][1]*100:.1f}% var.)")
    ax.text(0.022, 0.025, rf"$\rho$ (PC1, PPL) $= {rho:+.3f}$   ($n = {n_total:,}$)",
            transform=ax.transAxes, ha="left", va="bottom", fontweight="bold",
            color="#111111", bbox=dict(facecolor="white", edgecolor="#1B1B1B", lw=1.3,
                                       boxstyle="round,pad=0.32"), zorder=10)
    return {"rho": float(rho), "rho_within": rho_within, "n": n_total,
            "pc1_var": float(meta["evr"][0]),
            "pc2_var": float(meta["evr"][1]),
            "group_counts": {k: int(len(group_coords[k][0])) for _, k in PCA_GROUPS},
            "group_mean_ppl": {k: (float(np.mean(group_coords[k][1])) if len(group_coords[k][1]) else None)
                               for _, k in PCA_GROUPS}}


def panel_strip(ax, mk, rng, cbar_label, max_points=400):
    stats = []
    for key, label in KDE_GROUPS:
        vals = np.asarray(kde_ppl(mk, key), dtype=float)
        vals = vals[np.isfinite(vals)]
        stats.append({"label": label, "vals": vals, "n": int(len(vals)),
                      "median": float(np.median(vals))})
    stats.sort(key=lambda s: s["median"]); stats = list(reversed(stats))
    y_positions = np.arange(len(stats))

    for y, s in zip(y_positions, stats):
        col = color_for_ppl(s["median"])
        vals = s["vals"]
        sample = vals if len(vals) <= max_points else vals[rng.choice(len(vals), max_points, replace=False)]
        ax.scatter(np.clip(sample, VMIN, VMAX), y + rng.uniform(-0.22, 0.22, len(sample)),
                   s=7, color=col, alpha=0.32, edgecolors="none", rasterized=True, zorder=2)
        ax.boxplot([vals], positions=[y], vert=False, widths=0.48, whis=(5, 95),
                   showfliers=False, patch_artist=True,
                   medianprops=dict(color="white", lw=1.6, zorder=6),
                   boxprops=dict(facecolor=col, edgecolor="black", lw=0.9, alpha=0.92, zorder=5),
                   whiskerprops=dict(color="black", lw=0.9, zorder=4),
                   capprops=dict(color="black", lw=0.9, zorder=4))
        ax.text(26.0, y, f"n={s['n']:,}", va="center", ha="right", color="#333",
                fontsize=10, zorder=7)
    ax.set_xlim(VMIN, 26.2)
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_ylim(-0.6, len(stats) - 0.4)
    ax.set_yticks(y_positions); ax.set_yticklabels([s["label"] for s in stats])
    ax.invert_yaxis()
    ax.set_xlabel(cbar_label)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="x", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    for y in y_positions[:-1]:
        ax.axhline(y + 0.5, color="#E0E0E0", lw=0.5, zorder=0)
    return {s["label"]: {"n": s["n"], "median": s["median"]} for s in stats}


def main():
    c.apply_style(font_size=17, axes_linewidth=2.2)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "svg.fonttype": "path",
    })
    rng = np.random.default_rng(7)

    n_rows = len(MODELS)
    fig = plt.figure(figsize=(16.0, 6.3 * n_rows))
    gs = GridSpec(n_rows, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0] * n_rows,
                  wspace=0.28, hspace=0.30, left=0.060, right=0.985, top=0.965, bottom=0.050)
    row_letters = (("A", "B"), ("C", "D"), ("E", "F"))

    stats = {}
    for row, m in enumerate(MODELS):
        mk, title, cbar = m["key"], m["title"], m["cbar"]
        ax_pca = fig.add_subplot(gs[row, 0])
        ax_strip = fig.add_subplot(gs[row, 1])
        pca_stats = panel_pca(fig, ax_pca, mk, rng, cbar)
        strip_stats = panel_strip(ax_strip, mk, rng, cbar)
        ax_pca.set_title(f"{title} — PCA", fontweight="bold", pad=6)
        ax_strip.set_title(f"{title} — per-group PPL", fontweight="bold", pad=6)
        stats[mk] = {**pca_stats, "strip": strip_stats}
        for ax, letter in zip((ax_pca, ax_strip), row_letters[row]):
            ax.text(-0.085 if ax is ax_pca else -0.19, 1.04, letter,
                    transform=ax.transAxes, fontsize=18, fontweight="bold",
                    va="bottom", ha="right")

    c.finalize(fig, "appfig_crossarch_pca_ppl", formats=("pdf", "png"))
    print("Saved figures/appfig_crossarch_pca_ppl.{pdf,png}")
    with open(c.HERE / "_appfig_crossarch_pca_ppl.json", "w") as fh:
        json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
