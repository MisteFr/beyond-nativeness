"""Variant of `fig_pca_and_ppl_strip` that swaps the BlueVioletRed colormap
for the `RdYlBu_r` gradient used in
`prokaryote_phage_ood/figures/pca_ppl_continuous.png`. Same data, same layout.
"""
from __future__ import annotations

import colorsys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import _common as c


VMIN, VMAX = 2.0, 25.0
CMAP = mpl.colormaps["RdYlBu_r"]
_NORM = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)

PALE_SWAP = "#111111"


def color_for_ppl(ppl: float):
    return CMAP(_NORM(float(np.clip(ppl, VMIN, VMAX))))


def readable_text_color(rgba, max_lightness: float = 0.65):
    r, g, b, _ = rgba
    _, l, _ = colorsys.rgb_to_hls(r, g, b)
    if l > max_lightness:
        return PALE_SWAP
    return rgba


PCA_GROUPS = [
    ("Cellular",             "cellular"),
    ("Viral",                "viral"),
    ("Shuffled\ncellular",   "shuffled_nv"),
    ("Shuffled viral",       "shuffled"),
    ("Uniform random",       "random"),
]


KDE_GROUPS = [
    ("nonviral",           "Human"),
    ("bacteria",           "Bacteria"),
    ("archaea",            "Archaea"),
    ("plants",             "Plants"),
    ("fungi",              "Fungi"),
    ("insects",            "Insects"),
    ("phage",              "Bacteriophage"),
    ("plant_virus",        "Plant virus"),
    ("invertebrate_virus", "Invertebrate virus"),
    ("viral",              "Human virus"),
    ("shuffled_nonviral",  "Shuffled cellular"),
    ("shuffled_viral",     "Shuffled viral"),
    ("random",             "Uniform random"),
]


PHAGE_EMB  = c.LAB_ROOT / "prokaryote_phage_ood/data/embeddings/esmc_600m"
PHAGE_SHUF = c.LAB_ROOT / "prokaryote_phage_ood/data/embeddings_shuffled/esmc_600m"
HV_EMB     = c.LAB_ROOT / "esm_viral_probe/datasets/human_virus/data/embeddings/esmc_600m"


def _collect(emb_iter, ppl_map):
    pairs = [(emb, ppl_map[a]) for emb, a in emb_iter
             if a in ppl_map and np.isfinite(ppl_map[a])]
    if not pairs:
        return np.empty((0, 0)), np.empty((0,))
    return (np.stack([p[0] for p in pairs]),
            np.asarray([p[1] for p in pairs]))


def _phage_group(name: str):
    d = np.load(PHAGE_EMB / f"{name}.npz")
    df = pd.read_csv(c.PHAGE_OOD / f"masked_reconstruction/{name}_ppl.tsv", sep="\t")
    pm = dict(zip(df["accession"], df["mean_perplexity"]))
    return _collect(zip(d["embeddings"], d["accessions"]), pm)


def _human_virus(label: str):
    Xs, accs = [], []
    for s in ("train", "val", "test"):
        d = np.load(HV_EMB / f"{label}_{s}.npz")
        Xs.append(d["embeddings"]); accs.append(d["accessions"])
    df = c.esmc_600m_per_seq()
    pm = dict(zip(df[df["label"] == label]["accession"],
                  df[df["label"] == label]["mean_perplexity"]))
    return _collect(zip(np.concatenate(Xs), np.concatenate(accs)), pm)


def paired_sequences(key: str) -> tuple[np.ndarray, np.ndarray]:
    if key == "cellular":
        Xs, Ys = [], []
        for g in ("archaea", "bacteria", "fungi", "insects", "plants"):
            X, y = _phage_group(g)
            if len(X): Xs.append(X); Ys.append(y)
        X, y = _human_virus("nonviral")
        if len(X): Xs.append(X); Ys.append(y)
        return np.concatenate(Xs), np.concatenate(Ys)
    if key == "viral":
        Xs, Ys = [], []
        for g in ("phage", "plant_virus", "invertebrate_virus"):
            X, y = _phage_group(g)
            if len(X): Xs.append(X); Ys.append(y)
        X, y = _human_virus("viral")
        if len(X): Xs.append(X); Ys.append(y)
        return np.concatenate(Xs), np.concatenate(Ys)
    if key == "shuffled":
        d = np.load(PHAGE_SHUF / "shuffled_viral.npz")
        accs = [a.replace("SHUF_viral_", "") for a in d["accessions"]]
        df = pd.read_csv(c.MASKED_RECON / "shuffled_viral/per_sequence_results.tsv", sep="\t")
        pm = dict(zip(df["accession"].str.replace("SHUF_V_", "", regex=False),
                      df["mean_perplexity"]))
        return _collect(zip(d["embeddings"], accs), pm)
    if key == "shuffled_nv":
        d = np.load(PHAGE_SHUF / "shuffled_nonviral.npz")
        accs = [a.replace("SHUF_nonviral_", "") for a in d["accessions"]]
        df = pd.read_csv(c.MASKED_RECON / "shuffled_nonviral/per_sequence_results.tsv", sep="\t")
        pm = dict(zip(df["accession"].str.replace("SHUF_N_", "", regex=False),
                      df["mean_perplexity"]))
        return _collect(zip(d["embeddings"], accs), pm)
    if key == "random":
        d = np.load(PHAGE_SHUF / "random_uniform.npz")
        df = pd.read_csv(c.MASKED_RECON / "random_uniform/per_sequence_results.tsv", sep="\t")
        pm = dict(zip(df["accession"], df["mean_perplexity"]))
        return _collect(zip(d["embeddings"], d["accessions"]), pm)
    raise ValueError(key)


def kde_ppl(key: str) -> np.ndarray:
    if key == "viral":
        df = c.esmc_600m_per_seq()
        return df[df["label"] == "viral"]["mean_perplexity"].values
    if key == "nonviral":
        df = c.esmc_600m_per_seq()
        return df[df["label"] == "nonviral"]["mean_perplexity"].values
    if key == "shuffled_viral":
        df = pd.read_csv(c.MASKED_RECON / "shuffled_viral/per_sequence_results.tsv", sep="\t")
        return df["mean_perplexity"].values
    if key == "shuffled_nonviral":
        df = pd.read_csv(c.MASKED_RECON / "shuffled_nonviral/per_sequence_results.tsv", sep="\t")
        return df["mean_perplexity"].values
    if key == "random":
        df = pd.read_csv(c.MASKED_RECON / "random_uniform/per_sequence_results.tsv", sep="\t")
        return df["mean_perplexity"].values
    tsv = c.PHAGE_OOD / f"masked_reconstruction/{key}_ppl.tsv"
    return pd.read_csv(tsv, sep="\t")["mean_perplexity"].values


FIG_NAME = "fig_pca_and_ppl_strip"
MODEL_KEY = "esmc_600m"


def model_pca_groups():
    """Return ({group: (Z[n, 2], ppl[n])}, meta) for the PCA panel.

    Loads the committed coordinate cache when present; otherwise fits the PCA on
    the ESMC-600M embeddings and writes the cache (regeneration path)."""
    cached = c.load_pca_cache(FIG_NAME, MODEL_KEY)
    if cached is not None:
        return cached

    group_data = {k: paired_sequences(k) for _, k in PCA_GROUPS}
    all_X = np.concatenate([group_data[k][0] for _, k in PCA_GROUPS])
    pca = PCA(n_components=2, random_state=0).fit(all_X)
    full_Z = pca.transform(all_X)

    group_coords, off = {}, 0
    for _, k in PCA_GROUPS:
        n = len(group_data[k][0])
        group_coords[k] = (full_Z[off:off + n], group_data[k][1])
        off += n

    full_ppl = np.concatenate([group_data[k][1] for _, k in PCA_GROUPS])
    finite = np.isfinite(full_ppl)
    rho, _ = spearmanr(full_Z[finite, 0], full_ppl[finite])
    meta = {
        "evr": (float(pca.explained_variance_ratio_[0]),
                float(pca.explained_variance_ratio_[1])),
        "rho": float(rho),
        "n": int(finite.sum()),
        "rho_within": {},
    }
    c.save_pca_cache(FIG_NAME, MODEL_KEY, group_coords, meta)
    return group_coords, meta


def panel_pca(fig, ax, group_coords, meta, rng, budget_per_group: int = 1500):
    plot_Z, plot_ppl, plot_key = [], [], []
    for _, key in PCA_GROUPS:
        Zg, ppl = group_coords[key]
        idx = rng.permutation(len(Zg))[:budget_per_group]
        plot_Z.append(Zg[idx])
        plot_ppl.append(ppl[idx])
        plot_key.extend([key] * len(idx))
    Z = np.concatenate(plot_Z)
    ppl = np.concatenate(plot_ppl)
    plot_key = np.array(plot_key)

    rho = meta["rho"]
    n_total = meta["n"]

    order = rng.permutation(len(Z))
    sc = ax.scatter(Z[order, 0], Z[order, 1],
                    c=np.clip(ppl[order], VMIN, VMAX),
                    cmap=CMAP, s=14, alpha=0.70,
                    edgecolors="white", linewidths=0.3,
                    rasterized=True, vmin=VMIN, vmax=VMAX)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.015)
    cbar.set_label("Masked-reconstruction PPL")
    cbar.outline.set_linewidth(0.8)

    for disp, key in PCA_GROUPS:
        m = plot_key == key
        if not m.any():
            continue
        cx, cy = np.median(Z[m, 0]), np.median(Z[m, 1])
        med_ppl = float(np.median(group_coords[key][1]))
        col = color_for_ppl(med_ppl)
        txt = ax.text(cx, cy, disp.replace("\n", " "),
                fontsize=15, fontweight="bold", ha="center", va="center",
                color=col,
                bbox=dict(facecolor="white", edgecolor="#BDBDBD",
                          lw=0.6, boxstyle="round,pad=0.28", alpha=0.92))
        stroke_lw = 0.5 if key == "viral" else 0.3
        txt.set_path_effects([
            path_effects.Stroke(linewidth=stroke_lw, foreground="black"),
            path_effects.Normal(),
        ])

    ax.set_xlabel(f"PC1 ({meta['evr'][0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({meta['evr'][1]*100:.1f}% var.)")
    ax.text(
        0.022, 0.025,
        rf"$\rho$ (PC1, PPL) $= {rho:+.3f}$",
        transform=ax.transAxes, ha="left", va="bottom",
        fontweight="bold", color="#111111",
        bbox=dict(facecolor="white", edgecolor="#1B1B1B", lw=1.5,
                  boxstyle="round,pad=0.35"),
        zorder=10,
    )
    _ = n_total


def panel_strip(ax, kde_data, rng, max_points: int = 400):
    stats = []
    for key, label in KDE_GROUPS:
        vals = np.asarray(kde_data[key], dtype=float)
        vals = vals[np.isfinite(vals)]
        stats.append({
            "label": label,
            "vals": vals,
            "n": len(vals),
            "median": float(np.median(vals)),
        })
    stats.sort(key=lambda s: s["median"])
    stats = list(reversed(stats))

    n_rows = len(stats)
    y_positions = np.arange(n_rows)

    for y, s in zip(y_positions, stats):
        col = color_for_ppl(s["median"])
        vals = s["vals"]

        if len(vals) > max_points:
            idx = rng.choice(len(vals), size=max_points, replace=False)
            sample = vals[idx]
        else:
            sample = vals
        jitter = rng.uniform(-0.22, 0.22, size=len(sample))
        ax.scatter(np.clip(sample, VMIN, VMAX), y + jitter,
                   s=8, color=col, alpha=0.32,
                   edgecolors="none", rasterized=True, zorder=2)

        ax.boxplot(
            [vals],
            positions=[y],
            vert=False,
            widths=0.48,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="white", lw=1.6, zorder=6),
            boxprops=dict(facecolor=col, edgecolor="black", lw=0.9,
                          alpha=0.92, zorder=5),
            whiskerprops=dict(color="black", lw=0.9, zorder=4),
            capprops=dict(color="black", lw=0.9, zorder=4),
        )

        ax.text(26.0, y, f"n={s['n']:,}",
                va="center", ha="right", color="#333",
                fontsize=11,
                zorder=7)

    ax.set_xlim(VMIN, 26.2)
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s["label"] for s in stats])
    ax.invert_yaxis()
    ax.set_xlabel("Masked-reconstruction PPL")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    for y in y_positions[:-1]:
        ax.axhline(y + 0.5, color="#E0E0E0", lw=0.5, zorder=0)


def main():
    c.apply_style(font_size=17, axes_linewidth=2.2)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "svg.fonttype": "path",
    })
    rng = np.random.default_rng(7)

    print("Loading PCA coordinates …")
    group_coords, meta = model_pca_groups()
    for _, k in PCA_GROUPS:
        Z, y = group_coords[k]
        print(f"  {k:14s} n = {len(Z):6d}  mean PPL = {y.mean():.2f}")

    print("Loading strip groups …")
    kde_data = {g[0]: kde_ppl(g[0]) for g in KDE_GROUPS}
    for k, _ in KDE_GROUPS:
        v = kde_data[k]
        print(f"  {k:20s} n = {len(v):6d}  median = {np.median(v):.2f}")

    fig = plt.figure(figsize=(18.0, 8.6))
    gs = GridSpec(
        1, 2,
        width_ratios=[1.15, 1.0],
        wspace=0.30,
        left=0.065, right=0.985, top=0.940, bottom=0.115,
    )
    ax_pca   = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[0, 1])

    panel_pca(fig, ax_pca, group_coords, meta, rng)
    panel_strip(ax_strip, kde_data, rng)

    ax_pca.text(-0.090, 1.02, "A", transform=ax_pca.transAxes, fontsize=22,
                fontweight="bold", va="bottom", ha="right")
    ax_strip.text(-0.175, 1.02, "B", transform=ax_strip.transAxes, fontsize=22,
                  fontweight="bold", va="bottom", ha="right")

    c.finalize(fig, "fig_pca_and_ppl_strip",
               formats=("pdf", "png", "svg"))
    print("Saved figures/fig_pca_and_ppl_strip.{pdf,png,svg}")


if __name__ == "__main__":
    main()
