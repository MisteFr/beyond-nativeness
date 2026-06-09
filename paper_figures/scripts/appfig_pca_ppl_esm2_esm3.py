"""Appendix companion to main-text Fig 1 (PCA + PPL on ESMC-600M).

Reproduces both panels of main Fig 1 for three representative ESM checkpoints
(ESMC-600M at top as the main-text reference, plus ESM2-650M and ESM3-open) on
the *full* ten-group biological pool plus the three synthetic controls used in
main Fig 1 (cellular = archaea + bacteria + fungi + insects + plants + human
non-viral; viral = phage + plant_virus + invertebrate_virus + human viral; plus
shuffled cellular, shuffled viral, random uniform). Layout is 3 rows × 2 cols:
  row 1 (A, B): ESMC-600M  PCA scatter  |  per-group PPL strip/box
  row 2 (C, D): ESM2-650M  PCA scatter  |  per-group PPL strip/box
  row 3 (E, F): ESM3-open  PCA scatter  |  per-group PPL strip/box

The 8 non-human biological pools and the matching masked-reconstruction PPLs
for ESM2-650M and ESM3-open are produced by
`prokaryote_phage_ood/jobs/30_embed_ppl_esm2_650m_phage_ood.sh` and
`30_embed_ppl_esm3_open_phage_ood.sh`; the human pool and the three synthetic
controls reuse embeddings/PPLs that already lived under esm_viral_probe /
esm_random_ood / esm_zeroshot_ppl / esm3_masked_reconstruction.

The figure's purpose is to check that the qualitative claims of main Fig 1
hold outside ESMC on the same pool:
  (i)   a single low-dimensional axis captures most of the variance (Panel A);
  (ii)  that axis aligns with masked-reconstruction PPL (ρ near +1, Panel A);
  (iii) per-group medians order cellular → phage → viral → shuffled → random
        along the same PPL scale (Panel B).
"""
from __future__ import annotations

import colorsys
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


# --- Colormap (matches main Fig 1A) -----------------------------------------

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


# --- Groups ------------------------------------------------------------------
#
# Five display labels covering the full ten-group tree-of-life pool + three
# synthetic controls. "Cellular" aggregates 6 sources; "Viral" aggregates 4.
# This is the same aggregation used in the main Fig 1A PCA panel.

PCA_GROUPS = [
    ("Cellular",            "cellular"),
    ("Viral",               "viral"),
    ("Shuffled\ncellular",  "shuffled_nv"),
    ("Shuffled viral",      "shuffled_v"),
    ("Random uniform",      "random"),
]

# Panel B strip order — same 13 rows as main Fig 1B.
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
    ("shuffled_viral",     "Shuffled virus"),
    ("random",             "Random uniform"),
]

CELLULAR_PHAGE_GROUPS = ("archaea", "bacteria", "fungi", "insects", "plants")
VIRAL_PHAGE_GROUPS    = ("phage", "plant_virus", "invertebrate_virus")


# --- Paths -------------------------------------------------------------------

HV_EMB_ROOT      = c.LAB_ROOT / "esm_viral_probe/datasets/human_virus/data/embeddings"
PHAGE_EMB_ROOT   = c.LAB_ROOT / "prokaryote_phage_ood/data/embeddings"
RANDOM_EMB_ROOT  = c.LAB_ROOT / "esm_random_ood/data/embeddings"
ZEROSHOT_ROOT    = c.LAB_ROOT / "esm_zeroshot_ppl/results"
MASKED_RECON     = c.LAB_ROOT / "esm3_masked_reconstruction/results"
PHAGE_PPL_ROOT   = c.LAB_ROOT / "prokaryote_phage_ood/results"


# --- Accession helpers -------------------------------------------------------

def _strip_shuffle_prefix(accs: np.ndarray, *, prefix: str) -> np.ndarray:
    """The shuffled FASTAs re-prefix accessions with SHUF_V_ / SHUF_N_ so that
    downstream TSVs never collide with the original human pool. Strip the
    prefix to line PPL TSVs up with embedding NPZs."""
    return np.asarray([a.replace(prefix, "", 1) if a.startswith(prefix) else a
                       for a in accs])


# --- PPL loaders -------------------------------------------------------------

def _human_ppl_tsv(model_key: str) -> Path:
    """Per-sequence PPL for the human pool (viral + nonviral)."""
    if model_key == "esm3_open":
        # ESM3-open human pool PPLs live at the top of esm3_masked_reconstruction.
        return MASKED_RECON / "per_sequence_results.tsv"
    if model_key == "esmc_600m":
        return MASKED_RECON / "esmc_600m/per_sequence_results.tsv"
    return ZEROSHOT_ROOT / model_key / "per_sequence_results.tsv"


def _control_ppl_tsv(model_key: str, group: str) -> Path:
    """Per-sequence PPL for shuffled_viral / shuffled_nonviral / random_uniform."""
    if model_key == "esm3_open":
        return MASKED_RECON / f"{group}/per_sequence_results.tsv"
    if model_key == "esmc_600m":
        return MASKED_RECON / f"esmc_600m/{group}/per_sequence_results.tsv"
    return ZEROSHOT_ROOT / model_key / f"{group}/per_sequence_results.tsv"


def _phage_ppl_tsv(model_key: str, group: str) -> Path:
    """Per-sequence PPL for one of the 8 prokaryote_phage_ood groups."""
    if model_key == "esmc_600m":
        # ESMC-600M phage_ood PPL was produced by the main-text pipeline with a
        # flat `{group}_ppl.tsv` layout rather than per-group subdirectories.
        return PHAGE_PPL_ROOT / "masked_reconstruction" / f"{group}_ppl.tsv"
    return PHAGE_PPL_ROOT / f"masked_reconstruction_{model_key}" / group / "per_sequence_results.tsv"


def _read_ppl_map(tsv_path: Path, strip_prefixes: tuple[str, ...] = ()) -> dict[str, float]:
    """Accession → PPL map from a per_sequence_results.tsv. Optional shuffled-
    prefix stripping keeps things consistent with embedding NPZs."""
    df = pd.read_csv(tsv_path, sep="\t")
    accs = df["accession"].astype(str)
    for p in strip_prefixes:
        accs = accs.str.replace(p, "", regex=False)
    return dict(zip(accs, df["mean_perplexity"]))


def _human_ppl_for_label(model_key: str, label: str) -> dict[str, float]:
    """label in {'viral','nonviral'}; accessions match embedding NPZ (bare
    UniProt for viral, 'sp|ACC|NAME' for nonviral)."""
    df = pd.read_csv(_human_ppl_tsv(model_key), sep="\t")
    sub = df[df["label"] == label]
    return dict(zip(sub["accession"].astype(str), sub["mean_perplexity"]))


# --- Embedding loaders -------------------------------------------------------

def _load_human_embeddings(model_key: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    Xs, accs = [], []
    for split in ("train", "val", "test"):
        d = np.load(HV_EMB_ROOT / model_key / f"{label}_{split}.npz", allow_pickle=True)
        Xs.append(d["embeddings"]); accs.append(d["accessions"])
    return np.concatenate(Xs), np.concatenate(accs)


def _load_phage_embeddings(model_key: str, group: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(PHAGE_EMB_ROOT / model_key / f"{group}.npz", allow_pickle=True)
    return d["embeddings"], d["accessions"]


def _load_control_embeddings(model_key: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(RANDOM_EMB_ROOT / model_key / f"{name}.npz", allow_pickle=True)
    accs = d["accessions"]
    if name == "shuffled_viral":
        accs = _strip_shuffle_prefix(accs, prefix="SHUF_V_")
    elif name == "shuffled_nonviral":
        accs = _strip_shuffle_prefix(accs, prefix="SHUF_N_")
    return d["embeddings"], accs


# --- Group assembly ----------------------------------------------------------

def _align(X: np.ndarray, accs: np.ndarray, ppl_map: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    keep = np.asarray([a in ppl_map and np.isfinite(ppl_map[a]) for a in accs])
    y = np.asarray([ppl_map[a] for a in accs[keep]], dtype=float)
    return X[keep], y


def _group_data(model_key: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, ppl) for one of the 5 PCA display groups. `cellular` and
    `viral` aggregate multiple source pools (tree-of-life + human)."""

    if key == "cellular":
        Xs, Ys = [], []
        # Human non-viral
        X, acc = _load_human_embeddings(model_key, "nonviral")
        Xk, yk = _align(X, acc, _human_ppl_for_label(model_key, "nonviral"))
        if len(Xk): Xs.append(Xk); Ys.append(yk)
        # Prokaryote / phage_ood cellular eukaryotes
        for g in CELLULAR_PHAGE_GROUPS:
            X, acc = _load_phage_embeddings(model_key, g)
            Xk, yk = _align(X, acc, _read_ppl_map(_phage_ppl_tsv(model_key, g)))
            if len(Xk): Xs.append(Xk); Ys.append(yk)
        return np.concatenate(Xs), np.concatenate(Ys)

    if key == "viral":
        Xs, Ys = [], []
        # Human viral
        X, acc = _load_human_embeddings(model_key, "viral")
        Xk, yk = _align(X, acc, _human_ppl_for_label(model_key, "viral"))
        if len(Xk): Xs.append(Xk); Ys.append(yk)
        # Tree-of-life viral (phage + plant/invertebrate virus)
        for g in VIRAL_PHAGE_GROUPS:
            X, acc = _load_phage_embeddings(model_key, g)
            Xk, yk = _align(X, acc, _read_ppl_map(_phage_ppl_tsv(model_key, g)))
            if len(Xk): Xs.append(Xk); Ys.append(yk)
        return np.concatenate(Xs), np.concatenate(Ys)

    if key == "shuffled_nv":
        X, acc = _load_control_embeddings(model_key, "shuffled_nonviral")
        ppl = _read_ppl_map(_control_ppl_tsv(model_key, "shuffled_nonviral"),
                            strip_prefixes=("SHUF_N_",))
        return _align(X, acc, ppl)

    if key == "shuffled_v":
        X, acc = _load_control_embeddings(model_key, "shuffled_viral")
        ppl = _read_ppl_map(_control_ppl_tsv(model_key, "shuffled_viral"),
                            strip_prefixes=("SHUF_V_",))
        return _align(X, acc, ppl)

    if key == "random":
        X, acc = _load_control_embeddings(model_key, "random_uniform")
        ppl = _read_ppl_map(_control_ppl_tsv(model_key, "random_uniform"))
        return _align(X, acc, ppl)

    raise ValueError(key)


# --- Panel B: per-group PPL loader -------------------------------------------

def kde_ppl(model_key: str, key: str) -> np.ndarray:
    """Return the per-sequence PPL vector for one strip row under `model_key`."""
    if key in ("viral", "nonviral"):
        df = pd.read_csv(_human_ppl_tsv(model_key), sep="\t")
        return df[df["label"] == key]["mean_perplexity"].values
    if key in ("shuffled_viral", "shuffled_nonviral", "random"):
        name = "random_uniform" if key == "random" else key
        df = pd.read_csv(_control_ppl_tsv(model_key, name), sep="\t")
        return df["mean_perplexity"].values
    # 8 phage_ood groups
    df = pd.read_csv(_phage_ppl_tsv(model_key, key), sep="\t")
    return df["mean_perplexity"].values


# --- Panel drawing -----------------------------------------------------------

FIG_NAME = "appfig_pca_ppl_esm2_esm3"


def model_pca_groups(model_key: str):
    """Return ({group: (Z[n, 2], ppl[n])}, meta) for one model's PCA panel.

    Loads the committed coordinate cache when present; otherwise fits the PCA on
    that model's embeddings (PC1 oriented so low PPL sits left) and writes the
    cache (regeneration path)."""
    cached = c.load_pca_cache(FIG_NAME, model_key)
    if cached is not None:
        return cached

    group_data = {k: _group_data(model_key, k) for _, k in PCA_GROUPS}
    all_X = np.concatenate([group_data[k][0] for _, k in PCA_GROUPS])
    all_ppl = np.concatenate([group_data[k][1] for _, k in PCA_GROUPS])
    pca = PCA(n_components=2, random_state=0).fit(all_X)

    # Pin PC1 orientation so low PPL sits on the left, matching main Fig 1A.
    full_Z = pca.transform(all_X)
    finite = np.isfinite(all_ppl)
    rho_signed, _ = spearmanr(full_Z[finite, 0], all_ppl[finite])
    pc1_flip = -1.0 if rho_signed < 0 else 1.0
    full_Z[:, 0] *= pc1_flip
    rho, _ = spearmanr(full_Z[finite, 0], all_ppl[finite])

    group_coords, off = {}, 0
    for _, k in PCA_GROUPS:
        n = len(group_data[k][0])
        group_coords[k] = (full_Z[off:off + n], group_data[k][1])
        off += n

    meta = {
        "evr": (float(pca.explained_variance_ratio_[0]),
                float(pca.explained_variance_ratio_[1])),
        "rho": float(rho),
        "n": int(finite.sum()),
        "rho_within": {},
    }
    c.save_pca_cache(FIG_NAME, model_key, group_coords, meta)
    return group_coords, meta


def panel_pca(fig, ax, model_key: str, rng, budget_per_group: int = 1500):
    group_coords, meta = model_pca_groups(model_key)
    for disp, k in PCA_GROUPS:
        Zk, y = group_coords[k]
        print(f"  [{model_key}] {k:14s} n = {len(Zk):6d}  mean PPL = "
              f"{(y.mean() if len(y) else float('nan')):.2f}  "
              f"[{(y.min() if len(y) else float('nan')):.2f}, "
              f"{(y.max() if len(y) else float('nan')):.2f}]")

    rho = meta["rho"]
    n_total = meta["n"]

    plot_Z, plot_ppl, plot_key = [], [], []
    for _, key in PCA_GROUPS:
        Zk, ppl = group_coords[key]
        idx = rng.permutation(len(Zk))[:budget_per_group]
        plot_Z.append(Zk[idx])
        plot_ppl.append(ppl[idx])
        plot_key.extend([key] * len(idx))
    Z = np.concatenate(plot_Z)
    ppl = np.concatenate(plot_ppl)
    plot_key = np.array(plot_key)

    order = rng.permutation(len(Z))
    sc = ax.scatter(
        Z[order, 0], Z[order, 1],
        c=np.clip(ppl[order], VMIN, VMAX),
        cmap=CMAP, s=12, alpha=0.70,
        edgecolors="white", linewidths=0.25,
        rasterized=True, vmin=VMIN, vmax=VMAX,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.015)
    cbar.set_label("Masked-reconstruction PPL")
    cbar.outline.set_linewidth(0.8)

    entries = []
    for disp, key in PCA_GROUPS:
        m = plot_key == key
        if not m.any():
            continue
        cx, cy = float(np.median(Z[m, 0])), float(np.median(Z[m, 1]))
        med_ppl = float(np.median(group_coords[key][1])) if len(group_coords[key][1]) else 0.0
        entries.append({"disp": disp, "key": key, "cx": cx, "cy": cy, "med_ppl": med_ppl})

    # Collision resolution: push overlapping labels along PC2.
    x_extent = Z[:, 0].max() - Z[:, 0].min()
    y_extent = Z[:, 1].max() - Z[:, 1].min()
    dx_thresh = 0.14 * x_extent
    dy_thresh = 0.09 * y_extent
    dy_step   = 0.12 * y_extent

    placed = []
    for e in entries:
        cx, cy = e["cx"], e["cy"]
        while any(abs(cx - p["cx"]) < dx_thresh and abs(cy - p["cy"]) < dy_thresh
                  for p in placed):
            cy += dy_step
        e["cy"] = cy
        placed.append(e)

    for e in placed:
        col = color_for_ppl(e["med_ppl"])
        txt = ax.text(e["cx"], e["cy"], e["disp"].replace("\n", " "),
                fontsize=15, fontweight="bold", ha="center", va="center",
                color=col,
                bbox=dict(facecolor="white", edgecolor="#BDBDBD",
                          lw=0.6, boxstyle="round,pad=0.26", alpha=0.92))
        stroke_lw = 0.5 if e["key"] == "viral" else 0.3
        txt.set_path_effects([
            path_effects.Stroke(linewidth=stroke_lw, foreground="black"),
            path_effects.Normal(),
        ])

    ax.set_xlabel(f"PC1 ({meta['evr'][0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({meta['evr'][1]*100:.1f}% var.)")

    ax.text(
        0.022, 0.025,
        rf"$\rho$ (PC1, PPL) $= {rho:+.3f}$   ($n = {n_total:,}$)",
        transform=ax.transAxes, ha="left", va="bottom",
        fontweight="bold", color="#111111",
        bbox=dict(facecolor="white", edgecolor="#1B1B1B", lw=1.3,
                  boxstyle="round,pad=0.32"),
        zorder=10,
    )
    return {
        "rho": float(rho),
        "n": n_total,
        "pc1_var": float(meta["evr"][0]),
        "pc2_var": float(meta["evr"][1]),
        "group_counts": {k: int(len(group_coords[k][0])) for _, k in PCA_GROUPS},
        "group_mean_ppl": {k: float(np.mean(group_coords[k][1])) if len(group_coords[k][1]) else None
                           for _, k in PCA_GROUPS},
    }


def panel_strip(ax, model_key: str, rng, max_points: int = 400):
    """Per-group PPL strip/box on the same PPL colormap as panel A."""
    stats = []
    for key, label in KDE_GROUPS:
        vals = np.asarray(kde_ppl(model_key, key), dtype=float)
        vals = vals[np.isfinite(vals)]
        stats.append({"label": label, "vals": vals,
                      "n": int(len(vals)), "median": float(np.median(vals))})
    # Highest median at top, matching main Fig 1B.
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
                   s=7, color=col, alpha=0.32,
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
                va="center", ha="right", color="#333", fontsize=10,
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

    return {s["label"]: {"n": s["n"], "median": s["median"]} for s in stats}


# --- Main --------------------------------------------------------------------

def main() -> None:
    c.apply_style(font_size=17, axes_linewidth=2.2)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "svg.fonttype": "path",
    })
    rng = np.random.default_rng(7)

    # ESMC-600M leads as the main-text reference, then the two new
    # architectures ESM2-650M and ESM3-open (both ~open-weight flagships of
    # their family). Parameter counts annotated in titles; ESM3-open's is not
    # obvious from its name.
    models = [
        ("esmc_600m", "ESMC-600M (600M params)"),
        ("esm2_650m", "ESM2-650M (650M params)"),
        ("esm3_open", "ESM3-open (1.4B params)"),
    ]

    n_rows = len(models)
    fig = plt.figure(figsize=(16.0, 6.3 * n_rows))
    gs = GridSpec(
        n_rows, 2, width_ratios=[1.15, 1.0],
        height_ratios=[1.0] * n_rows,
        wspace=0.28, hspace=0.30,
        left=0.060, right=0.985, top=0.960, bottom=0.055,
    )

    row_letters = (("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"))

    stats = {}
    for row, (mk, title) in enumerate(models):
        # Guard: if neither the committed PCA cache nor the raw embedding/PPL
        # outputs are in place, print a clear message and draw a placeholder
        # panel so the script is still runnable (e.g. before regeneration jobs
        # finish). With the vendored cache present this guard always passes.
        missing = []
        if c.load_pca_cache(FIG_NAME, mk) is None:
            for g in (*CELLULAR_PHAGE_GROUPS, *VIRAL_PHAGE_GROUPS):
                emb = PHAGE_EMB_ROOT / mk / f"{g}.npz"
                ppl = _phage_ppl_tsv(mk, g)
                if not emb.exists() or not ppl.exists():
                    missing.append(g)

        ax_pca   = fig.add_subplot(gs[row, 0])
        ax_strip = fig.add_subplot(gs[row, 1])

        if missing:
            msg = (f"{title} 10-group pool incomplete\n"
                   f"  missing: {', '.join(missing)}\n"
                   f"  (run jobs/30_embed_ppl_{mk}_phage_ood.sh)")
            print(f"[WARN] {msg}")
            for ax in (ax_pca, ax_strip):
                ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, color="#555")
                ax.set_xticks([]); ax.set_yticks([])
            continue

        pca_stats   = panel_pca(fig, ax_pca, mk, rng)
        strip_stats = panel_strip(ax_strip, mk, rng)
        ax_pca.set_title(f"{title} — PCA", fontweight="bold", pad=6)
        ax_strip.set_title(f"{title} — per-group PPL", fontweight="bold", pad=6)
        stats[mk] = {**pca_stats, "strip": strip_stats}

        letters = row_letters[row]
        for ax, letter in zip((ax_pca, ax_strip), letters):
            ax.text(-0.085 if ax is ax_pca else -0.19, 1.04, letter,
                    transform=ax.transAxes, fontsize=18,
                    fontweight="bold", va="bottom", ha="right")

    c.finalize(fig, "appfig_pca_ppl_esm2_esm3",
               formats=("pdf", "png", "svg"))
    print("Saved figures/appfig_pca_ppl_esm2_esm3.{pdf,png,svg}")
    print("Stats:", stats)

    # Dump numbers so the paper text can be locked.
    import json
    with open(c.HERE / "_appfig_pca_ppl_esm2_esm3.json", "w") as fh:
        json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
