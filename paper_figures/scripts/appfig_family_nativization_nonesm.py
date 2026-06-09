"""Per-family viral nativization across scale for NON-ESM pLMs.

Cross-architecture companion to `fig_family_nativization_esmc.py` (main Fig 2) and
`appfig_family_nativization_all.py` (App. Fig 5, ESM2/ESM3). Reproduces the §4.2
result — *scale contracts the nativeness axis heterogeneously across viral families* —
on the non-ESM architectures scored in `cross_architecture_nativeness`:

  ProGen2 (autoregressive)        4-point ladder: 151M / 764M / 2.7B / 6.4B
  EvoDiff OA-DM (discrete diff.)  2-point ladder: 38M / 640M
  ProtT5-XL (T5 span-denoise)     single point (reference only -> sidecar, no line)

Because absolute PPL is NOT comparable across these objectives (causal vs masked vs
span-denoise; ProGen2 cellular mean 7.24, ProtT5 natural PPL ~1.6), the paper's fixed
PPL<5 native-like bar is meaningless here. Instead we set a PER-ARCHITECTURE
native-like threshold tau = the 90th percentile of that architecture's cellular
(non-viral) PPL at its *reference scale* (ProGen2-base, EvoDiff-640M), held FIXED across
the architecture's scales (mirrors the paper's "cellular mostly below the bar"). The
companion figure is threshold-free (per-family MEDIAN PPL vs scale), which also makes
ProtT5's compressed PPL range visible.

The 8 viral families and their colours are inherited from the ESMC-6B ranking of main
Fig 2, so a family reads identically across the ESM and non-ESM figures and the
cross-architecture question — "do the SAME families nativize off the ESM family?" — is
read directly.

Source:
  cross_architecture_nativeness/results/{model_key}/per_sequence_results.tsv
  esm_viral_probe/.../leave_family_out/family_metadata.tsv  (family labels)
Outputs:
  figures/appfig_family_nativization_nonesm.{pdf,png}   (% of family with PPL < tau)
  figures/appfig_family_medianppl_nonesm.{pdf,png}      (median PPL, threshold-free)
  scripts/_appfig_family_nativization_nonesm.{tsv,json} (numbers + tau definitions)
"""
from __future__ import annotations

import json
import re

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as c


# --- Constants --------------------------------------------------------------

CROSS = c.LAB_ROOT / "cross_architecture_nativeness/results"
FAMILY_META_TSV = (c.LAB_ROOT / "esm_viral_probe/datasets/human_virus/data/"
                   "controls/leave_family_out/family_metadata.tsv")

MIN_N_PER_FAMILY = 50
CELL_PCTL = 90          # cellular percentile defining the native-like threshold tau
N_SOLID = 3             # first 3 families drawn solid (rest dotted), as in Fig 2

# 8 families + colours inherited from the ESMC-6B ranking of main Fig 2, so the
# non-ESM panels are colour-comparable to the ESM figures.
FAMILIES_ESM_ORDER = [
    "Papillomaviridae", "Retroviridae", "Adenoviridae", "Poxviridae",
    "Coronaviridae", "Orthoherpesviridae", "Orthomyxoviridae", "Sedoreoviridae",
]
FAMILY_COLOR = {
    "Papillomaviridae":   "#7fc97f",
    "Retroviridae":       "#beaed4",
    "Adenoviridae":       "#fdc086",
    "Poxviridae":         "#00838f",
    "Coronaviridae":      "#386cb0",
    "Orthoherpesviridae": "#f0027f",
    "Orthomyxoviridae":   "#bf5b17",
    "Sedoreoviridae":     "#666666",
}

# Per-architecture scaling ladders. ref = matched-scale default already scored
# (used to fix tau). ProtT5 is single-scale -> reported in the sidecar only.
ARCHES: dict[str, dict] = {
    "ProGen2": dict(
        ref="progen2_base",
        models=[("progen2_small", "151M", int(151e6)),
                ("progen2_base", "764M", int(764e6)),
                ("progen2_large", "2.7B", int(2.7e9)),
                ("progen2_xlarge", "6.4B", int(6.4e9))],
    ),
    # EvoDiff scaling uses the *_elbo reruns (24-seed, training-faithful OA-ARDM
    # per-residue ELBO PPL; see cross_architecture_nativeness/scripts/score_evodiff.py).
    # _load_ppl aliases mean_perplexity <- mean_perplexity_elbo when that column exists,
    # so these panels read the ELBO estimate; the legacy pooled keys stay on disk.
    "EvoDiff": dict(
        ref="evodiff_oadm_640m_elbo",
        models=[("evodiff_oadm_38m_elbo", "38M", int(38e6)),
                ("evodiff_oadm_640m_elbo", "640M", int(640e6))],
    ),
    "ProtT5": dict(
        ref="prott5_xl",
        models=[("prott5_xl", "3B", int(3e9))],
    ),
}
SCALING_ARCHES = ["ProGen2", "EvoDiff"]   # >=2 scales -> get a panel/line


# --- Loaders ----------------------------------------------------------------

def _strip_sp(acc) -> str:
    if pd.isna(acc):
        return acc
    m = re.match(r"(?:sp|tr|ref)\|([^|]+)\|", str(acc))
    return m.group(1) if m else str(acc)


def _load_family_meta() -> pd.DataFrame:
    meta = pd.read_csv(FAMILY_META_TSV, sep="\t", usecols=["accession", "family"])
    meta["accession"] = meta["accession"].apply(_strip_sp)
    return meta.drop_duplicates("accession")


def _load_ppl(model_key: str) -> pd.DataFrame | None:
    """Per-sequence PPL for one cross-arch model (human pool: viral + nonviral)."""
    p = CROSS / model_key / "per_sequence_results.tsv"
    if not p.exists():
        return None
    df = pd.read_csv(p, sep="\t")
    # EvoDiff *_elbo reruns carry the training-faithful OA-ARDM ELBO PPL alongside the
    # legacy pooled one; prefer it so every downstream tau/median uses the ELBO estimate.
    if "mean_perplexity_elbo" in df.columns:
        df["mean_perplexity"] = df["mean_perplexity_elbo"]
    df["accession"] = df["accession"].apply(_strip_sp)
    return df


def _viral_with_family(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    v = df[df["label"] == "viral"].merge(meta, on="accession", how="left")
    v["family"] = v["family"].fillna("Unknown")
    return v


# --- Per-architecture computation -------------------------------------------

def compute_arch(arch_name: str, meta: pd.DataFrame) -> dict:
    """Return per-scale family fractions / medians + the fixed tau for one arch."""
    spec = ARCHES[arch_name]
    models = spec["models"]
    params = np.asarray([p for _, _, p in models], dtype=float)

    # tau = CELL_PCTL-th pctl of cellular PPL at the reference scale (fixed).
    ref_df = _load_ppl(spec["ref"])
    tau = float("nan")
    tau_alts: dict[str, float] = {}
    if ref_df is not None:
        cell_ref = ref_df.loc[ref_df["label"] == "nonviral", "mean_perplexity"].dropna()
        if len(cell_ref):
            tau = float(np.percentile(cell_ref, CELL_PCTL))
    # alternates (pooled across scales / largest scale) for the sidecar sanity check
    cell_pool, last_cell = [], None
    for mk, _, _ in models:
        d = _load_ppl(mk)
        if d is None:
            continue
        cv = d.loc[d["label"] == "nonviral", "mean_perplexity"].dropna().values
        cell_pool.append(cv)
        last_cell = cv
    if cell_pool:
        tau_alts["pooled_p90"] = float(np.percentile(np.concatenate(cell_pool), CELL_PCTL))
    if last_cell is not None and len(last_cell):
        tau_alts["largest_scale_p90"] = float(np.percentile(last_cell, CELL_PCTL))

    fam_pct = {f: [] for f in FAMILIES_ESM_ORDER}
    fam_med = {f: [] for f in FAMILIES_ESM_ORDER}
    fam_n = {f: [] for f in FAMILIES_ESM_ORDER}
    all_viral_pct, all_viral_med, cell_med = [], [], []

    for mk, _, _ in models:
        df = _load_ppl(mk)
        if df is None:
            for f in FAMILIES_ESM_ORDER:
                fam_pct[f].append(np.nan); fam_med[f].append(np.nan); fam_n[f].append(0)
            all_viral_pct.append(np.nan); all_viral_med.append(np.nan); cell_med.append(np.nan)
            continue
        v = _viral_with_family(df, meta)
        all_viral_pct.append(float(100.0 * (v["mean_perplexity"] < tau).mean())
                             if len(v) and np.isfinite(tau) else np.nan)
        all_viral_med.append(float(v["mean_perplexity"].median()) if len(v) else np.nan)
        cell = df.loc[df["label"] == "nonviral", "mean_perplexity"].dropna()
        cell_med.append(float(cell.median()) if len(cell) else np.nan)
        for f in FAMILIES_ESM_ORDER:
            sub = v[v["family"] == f]["mean_perplexity"].dropna()
            fam_n[f].append(int(len(sub)))
            fam_pct[f].append(float(100.0 * (sub < tau).mean())
                              if len(sub) and np.isfinite(tau) else np.nan)
            fam_med[f].append(float(sub.median()) if len(sub) else np.nan)

    return dict(
        models=models, params=params, tau=tau, tau_alts=tau_alts,
        fam_pct={f: np.asarray(v, float) for f, v in fam_pct.items()},
        fam_med={f: np.asarray(v, float) for f, v in fam_med.items()},
        fam_n=fam_n,
        all_viral_pct=np.asarray(all_viral_pct, float),
        all_viral_med=np.asarray(all_viral_med, float),
        cell_med=np.asarray(cell_med, float),
    )


# --- Plot helpers -----------------------------------------------------------

def _xaxis(ax, params: np.ndarray) -> None:
    ax.set_xscale("log")
    ax.set_xlim(params.min() * 0.6, params.max() * 1.8)
    ax.set_xticks(params)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(
        lambda v, _p: f"{v/1e9:g}B" if v >= 1e9 else f"{int(round(v/1e6))}M"))
    ax.minorticks_off()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _plot_families(ax, params, getter, solid, dotted, ref_y=None, ref_kw=None):
    for fam in dotted:
        ax.plot(params, getter[fam], color=FAMILY_COLOR[fam], lw=2.4, ls=":",
                marker="o", ms=8, markeredgecolor="#333333", markeredgewidth=1.0, zorder=4)
    for fam in solid:
        ax.plot(params, getter[fam], color=FAMILY_COLOR[fam], lw=3.6, ls="-",
                marker="o", ms=10, markeredgecolor="#333333", markeredgewidth=1.2, zorder=6)
    if ref_y is not None:
        ax.plot(params, ref_y, **(ref_kw or {}))


def _legend_handles(solid, dotted, extra):
    handles, labels = [], []
    for fam in solid:
        handles.append(mlines.Line2D([], [], color=FAMILY_COLOR[fam], marker="o", ms=9,
                                     lw=3.0, ls="-", markeredgecolor="#333333",
                                     markeredgewidth=1.2))
        labels.append(fam)
    for fam in dotted:
        handles.append(mlines.Line2D([], [], color=FAMILY_COLOR[fam], marker="o", ms=8,
                                     lw=2.0, ls=":", markeredgecolor="#333333",
                                     markeredgewidth=1.0))
        labels.append(fam)
    for h, l in extra:
        handles.append(h); labels.append(l)
    return handles, labels


def _serif():
    c.apply_style(font_size=20, axes_linewidth=2.2)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "svg.fonttype": "path",
        "axes.labelsize": 22, "xtick.labelsize": 18, "ytick.labelsize": 18,
    })


# --- Figures ----------------------------------------------------------------

def fig_threshold(results: dict) -> None:
    _serif()
    solid = FAMILIES_ESM_ORDER[:N_SOLID]
    dotted = FAMILIES_ESM_ORDER[N_SOLID:]
    n = len(SCALING_ARCHES)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 6.0), sharey=True)
    if n == 1:
        axes = [axes]
    y_max = 0.0
    for ax, arch in zip(axes, SCALING_ARCHES):
        R = results[arch]
        _plot_families(ax, R["params"], R["fam_pct"], solid, dotted,
                       ref_y=R["all_viral_pct"],
                       ref_kw=dict(color="#000000", lw=2.2, ls="--", marker="s", ms=7, zorder=5))
        _xaxis(ax, R["params"])
        ax.set_xlabel(f"{arch} parameters", labelpad=6)
        ax.set_title(f"{arch}  (τ = {R['tau']:.1f})", fontsize=18, pad=8)
        vals = [v for f in FAMILIES_ESM_ORDER for v in R["fam_pct"][f] if np.isfinite(v)]
        if vals:
            y_max = max(y_max, max(vals))
    axes[0].set_ylabel(r"% of family with PPL $<\ \tau$", labelpad=6)
    for ax in axes:
        ax.set_ylim(-2, max(40.0, 1.15 * y_max))
    h, l = _legend_handles(solid, dotted,
                           [(mlines.Line2D([], [], color="#000000", marker="s", ms=8, lw=2.2,
                                           ls="--"), "All viral")])
    axes[-1].legend(h, l, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                    fontsize=12, handlelength=2.0, handletextpad=0.6,
                    borderaxespad=0.0, labelspacing=0.4)
    fig.subplots_adjust(left=0.075, right=0.80, top=0.92, bottom=0.13, wspace=0.08)
    c.finalize(fig, "appfig_family_nativization_nonesm")
    print("Saved figures/appfig_family_nativization_nonesm.{pdf,png}")


def fig_medianppl(results: dict) -> None:
    _serif()
    solid = FAMILIES_ESM_ORDER[:N_SOLID]
    dotted = FAMILIES_ESM_ORDER[N_SOLID:]
    n = len(SCALING_ARCHES)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 6.0), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, arch in zip(axes, SCALING_ARCHES):
        R = results[arch]
        _plot_families(ax, R["params"], R["fam_med"], solid, dotted)
        # cellular reference (grey solid) + all-viral median (black dashed) + tau line
        ax.plot(R["params"], R["cell_med"], color="#888888", lw=2.4, ls="-",
                marker="D", ms=6, zorder=3)
        ax.plot(R["params"], R["all_viral_med"], color="#000000", lw=2.2, ls="--",
                marker="s", ms=7, zorder=5)
        if np.isfinite(R["tau"]):
            ax.axhline(R["tau"], color="#bbbbbb", lw=1.4, ls=":", zorder=1)
        _xaxis(ax, R["params"])
        ax.set_xlabel(f"{arch} parameters", labelpad=6)
        ax.set_title(arch, fontsize=18, pad=8)
        ax.set_ylabel("median masked-recon. PPL" if ax is axes[0] else "", labelpad=6)
    h, l = _legend_handles(solid, dotted, [
        (mlines.Line2D([], [], color="#888888", marker="D", ms=7, lw=2.4, ls="-"), "Cellular (ref.)"),
        (mlines.Line2D([], [], color="#000000", marker="s", ms=8, lw=2.2, ls="--"), "All viral"),
        (mlines.Line2D([], [], color="#bbbbbb", lw=1.6, ls=":"), r"$\tau$"),
    ])
    # Legend OUTSIDE to the right so it never overlaps the (clustered) family lines.
    axes[-1].legend(h, l, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                    fontsize=12, handlelength=2.0, handletextpad=0.6,
                    borderaxespad=0.0, labelspacing=0.4)
    fig.subplots_adjust(left=0.075, right=0.80, top=0.92, bottom=0.13, wspace=0.18)
    c.finalize(fig, "appfig_family_medianppl_nonesm")
    print("Saved figures/appfig_family_medianppl_nonesm.{pdf,png}")


# --- Sidecar ----------------------------------------------------------------

def dump_sidecar(results: dict) -> None:
    rows = []
    meta_json: dict = {"cell_pctl": CELL_PCTL, "min_n_per_family": MIN_N_PER_FAMILY,
                       "families": FAMILIES_ESM_ORDER, "arches": {}}
    for arch, R in results.items():
        meta_json["arches"][arch] = {
            "tau_ref_scale_p90": R["tau"], "tau_alts": R["tau_alts"],
            "ref_model": ARCHES[arch]["ref"],
            "models": [mk for mk, _, _ in R["models"]],
        }
        for j, (mk, lbl, prm) in enumerate(R["models"]):
            for f in FAMILIES_ESM_ORDER:
                rows.append(dict(architecture=arch, model=mk, label=lbl, params=prm,
                                 family=f, n=R["fam_n"][f][j],
                                 pct_below_tau=R["fam_pct"][f][j],
                                 median_ppl=R["fam_med"][f][j], tau=R["tau"]))
            rows.append(dict(architecture=arch, model=mk, label=lbl, params=prm,
                             family="ALL_VIRAL", n=-1,
                             pct_below_tau=R["all_viral_pct"][j],
                             median_ppl=R["all_viral_med"][j], tau=R["tau"]))
            rows.append(dict(architecture=arch, model=mk, label=lbl, params=prm,
                             family="CELLULAR_REF", n=-1, pct_below_tau=np.nan,
                             median_ppl=R["cell_med"][j], tau=R["tau"]))
    df = pd.DataFrame(rows)
    df.to_csv(c.HERE / "_appfig_family_nativization_nonesm.tsv", sep="\t", index=False)
    with open(c.HERE / "_appfig_family_nativization_nonesm.json", "w") as fh:
        json.dump(meta_json, fh, indent=2)
    print("Saved scripts/_appfig_family_nativization_nonesm.{tsv,json}")


def main() -> None:
    meta = _load_family_meta()
    results = {arch: compute_arch(arch, meta) for arch in ARCHES}
    # console summary
    for arch in ARCHES:
        R = results[arch]
        avail = sum(_load_ppl(mk) is not None for mk, _, _ in R["models"])
        print(f"{arch}: tau={R['tau']:.3f}  scales available {avail}/{len(R['models'])}  "
              f"alts={R['tau_alts']}")
    fig_threshold(results)
    fig_medianppl(results)
    dump_sidecar(results)


if __name__ == "__main__":
    main()
