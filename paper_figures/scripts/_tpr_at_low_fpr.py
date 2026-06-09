"""Compute TPR at low FPR for the embedding probe and PPL zero-shot, all models.

For a probe with AUC near ceiling, TPR @ FPR=1e-3 / 1e-2 has more dynamic range
than AUC and is the operating point that matters for "viral detector" framing.

Outputs
-------
- _tpr_at_low_fpr.tsv  : long-form table (model, modality, fpr, tpr, ...)
- prints a wide table to stdout for quick reading
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import _common as c


FPRS = (1e-3, 1e-2, 5e-2, 1e-1)
N_BOOT = 2000
RNG = np.random.default_rng(20260427)


def tpr_at_fpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float) -> float:
    """Largest TPR at FPR <= target_fpr along the empirical ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, score)
    mask = fpr <= target_fpr + 1e-12
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def bootstrap_tpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float,
                  n_boot: int = N_BOOT) -> tuple[float, float]:
    n = len(y_true)
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        # Skip degenerate draws with no positives or no negatives.
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(idx):
            out[i] = np.nan
            continue
        out[i] = tpr_at_fpr(y_true[idx], score[idx], target_fpr)
    out = out[~np.isnan(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def collect():
    rows = []
    for fam, lst in c.MODEL_FAMILIES.items():
        for mk, label, params in lst:
            # --- Embedding probe ---
            d = c.probe_preds(mk)
            y, s = d["labels"], d["scores"]
            for f in FPRS:
                tpr = tpr_at_fpr(y, s, f)
                lo, hi = bootstrap_tpr(y, s, f)
                rows.append(dict(family=fam, model=mk, label=label, params=params,
                                 modality="probe", fpr=f, tpr=tpr,
                                 tpr_lo=lo, tpr_hi=hi))

            # --- PPL zero-shot (lower PPL = more native = nonviral; flip sign
            # so higher score = more "viral"). per_seq_ppl is keyed by accession;
            # use the same accession ordering as probe_preds for an apples-to-apples
            # comparison on the test set. ---
            try:
                ppl_df = c.per_seq_ppl(mk).set_index("accession")
            except FileNotFoundError:
                continue
            keep = [a for a in d["accessions"] if a in ppl_df.index]
            if len(keep) < 0.5 * len(d["accessions"]):
                # Coverage too poor; skip rather than mislead.
                continue
            sub = ppl_df.loc[keep]
            label_raw = sub["label"].astype(str).to_numpy()
            if set(label_raw) <= {"viral", "nonviral"}:
                y_p = (label_raw == "viral").astype(int)
            else:
                y_p = sub["label"].astype(int).to_numpy()
            ppl = sub["mean_perplexity"].astype(float).to_numpy()
            score_ppl = ppl  # higher PPL = more viral in this dataset
            for f in FPRS:
                tpr = tpr_at_fpr(y_p, score_ppl, f)
                lo, hi = bootstrap_tpr(y_p, score_ppl, f)
                rows.append(dict(family=fam, model=mk, label=label, params=params,
                                 modality="ppl_zeroshot", fpr=f, tpr=tpr,
                                 tpr_lo=lo, tpr_hi=hi))
    return pd.DataFrame(rows)


def main():
    df = collect()
    out = c.HERE / "_tpr_at_low_fpr.tsv"
    df.to_csv(out, sep="\t", index=False, float_format="%.4f")
    print(f"Wrote {out}")

    # Wide table: rows = model, columns = (modality, fpr).
    pivot = (df.pivot_table(index=["family", "label", "model", "params"],
                            columns=["modality", "fpr"],
                            values="tpr")
               .sort_index(level=["family", "params"]))
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(pivot)


if __name__ == "__main__":
    main()
