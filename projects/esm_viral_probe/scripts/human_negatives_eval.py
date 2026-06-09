#!/usr/bin/env python3
"""
human_negatives_eval.py
=======================
Re-evaluate trained probes using ONLY Homo sapiens (OX=9606) proteins as the
negative class. This is the hardest possible test: can the probe distinguish
proteins from human-infecting viruses from proteins of the human host itself?

If the model is exploiting easy inter-kingdom differences (viral vs bacteria/yeast),
this control will expose it: AUC will drop dramatically.

Algorithm:
  1. Parse data/nonviral/uniprot_nonviral.faa to identify Homo sapiens accessions
     (SwissProt headers contain "OX=9606")
  2. Filter test_preds_linear.npz non-viral entries to human-only
  3. Re-compute AUC vs the same viral test set
  4. Save human_neg_results.json per model

Usage:
  python human_negatives_eval.py \\
      --project_dir    datasets/human_virus \\
      --nonviral_fasta data/nonviral/uniprot_nonviral.faa
"""

import argparse
import json
import os
import re

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve,
)


# ---------------------------------------------------------------------------
# Parse SwissProt headers for organism taxon
# ---------------------------------------------------------------------------

def build_human_accession_set(fasta_path: str) -> set[str]:
    """
    Parse a SwissProt FASTA and return the set of accessions where OX=9606
    (Homo sapiens).

    SwissProt header format:
      >sp|P01308|INS_HUMAN Insulin OS=Homo sapiens OX=9606 GN=INS PE=1 SV=1
    The accession returned is the first whitespace-delimited field: sp|P01308|INS_HUMAN
    """
    ox_pattern = re.compile(r"\bOX=(\d+)\b")
    human_accs: set[str] = set()

    with open(fasta_path) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].rstrip()
            acc = header.split()[0]
            m = ox_pattern.search(header)
            if m and m.group(1) == "9606":
                human_accs.add(acc)

    return human_accs


# ---------------------------------------------------------------------------
# Metrics (same schema as evaluate.py)
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    fpr, tpr, roc_t = roc_curve(y_true, y_score)
    prec, rec, pr_t  = precision_recall_curve(y_true, y_score)
    return {
        "auc_roc":          float(roc_auc_score(y_true, y_score)),
        "auc_pr":           float(average_precision_score(y_true, y_score)),
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc":              float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_t.tolist()
        },
        "pr_curve": {
            "precision": prec.tolist(), "recall": rec.tolist(), "thresholds": pr_t.tolist()
        },
    }


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def run_model(model_key: str, project_dir: str, human_accs: set[str]) -> dict | None:
    results_dir = os.path.join(project_dir, "results", model_key)
    preds_path  = os.path.join(results_dir, "test_preds_linear.npz")

    if not os.path.exists(preds_path):
        print(f"  [{model_key}] SKIP: {preds_path} not found")
        return None

    data       = np.load(preds_path, allow_pickle=True)
    accessions = np.array([str(a) for a in data["accessions"]])
    labels     = data["labels"]
    scores     = data["scores"]

    # Viral indices (keep all viral)
    viral_mask    = labels == 1
    # Non-viral indices filtered to human-only
    nonviral_mask = (labels == 0) & np.array([a in human_accs for a in accessions])

    n_viral    = int(viral_mask.sum())
    n_nonviral = int(nonviral_mask.sum())
    n_nonviral_total = int((labels == 0).sum())

    print(f"  [{model_key}] viral={n_viral:,}  human_nonviral={n_nonviral:,} "
          f"({n_nonviral/n_nonviral_total:.1%} of {n_nonviral_total:,} non-viral)")

    if n_nonviral < 10:
        print(f"  [{model_key}] WARNING: only {n_nonviral} human non-viral sequences found, "
              "skipping (check --nonviral_fasta path)")
        return None

    combined_mask = viral_mask | nonviral_mask
    y_sub = labels[combined_mask]
    s_sub = scores[combined_mask]

    if len(np.unique(y_sub)) < 2:
        print(f"  [{model_key}] WARNING: only one class in human-negatives subset, skipping")
        return None

    metrics = compute_metrics(y_sub, s_sub)
    print(f"  [{model_key}] AUC-ROC: {metrics['auc_roc']:.4f}  "
          f"F1: {metrics['f1']:.4f}  MCC: {metrics['mcc']:.4f}")

    result = {
        "n_viral":              n_viral,
        "n_human_nonviral":     n_nonviral,
        "n_nonviral_total":     n_nonviral_total,
        "frac_human":           float(n_nonviral / n_nonviral_total),
        "metrics": metrics,
    }

    # Load full-test AUC for delta comparison
    full_path = os.path.join(results_dir, "test_results.json")
    if os.path.exists(full_path):
        with open(full_path) as f:
            full_res = json.load(f)
        result["full_test_auc"] = full_res.get("linear", {}).get("auc_roc")

    out_path = os.path.join(results_dir, "human_neg_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [{model_key}] Saved to {out_path}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project_dir",    required=True,
                        help="Experiment root (e.g. datasets/human_virus)")
    parser.add_argument("--nonviral_fasta", required=True,
                        help="Full SwissProt non-viral FASTA (data/nonviral/uniprot_nonviral.faa)")
    args = parser.parse_args()

    print("=" * 60)
    print("Human-Only Negatives Evaluation")
    print(f"  project_dir:    {args.project_dir}")
    print(f"  nonviral_fasta: {args.nonviral_fasta}")
    print("=" * 60)

    print("\nParsing SwissProt FASTA for OX=9606 (Homo sapiens) accessions...")
    human_accs = build_human_accession_set(args.nonviral_fasta)
    print(f"  Found {len(human_accs):,} Homo sapiens accessions in the full non-viral pool")

    results_root = os.path.join(args.project_dir, "results")
    model_keys = sorted(
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d)) and d != "baseline"
    )
    print(f"\nFound {len(model_keys)} model directories: {model_keys}\n")

    summary = {}
    for model_key in model_keys:
        result = run_model(model_key, args.project_dir, human_accs)
        if result is not None:
            summary[model_key] = {
                "full_auc":   result.get("full_test_auc"),
                "human_auc":  result["metrics"]["auc_roc"],
                "n_human":    result["n_human_nonviral"],
                "frac_human": result["frac_human"],
            }

    print("\n" + "=" * 60)
    print("Summary — Human-Only Negatives")
    print("=" * 60)
    print(f"  {'Model':<20}  {'Full AUC':>9}  {'Human AUC':>10}  {'Delta':>7}  {'N human':>8}")
    print("  " + "-" * 65)
    for mk, r in summary.items():
        full  = r["full_auc"] if r["full_auc"] is not None else float("nan")
        delta = r["human_auc"] - full if r["full_auc"] is not None else float("nan")
        print(f"  {mk:<20}  {full:>9.4f}  {r['human_auc']:>10.4f}  "
              f"{delta:>+7.4f}  {r['n_human']:>8,}")

    out = os.path.join(args.project_dir, "results", "human_neg_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
