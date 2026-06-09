#!/usr/bin/env python3
"""
evaluate.py
===========
Evaluate trained probes on the held-out test set.

Computes and saves:
  - AUC-ROC, AUC-PR, Accuracy, F1, MCC (Matthews Correlation Coefficient)
  - ROC curve data (fpr, tpr, thresholds)
  - PR curve data (precision, recall, thresholds)
  - Confusion matrix
  - Per-probe test predictions (for downstream analysis)

Outputs (in --outdir):
  - test_results.json          all scalar metrics
  - test_preds_linear.npz      accessions, labels, scores

Usage:
  python evaluate.py \
      --viral_test    ../data/embeddings/viral_test.npz \
      --nonviral_test ../data/embeddings/nonviral_test.npz \
      --results_dir   ../results \
      --device        cuda
"""

import argparse
import json
import os
import pickle

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from train_probe import load_embeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_test_set(viral_path: str, nonviral_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_viral,    acc_viral    = load_embeddings(viral_path)
    X_nonviral, acc_nonviral = load_embeddings(nonviral_path)
    X   = np.vstack([X_viral, X_nonviral])
    y   = np.concatenate([np.ones(len(X_viral)), np.zeros(len(X_nonviral))])
    acc = np.concatenate([acc_viral, acc_nonviral])
    print(f"  Viral:     {len(X_viral):,}")
    print(f"  Non-viral: {len(X_nonviral):,}")
    return X, y, acc


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute a comprehensive set of binary classification metrics."""
    y_pred = (y_score >= threshold).astype(int)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    prec, rec, pr_thresholds = precision_recall_curve(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "auc_roc":  float(roc_auc_score(y_true, y_score)),
        "auc_pr":   float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc":      float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "pr_curve": {
            "precision": prec.tolist(), "recall": rec.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--viral_test",    required=True)
    parser.add_argument("--nonviral_test", required=True)
    parser.add_argument("--results_dir",   required=True,
                        help="Directory containing trained probes + scaler")
    args = parser.parse_args()

    print("=" * 60)
    print("ESM2 Viral Probe — Evaluation (Test Set)")
    print("=" * 60)

    # ---- Load test set ----
    print("\n[Test set]")
    X_test, y_test, acc_test = build_test_set(args.viral_test, args.nonviral_test)

    # ---- Load scaler ----
    scaler_path = os.path.join(args.results_dir, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)
    X_test_sc = scaler.transform(X_test)

    all_results = {}

    # ----------------------------------------------------------------
    # 1. Linear Probe
    # ----------------------------------------------------------------
    print("\n[Linear Probe]")
    linear_path = os.path.join(args.results_dir, "linear_probe.pkl")
    with open(linear_path, "rb") as f:
        linear = pickle.load(f)

    scores_lin = linear.predict_proba(X_test_sc)[:, 1]
    metrics_lin = compute_metrics(y_test, scores_lin)
    all_results["linear"] = metrics_lin

    print(f"  AUC-ROC:  {metrics_lin['auc_roc']:.4f}")
    print(f"  AUC-PR:   {metrics_lin['auc_pr']:.4f}")
    print(f"  Accuracy: {metrics_lin['accuracy']:.4f}")
    print(f"  F1:       {metrics_lin['f1']:.4f}")
    print(f"  MCC:      {metrics_lin['mcc']:.4f}")
    print(f"  Confusion matrix: {metrics_lin['confusion_matrix']}")

    np.savez_compressed(
        os.path.join(args.results_dir, "test_preds_linear.npz"),
        accessions=acc_test, labels=y_test, scores=scores_lin,
    )

    # ---- Save all metrics ----
    out_json = os.path.join(args.results_dir, "test_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"{'Probe':<12}  {'AUC-ROC':>8}  {'AUC-PR':>8}  {'Acc':>8}  {'F1':>8}  {'MCC':>8}")
    print("-" * 60)
    print(f"{'Linear':<12}  {metrics_lin['auc_roc']:>8.4f}  {metrics_lin['auc_pr']:>8.4f}  "
          f"{metrics_lin['accuracy']:>8.4f}  {metrics_lin['f1']:>8.4f}  {metrics_lin['mcc']:>8.4f}")


if __name__ == "__main__":
    main()
