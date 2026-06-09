#!/usr/bin/env python3
"""
train_probe.py
==============
Train a linear probe (y = Wh + b) on frozen ESM2 embeddings to classify
viral vs. non-viral sequences.

Outputs (in --outdir):
  - linear_probe.pkl       scikit-learn model
  - scaler.pkl             fitted StandardScaler
  - train_metrics.json     val AUC / acc
  - probe_config.json      hyperparameters used

Usage:
  python train_probe.py \
      --viral_train    ../data/embeddings/viral_train.npz \
      --nonviral_train ../data/embeddings/nonviral_train.npz \
      --viral_val      ../data/embeddings/viral_val.npz \
      --nonviral_val   ../data/embeddings/nonviral_val.npz \
      --outdir         ../results
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and accessions from a .npz file."""
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["accessions"]


def build_dataset(viral_path: str, nonviral_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load viral (label=1) and non-viral (label=0) embeddings.
    Returns (X, y, accessions).
    """
    X_viral,    acc_viral    = load_embeddings(viral_path)
    X_nonviral, acc_nonviral = load_embeddings(nonviral_path)

    X   = np.vstack([X_viral, X_nonviral])
    y   = np.concatenate([np.ones(len(X_viral)), np.zeros(len(X_nonviral))])
    acc = np.concatenate([acc_viral, acc_nonviral])

    print(f"  Viral:     {len(X_viral):>8,}  sequences")
    print(f"  Non-viral: {len(X_nonviral):>8,}  sequences")
    print(f"  Embedding dim: {X.shape[1]}")
    return X, y, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--viral_train",    required=True)
    parser.add_argument("--nonviral_train", required=True)
    parser.add_argument("--viral_val",      required=True)
    parser.add_argument("--nonviral_val",   required=True)
    parser.add_argument("--outdir",         required=True)
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse regularization strength for logistic regression (default: 1.0)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("ESM2 Viral Probe — Training")
    print("=" * 60)

    # ---- Load data ----
    print("\n[Train set]")
    X_train, y_train, _ = build_dataset(args.viral_train, args.nonviral_train)
    print("\n[Validation set]")
    X_val,   y_val,   _ = build_dataset(args.viral_val,   args.nonviral_val)

    # ---- Normalize features ----
    print("\nFitting StandardScaler on train embeddings...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    # Save scaler for evaluation
    with open(os.path.join(args.outdir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # ----------------------------------------------------------------
    # 1. Linear Probe (logistic regression)
    # ----------------------------------------------------------------
    print("\n" + "=" * 40)
    print("1. Linear Probe (Logistic Regression)")
    print("=" * 40)
    t0 = time.time()

    linear = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",   # handles class imbalance
        C=args.C,
        n_jobs=-1,
        verbose=1,
    )
    linear.fit(X_train_sc, y_train)

    val_probs_lin = linear.predict_proba(X_val_sc)[:, 1]
    val_auc_lin   = roc_auc_score(y_val, val_probs_lin)
    val_acc_lin   = accuracy_score(y_val, linear.predict(X_val_sc))
    print(f"\n  Val AUC: {val_auc_lin:.4f}  |  Val Acc: {val_acc_lin:.4f}")
    print(f"  Time:    {time.time()-t0:.1f}s")

    with open(os.path.join(args.outdir, "linear_probe.pkl"), "wb") as f:
        pickle.dump(linear, f)

    # ---- Save outputs ----
    with open(os.path.join(args.outdir, "train_metrics.json"), "w") as f:
        json.dump({"val_auc": val_auc_lin, "val_acc": val_acc_lin}, f, indent=2)

    with open(os.path.join(args.outdir, "probe_config.json"), "w") as f:
        json.dump({"C": args.C}, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Linear Probe — AUC: {val_auc_lin:.4f}  Acc: {val_acc_lin:.4f}")
    print(f"\nOutputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
