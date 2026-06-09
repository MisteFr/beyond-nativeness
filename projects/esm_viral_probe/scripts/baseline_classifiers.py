#!/usr/bin/env python3
"""
baseline_classifiers.py
=======================
Train sequence-statistics classifiers (length, AA composition, combined)
as ablation baselines to rule out trivial confounds (Possibility B).

Uses the same train/val/test splits as the embedding probes and outputs
results in the same JSON schema as evaluate.py for direct comparison.

Feature sets:
  length_only              — 1-D: [sequence_length]
  aa_composition           — 20-D: per-residue frequency for each standard AA
  length_plus_composition  — 21-D: length + composition combined
  dipeptide_composition    — 400-D: frequency of each ordered AA pair (AA,AA) at adjacent positions

Usage:
  python baseline_classifiers.py \\
      --data_dir  datasets/human_virus/data/processed \\
      --outdir    datasets/human_virus/results/baseline
      [--feature_sets length_only,aa_composition,length_plus_composition,dipeptide_composition]
"""

import argparse
import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
# Ordered AA pairs: (AA,AA) in row-major order of STANDARD_AA
DIPEP_PAIRS = [a + b for a in STANDARD_AA for b in STANDARD_AA]
DIPEP_IDX   = {pair: i for i, pair in enumerate(DIPEP_PAIRS)}


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

def parse_fasta(path: str):
    """Yield (accession, sequence) pairs from a FASTA file."""
    header, parts = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header.split()[0], "".join(parts)
                header, parts = line[1:], []
            else:
                parts.append(line.upper())
    if header is not None:
        yield header.split()[0], "".join(parts)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _dipeptide_vector(seq: str) -> np.ndarray:
    """Return a 400-D frequency vector of ordered adjacent AA pairs.
    Pairs containing a non-standard AA are ignored; the vector is normalized
    by the count of valid pairs (not n-1), so it's a proper frequency
    over observed dipeptides. All-zero vector if no valid pairs."""
    vec = np.zeros(400, dtype=np.float64)
    if len(seq) < 2:
        return vec
    valid = 0
    for i in range(len(seq) - 1):
        idx = DIPEP_IDX.get(seq[i:i + 2])
        if idx is not None:
            vec[idx] += 1.0
            valid += 1
    if valid > 0:
        vec /= valid
    return vec


def extract_features(records: list[tuple[str, str]], feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X [N, D], accessions [N]) for the chosen feature set."""
    accs, rows = [], []
    for acc, seq in records:
        n = len(seq)
        if feature_set == "length_only":
            rows.append([float(n)])
        elif feature_set == "aa_composition":
            rows.append([seq.count(aa) / n for aa in STANDARD_AA])
        elif feature_set == "length_plus_composition":
            rows.append([float(n)] + [seq.count(aa) / n for aa in STANDARD_AA])
        elif feature_set == "dipeptide_composition":
            rows.append(_dipeptide_vector(seq).tolist())
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")
        accs.append(acc)
    return np.array(rows, dtype=np.float64), np.array(accs)


def load_split(
    data_dir: str, split: str, feature_set: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (X, y, accessions) for viral (y=1) + nonviral (y=0) in one split."""
    viral_records    = list(parse_fasta(os.path.join(data_dir, f"viral_{split}.faa")))
    nonviral_records = list(parse_fasta(os.path.join(data_dir, f"nonviral_{split}.faa")))
    X_v, acc_v = extract_features(viral_records,    feature_set)
    X_n, acc_n = extract_features(nonviral_records, feature_set)
    X   = np.vstack([X_v, X_n])
    y   = np.concatenate([np.ones(len(X_v)), np.zeros(len(X_n))])
    acc = np.concatenate([acc_v, acc_n])
    print(f"    viral={len(X_v):,}  nonviral={len(X_n):,}")
    return X, y, acc


# ---------------------------------------------------------------------------
# Metrics (same schema as evaluate.py)
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    fpr, tpr, roc_t = roc_curve(y_true, y_score)
    prec, rec, pr_t = precision_recall_curve(y_true, y_score)
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
# Main
# ---------------------------------------------------------------------------

FEATURE_SETS = [
    "length_only",
    "aa_composition",
    "length_plus_composition",
    "dipeptide_composition",
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing processed {viral,nonviral}_{train,val,test}.faa")
    parser.add_argument("--outdir",   required=True,
                        help="Base output directory; one subdirectory per feature set")
    parser.add_argument("--feature_sets", default=None,
                        help=f"Comma-separated subset of {FEATURE_SETS}. "
                             f"Defaults to all.")
    args = parser.parse_args()

    if args.feature_sets:
        requested = [s.strip() for s in args.feature_sets.split(",") if s.strip()]
        unknown   = [s for s in requested if s not in FEATURE_SETS]
        if unknown:
            raise ValueError(f"Unknown feature sets: {unknown}. "
                             f"Known: {FEATURE_SETS}")
        feature_sets = requested
    else:
        feature_sets = FEATURE_SETS

    print("=" * 60)
    print("Sequence-Statistics Baseline Classifiers")
    print("=" * 60)

    summary: dict[str, dict] = {}

    summary_path = os.path.join(args.outdir, "summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except json.JSONDecodeError:
            summary = {}

    for fs in feature_sets:
        print(f"\n{'='*40}")
        print(f"Feature set: {fs}")
        print(f"{'='*40}")

        out_dir = os.path.join(args.outdir, fs)
        os.makedirs(out_dir, exist_ok=True)

        print("  Loading train split...")
        X_train, y_train, _ = load_split(args.data_dir, "train", fs)
        print("  Loading val split...")
        X_val,   y_val,   _ = load_split(args.data_dir, "val",   fs)
        print("  Loading test split...")
        X_test,  y_test,  acc_test = load_split(args.data_dir, "test", fs)
        print(f"  Feature dim: {X_train.shape[1]}")

        # Scale
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc   = scaler.transform(X_val)
        X_test_sc  = scaler.transform(X_test)

        # Train
        clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000,
            solver="lbfgs", n_jobs=-1,
        )
        clf.fit(X_train_sc, y_train)

        # Validate
        val_probs = clf.predict_proba(X_val_sc)[:, 1]
        val_auc   = roc_auc_score(y_val, val_probs)
        print(f"  Val AUC: {val_auc:.4f}")

        # Test
        test_scores = clf.predict_proba(X_test_sc)[:, 1]
        metrics     = compute_metrics(y_test, test_scores)
        print(f"  Test AUC-ROC: {metrics['auc_roc']:.4f}  "
              f"AUC-PR: {metrics['auc_pr']:.4f}  "
              f"F1: {metrics['f1']:.4f}  MCC: {metrics['mcc']:.4f}")

        # Save
        with open(os.path.join(out_dir, "test_results.json"), "w") as f:
            json.dump({"linear": metrics}, f, indent=2)
        with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(out_dir, "linear_probe.pkl"), "wb") as f:
            pickle.dump(clf, f)
        with open(os.path.join(out_dir, "train_metrics.json"), "w") as f:
            json.dump({"val_auc": val_auc, "feature_set": fs}, f, indent=2)

        summary[fs] = {
            "auc_roc": metrics["auc_roc"],
            "auc_pr":  metrics["auc_pr"],
            "f1":      metrics["f1"],
            "mcc":     metrics["mcc"],
        }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  {'Feature set':<30}  {'AUC-ROC':>8}  {'AUC-PR':>8}  {'F1':>8}  {'MCC':>8}")
    print("  " + "-" * 62)
    for fs, m in summary.items():
        print(f"  {fs:<30}  {m['auc_roc']:>8.4f}  {m['auc_pr']:>8.4f}  "
              f"{m['f1']:>8.4f}  {m['mcc']:>8.4f}")

    # Write combined summary JSON
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
