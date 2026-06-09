#!/usr/bin/env python3
"""
eval_leave_family_out.py
========================
Leave-family-out viral evaluation (Control 6).

For each viral family with enough sequences, train a probe on all OTHER
viral families (using existing precomputed embeddings) and evaluate on
the held-out family.  Non-viral data is unchanged.

Tests whether the probe generalizes to unseen viral families, or only
memorizes family-level patterns.

Output per model:
  datasets/human_virus/results/{model}/leave_family_out_results.json

Summary:
  datasets/human_virus/results/leave_family_out_summary.json

Usage:
  python eval_leave_family_out.py \\
      --project_dir  datasets/human_virus \\
      --family_meta  datasets/human_virus/data/controls/leave_family_out/family_metadata.tsv \\
      [--min_family_size 50] \\
      [--threshold 0.5]
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


MODEL_KEYS = [
    "esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m",
    "esmc_300m", "esmc_600m", "esmc_6b",
    "esm3_small", "esm3_open", "esm3_medium", "esm3_large",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embeddings(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], np.array([str(a) for a in data["accessions"]])


def load_family_meta(tsv_path: str) -> dict[str, str]:
    """Return {accession: family}."""
    meta: dict[str, str] = {}
    with open(tsv_path) as fh:
        next(fh)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                meta[parts[0]] = parts[1]
    return meta


def load_split_meta(tsv_path: str) -> dict[str, str]:
    """Return {accession: split} for viral sequences only."""
    acc_to_split: dict[str, str] = {}
    with open(tsv_path) as fh:
        next(fh)  # header: accession, label, split
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[1] == "viral":
                acc_to_split[parts[0]] = parts[2]
    return acc_to_split


def train_probe(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                C: float = 1.0) -> tuple:
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_vl_sc = scaler.transform(X_val)

    probe = LogisticRegression(
        max_iter=1000, solver="lbfgs",
        class_weight="balanced", C=C, n_jobs=-1,
    )
    probe.fit(X_tr_sc, y_train)

    val_auc = roc_auc_score(y_val, probe.predict_proba(X_vl_sc)[:, 1])
    return probe, scaler, val_auc


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                    threshold: float) -> dict:
    auc_roc = float(roc_auc_score(y_true, y_score))
    auc_pr  = float(average_precision_score(y_true, y_score))
    n_viral = int(y_true.sum())
    n_pos   = int(((y_score >= threshold) & (y_true == 1)).sum())
    sensitivity = float(n_pos / n_viral) if n_viral > 0 else 0.0
    return {
        "auc_roc":     auc_roc,
        "auc_pr":      auc_pr,
        "sensitivity": sensitivity,
    }


# ---------------------------------------------------------------------------
# Per-model leave-family-out evaluation
# ---------------------------------------------------------------------------

def run_model(
    model_key: str,
    project_dir: str,
    acc_to_family: dict[str, str],
    acc_to_split: dict[str, str],
    families: list[str],
    min_family_size: int,
    threshold: float,
) -> dict | None:
    emb_dir = os.path.join(project_dir, "data", "embeddings", model_key)

    # Load all viral embeddings (train + val + test)
    viral_parts: list[tuple[np.ndarray, np.ndarray, str]] = []
    for split in ("train", "val", "test"):
        p = os.path.join(emb_dir, f"viral_{split}.npz")
        if not os.path.exists(p):
            print(f"  [{model_key}] SKIP: missing {p}")
            return None
        embs, accs = load_embeddings(p)
        for i, acc in enumerate(accs):
            viral_parts.append((embs[i], acc, split))

    # Also need nonviral embeddings
    nv_train_path = os.path.join(emb_dir, "nonviral_train.npz")
    nv_val_path   = os.path.join(emb_dir, "nonviral_val.npz")
    nv_test_path  = os.path.join(emb_dir, "nonviral_test.npz")
    for p in [nv_train_path, nv_val_path, nv_test_path]:
        if not os.path.exists(p):
            print(f"  [{model_key}] SKIP: missing {p}")
            return None

    X_nv_train, _ = load_embeddings(nv_train_path)
    X_nv_val,   _ = load_embeddings(nv_val_path)
    X_nv_test,  _ = load_embeddings(nv_test_path)

    # Build per-family index
    from collections import defaultdict
    family_by_split: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    emb_matrix = np.vstack([e for e, _, _ in viral_parts])
    for idx, (_, acc, split) in enumerate(viral_parts):
        fam = acc_to_family.get(acc, "Unknown")
        # Prefer split from metadata (more reliable), fall back to npz split
        split_label = acc_to_split.get(acc, split)
        family_by_split[fam][split_label].append(idx)

    print(f"  [{model_key}] Total viral embeddings: {len(viral_parts):,}")

    model_results: dict[str, dict] = {}

    for family in families:
        fam_indices = []
        for s_indices in family_by_split[family].values():
            fam_indices.extend(s_indices)

        n_fam = len(fam_indices)
        if n_fam < min_family_size:
            print(f"  [{model_key}] {family}: {n_fam} seqs < {min_family_size}, skip")
            continue

        fam_set = set(fam_indices)

        # Training viral: NOT this family, from train split
        train_idx = [
            idx for idx, (_, acc, split) in enumerate(viral_parts)
            if idx not in fam_set
            and acc_to_split.get(acc, split) == "train"
        ]
        # Val viral: NOT this family, from val split
        val_idx = [
            idx for idx, (_, acc, split) in enumerate(viral_parts)
            if idx not in fam_set
            and acc_to_split.get(acc, split) == "val"
        ]

        if len(train_idx) < 10 or len(val_idx) < 5:
            print(f"  [{model_key}] {family}: too few training seqs, skip")
            continue

        X_v_train = emb_matrix[train_idx]
        X_v_val   = emb_matrix[val_idx]
        X_v_test  = emb_matrix[list(fam_set)]

        # Combine with nonviral
        X_train = np.vstack([X_v_train, X_nv_train])
        y_train = np.concatenate([np.ones(len(X_v_train)), np.zeros(len(X_nv_train))])
        X_val   = np.vstack([X_v_val, X_nv_val])
        y_val   = np.concatenate([np.ones(len(X_v_val)),   np.zeros(len(X_nv_val))])
        X_test  = np.vstack([X_v_test, X_nv_test])
        y_test  = np.concatenate([np.ones(len(X_v_test)),  np.zeros(len(X_nv_test))])

        t0 = time.time()
        probe, scaler, val_auc = train_probe(X_train, y_train, X_val, y_val)

        X_test_sc = scaler.transform(X_test)
        scores    = probe.predict_proba(X_test_sc)[:, 1]
        metrics   = compute_metrics(y_test, scores, threshold)

        elapsed = time.time() - t0
        print(
            f"  [{model_key}] {family:<35}  "
            f"n_train={len(train_idx):4d}  n_test={n_fam:4d}  "
            f"val_auc={val_auc:.3f}  "
            f"test_auc={metrics['auc_roc']:.3f}  "
            f"sens={metrics['sensitivity']:.3f}  "
            f"({elapsed:.1f}s)"
        )

        model_results[family] = {
            "n_train_viral": len(train_idx),
            "n_val_viral":   len(val_idx),
            "n_test_viral":  n_fam,
            "val_auc":       float(val_auc),
            **metrics,
        }

    return model_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project_dir",    required=True,
                        help="e.g. datasets/human_virus")
    parser.add_argument("--family_meta",    required=True,
                        help="TSV: accession → family (from prepare_family_splits.py)")
    parser.add_argument("--min_family_size", type=int, default=50,
                        help="Minimum viral sequences per family to run (default: 50)")
    parser.add_argument("--threshold",       type=float, default=0.5,
                        help="Score threshold for sensitivity (default: 0.5)")
    args = parser.parse_args()

    meta_tsv = os.path.join(args.project_dir, "data", "processed", "metadata.tsv")
    if not os.path.exists(meta_tsv):
        raise FileNotFoundError(f"metadata.tsv not found: {meta_tsv}")
    if not os.path.exists(args.family_meta):
        raise FileNotFoundError(f"family metadata not found: {args.family_meta}")

    print("=" * 70)
    print("Leave-Family-Out Viral Evaluation")
    print(f"  project_dir:      {args.project_dir}")
    print(f"  family_meta:      {args.family_meta}")
    print(f"  min_family_size:  {args.min_family_size}")
    print(f"  threshold:        {args.threshold}")
    print("=" * 70)

    acc_to_family = load_family_meta(args.family_meta)
    acc_to_split  = load_split_meta(meta_tsv)

    print(f"\nLoaded family labels for {len(acc_to_family):,} accessions")
    print(f"Loaded split labels  for {len(acc_to_split):,} viral accessions")

    # Determine eligible families — count only accessions that are actually in
    # the probe dataset (i.e. passed quality filters and were embedded).
    # acc_to_split contains exactly those viral accessions.
    from collections import Counter
    family_counts = Counter(
        acc_to_family[acc] for acc in acc_to_split
        if acc in acc_to_family
    )
    eligible = sorted(
        fam for fam, cnt in family_counts.items()
        if fam != "Unknown" and cnt >= args.min_family_size
    )
    print(f"\nEligible families (≥{args.min_family_size} seqs): {len(eligible)}")
    for fam in eligible:
        print(f"  {fam:<35}  {family_counts[fam]:>5}")

    all_results: dict = {}
    results_root = os.path.join(args.project_dir, "results")

    for model_key in MODEL_KEYS:
        print(f"\n{'='*70}")
        print(f"Model: {model_key}")
        print(f"{'='*70}")

        model_res = run_model(
            model_key, args.project_dir,
            acc_to_family, acc_to_split,
            eligible, args.min_family_size, args.threshold,
        )
        if model_res is None:
            continue

        result = {
            "model":           model_key,
            "min_family_size": args.min_family_size,
            "threshold":       args.threshold,
            "families":        model_res,
        }
        all_results[model_key] = result

        out_dir  = os.path.join(results_root, model_key)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "leave_family_out_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [{model_key}] Written: {out_path}")

    # ---- Summary ----
    summary: dict = {
        "min_family_size": args.min_family_size,
        "threshold":       args.threshold,
        "families":        eligible,
        "models":          {},
    }
    for mk, res in all_results.items():
        summary["models"][mk] = {
            fam: {
                "auc_roc":     fd["auc_roc"],
                "sensitivity": fd["sensitivity"],
                "n_test":      fd["n_test_viral"],
            }
            for fam, fd in res["families"].items()
        }

    summary_path = os.path.join(results_root, "leave_family_out_summary.json")
    os.makedirs(results_root, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written: {summary_path}")

    # ---- Print summary table ----
    if all_results and eligible:
        header_models = [mk for mk in MODEL_KEYS if mk in all_results]
        col_w = 10
        print(f"\n{'Family':<35}", end="")
        for mk in header_models:
            print(f"  {mk[:col_w]:>{col_w}}", end="")
        print()
        print("-" * (35 + (col_w + 2) * len(header_models)))

        for fam in eligible:
            print(f"  {fam:<33}", end="")
            for mk in header_models:
                val = all_results.get(mk, {}).get("families", {}).get(fam, {}).get("auc_roc")
                s = f"{val:.3f}" if val is not None else "  N/A"
                print(f"  {s:>{col_w}}", end="")
            print()


if __name__ == "__main__":
    main()
